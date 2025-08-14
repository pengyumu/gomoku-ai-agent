import re
import json
import random
from typing import Tuple, List, Optional

from gomoku.agents.base import Agent
from gomoku.core.models import GameState
from gomoku.llm.openai_client import OpenAIGomokuClient


class StudentLLMAgent(Agent):
    """
    Deterministic-but-robust Gomoku LLM agent.
    - Fixes: falling back too often; fragile JSON; center-biased fallback; invisible failures.
    - Adds: debug logging, retry-on-bad-output, optional tactical pre-move (auto-disable if board unreadable).
    """

    DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

    def __init__(
        self,
        agent_id: str,
        *,
        model: Optional[str] = None,
        debug: bool = False,
        use_tactical_premove: bool = True,
        fallback_mode: str = "first"  # "first" or "random"
    ):
        super().__init__(agent_id)
        self._model = model or "deepseek/deepseek-r1-0528-qwen3-8b"
        self.DEBUG = debug
        self.use_tactical_premove = use_tactical_premove
        self.fallback_mode = fallback_mode

    # ---------- lifecycle ----------

    def _setup(self):
        self.system_prompt = self._create_system_prompt()
        # NOTE: ensure your API key / base_url are configured per OpenAIGomokuClient docs
        self.llm_client = OpenAIGomokuClient(model=self._model)

    # ---------- utils ----------

    def _log(self, *a):
        if self.DEBUG:
            print(f"[{self.__class__.__name__}]", *a)

    def _create_system_prompt(self) -> str:
        return (
            "You are a master-level Gomoku AI for an 8x8 board (0-indexed).\n"
            "Goal: place five consecutive stones in any direction. Never play on occupied cells.\n\n"
            "OUTPUT FORMAT (STRICT): Output ONLY one JSON object -> {\"row\": <int>, \"col\": <int>}.\n"
            "No prose, no markdown, no extra text. The move MUST be one of LEGAL_MOVES.\n\n"
            "PRIORITY RULES (apply in order; choosing a lower rule when a higher one exists is an error):\n"
            "1) WIN NOW: if any move makes five-in-a-row for YOU, you MUST choose it (pick the earliest in LEGAL_MOVES).\n"
            "2) BLOCK LOSS: if the opponent can win next move with any single placement, you MUST block it unless you can win now.\n"
            "3) CREATE OPEN FOUR (forcing) if safe.\n"
            "4) CREATE DOUBLE OPEN THREE (fork) if safe.\n"
            "5) EXTEND strongest chains (open threes / broken fours).\n"
            "6) CENTER & FLEXIBILITY when above do not apply.\n"
            "Tie-breaker: choose the earliest move in LEGAL_MOVES.\n"
            "Before output, re-check (row,col) is in LEGAL_MOVES; if not, scan LEGAL_MOVES in order and pick the first satisfying the highest rule you found.\n"
        )

    def _build_user_prompt(self, game_state: GameState, legal_moves: List[Tuple[int, int]]) -> str:
        board_str = game_state.format_board("standard")
        you = game_state.current_player.value  # 'X' or 'O'
        opp = 'O' if you == 'X' else 'X'
        legal_list = [[r, c] for (r, c) in legal_moves]
        return (
            f"### CURRENT GAME STATE ###\n"
            f"BOARD (8x8, 0-indexed):\n{board_str}\n\n"
            f"You play as: {you}\n"
            f"Opponent: {opp}\n\n"
            f"LEGAL_MOVES (row, col): {legal_list}\n\n"
            f"Choose your move according to PRIORITY RULES. Output JSON only."
        )

    # ---------- robust JSON ----------

    def _safe_extract_json(self, text) -> Optional[dict]:
        """Try hard to get a single JSON object {row:int, col:int} from model output."""
        if isinstance(text, dict):
            return text

        s = (text or "").strip()

        # 1) direct parse
        try:
            return json.loads(s)
        except Exception:
            pass

        # 2) first JSON object (non-greedy)
        m = re.search(r"\{.*?\}", s, re.DOTALL)
        if not m:
            return None
        block = m.group(0)

        # 3) common repairs
        block = re.sub(r",\s*}", "}", block)  # trailing comma
        # convert floats like 3.0 -> 3 only inside numbers followed by non-digit
        block = re.sub(r'(\D)(\d+)\.0(\D)', r'\1\2\3', " " + block + " ")
        block = block.strip()

        try:
            return json.loads(block)
        except Exception:
            return None

    # ---------- fallback (unbiased) ----------

    def _fallback_move(self, game_state: GameState) -> Tuple[int, int]:
        legal = game_state.get_legal_moves()
        if not legal:
            center = game_state.board_size // 2
            return (center, center)
        if self.fallback_mode == "random":
            return random.choice(legal)
        return legal[0]  # deterministic但无中心偏置

    # ---------- optional tactical pre-move ----------

    def _try_get_cell(self, gs: GameState, r: int, c: int) -> Optional[str]:
        """Return 'X','O','.' if discoverable; else None. Works with common fields, else parse."""
        # Common internal fields
        for name in ("board", "grid", "state", "cells", "matrix"):
            b = getattr(gs, name, None)
            if b is not None:
                try:
                    v = b[r][c]
                    if v in ('X', 'O', '.'):
                        return v
                    # numeric encodings (guess): 0 empty, 1 X, 2 O
                    if v in (0, 1, 2):
                        return {0: '.', 1: 'X', 2: 'O'}[v]
                except Exception:
                    pass
        # Try parse formatted board (last resort; may be fragile)
        try:
            s = gs.format_board("standard")
            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            # Heuristic: pick last n lines that look like rows
            n = gs.board_size
            rows = lines[-n:]
            row = rows[r]
            symbols = [ch for ch in row if ch in ('.', 'X', 'O')]
            if len(symbols) >= n:
                return symbols[c]
        except Exception:
            pass
        return None

    def _count_dir(self, gs: GameState, r: int, c: int, dr: int, dc: int, me: str) -> int:
        n = gs.board_size
        rr, cc = r + dr, c + dc
        cnt = 0
        while 0 <= rr < n and 0 <= cc < n:
            cell = self._try_get_cell(gs, rr, cc)
            if cell != me:
                break
            cnt += 1
            rr += dr
            cc += dc
        return cnt

    def _is_five_if_play(self, gs: GameState, r: int, c: int, me: str) -> bool:
        cell = self._try_get_cell(gs, r, c)
        if cell is None or cell != '.':
            return False
        for dr, dc in self.DIRECTIONS:
            a = self._count_dir(gs, r, c, -dr, -dc, me)
            b = self._count_dir(gs, r, c,  dr,  dc, me)
            if a + 1 + b >= 5:
                return True
        return False

    def _opponent_immediate_wins(self, gs: GameState, legal_moves: List[Tuple[int, int]], opp: str):
        out = []
        for r, c in legal_moves:
            if self._is_five_if_play(gs, r, c, opp):
                out.append((r, c))
        return out

    def _open_four_if_play(self, gs: GameState, r: int, c: int, me: str) -> bool:
        cell = self._try_get_cell(gs, r, c)
        if cell is None or cell != '.':
            return False
        n = gs.board_size
        for dr, dc in self.DIRECTIONS:
            a = self._count_dir(gs, r, c, -dr, -dc, me)
            b = self._count_dir(gs, r, c,  dr,  dc, me)
            total = a + 1 + b
            if total == 4:
                end1 = (r - (a + 1) * dr, c - (a + 1) * dc)
                end2 = (r + (b + 1) * dr, c + (b + 1) * dc)
                def open_end(rr, cc):
                    return 0 <= rr < n and 0 <= cc < n and self._try_get_cell(gs, rr, cc) == '.'
                if open_end(*end1) or open_end(*end2):
                    return True
        return False

    def _double_open_three_if_play(self, gs: GameState, r: int, c: int, me: str) -> bool:
        cell = self._try_get_cell(gs, r, c)
        if cell is None or cell != '.':
            return False
        n = gs.board_size
        dirs = 0
        for dr, dc in self.DIRECTIONS:
            a = self._count_dir(gs, r, c, -dr, -dc, me)
            b = self._count_dir(gs, r, c,  dr,  dc, me)
            total = a + 1 + b
            if total == 3:
                end1 = (r - (a + 1) * dr, c - (a + 1) * dc)
                end2 = (r + (b + 1) * dr, c + (b + 1) * dc)
                def open_end(rr, cc):
                    return 0 <= rr < n and 0 <= cc < n and self._try_get_cell(gs, rr, cc) == '.'
                if open_end(*end1) and open_end(*end2):
                    dirs += 1
        return dirs >= 2

    def _tactical_premove(self, gs: GameState, legal_moves: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Return a high-priority tactical move if available; else None."""
        if not self.use_tactical_premove:
            return None

        me = gs.current_player.value
        opp = 'O' if me == 'X' else 'X'

        # If we cannot read the board reliably, skip and log once.
        probe = self._try_get_cell(gs, 0, 0)
        if probe is None:
            self._log("Board unreadable → skip tactical premove")
            return None

        # 1) win now
        winners = [(r, c) for (r, c) in legal_moves if self._is_five_if_play(gs, r, c, me)]
        if winners:
            self._log("Tactical: WIN NOW", winners[0])
            return winners[0]

        # 2) block opponent immediate win
        opp_wins = self._opponent_immediate_wins(gs, legal_moves, opp)
        if opp_wins:
            self._log("Tactical: BLOCK LOSS", opp_wins[0])
            return opp_wins[0]

        # 3) create open four
        for r, c in legal_moves:
            if self._open_four_if_play(gs, r, c, me):
                self._log("Tactical: OPEN FOUR", (r, c))
                return (r, c)

        # 4) double open three
        for r, c in legal_moves:
            if self._double_open_three_if_play(gs, r, c, me):
                self._log("Tactical: DOUBLE THREE", (r, c))
                return (r, c)

        return None

    # ---------- main ----------

    async def get_move(self, game_state: GameState) -> Tuple[int, int]:
        legal_moves = game_state.get_legal_moves()
        if not legal_moves:
            self._log("No legal moves → fallback")
            return self._fallback_move(game_state)

        # Tactical premove (hard rules)
        premove = self._tactical_premove(game_state, legal_moves)
        if premove is not None:
            return premove

        if not hasattr(self, "llm_client") or self.llm_client is None:
            self._log("llm_client missing → fallback")
            return self._fallback_move(game_state)

        user_prompt = self._build_user_prompt(game_state, legal_moves)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Two attempts before fallback
        for attempt in range(2):
            try:
                self._log(f"Calling LLM (attempt {attempt+1})…")
                response = await self.llm_client.complete(
                    messages,
                    temperature=0.0,   # deterministic; raise to 0.2 if you want diversity
                    top_p=0.9,
                    max_tokens=128,
                )
                self._log("LLM raw:", repr(response)[:180])

                data = self._safe_extract_json(response) or {}
                r, c = data.get("row"), data.get("col")

                # Some models output floats/strings
                if isinstance(r, (float, str)) and str(r).replace('.', '', 1).isdigit():
                    r = int(float(r))
                if isinstance(c, (float, str)) and str(c).replace('.', '', 1).isdigit():
                    c = int(float(c))

                move = (r, c) if isinstance(r, int) and isinstance(c, int) else None
                if move and move in set(legal_moves):
                    self._log("LLM move →", move)
                    return move

                self._log("Invalid LLM move:", data, "→ retry" if attempt == 0 else "→ fallback")

            except Exception as e:
                self._log("LLM exception:", e, "→ retry" if attempt == 0 else "→ fallback")

        # Final fallback
        return self._fallback_move(game_state)
