# Gomoku-StudentLLMAgent

A lightweight, LLM-steered Gomoku (8×8) agent that combines minimal code heuristics with prompted decision rules to pick moves. It favors central control, detects immediate tactical threats/wins for prompt hints, and robustly parses the model’s JSON move with a safe fallback.



## 📌 Features
- Uses `google/gemma-2-9b-it` model
- Non–code-based strategy: engine does not auto-place winning/blocking moves — it prompts the LLM to do so
- Detects immediate win (for either side) and open-three threats to guide prompts
- Center-first bias for the opening and as a final fallback
- Strict one-line JSON output contract with lenient parsing + safe fallback

## 🤖 Key functions
- get_max_chain_head(board, p): Counts the longest consecutive chain for player p in four directions, starting only from chain heads.
- has_open_three(board, p): Returns True if the  open three exists anywhere (H/V/diagonals).
- find_immediate_win(board, legal_moves, p): Returns a coordinate (r, c) if playing there creates 5 in a row (H/V/diagonals). Used for hints only.
- sorted_moves_center_first(game_state): Returns all legal moves sorted by closeness to center (used in prompt & fallback).
- get_move(game_state): Orchestrates analysis → prompt → LLM → parsing → fallback to produce a final move.

## 📂 Repository Structure
- my_example.py 
- agent.json


