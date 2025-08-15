# Gomoku-StudentLLMAgent

A lightweight, LLM-steered Gomoku (8×8) agent that combines minimal code heuristics with prompted decision rules to pick moves. It favors central control, detects immediate tactical threats/wins for prompt hints, and robustly parses the model’s JSON move with a safe fallback.




## 📌 Features
- Uses `qwen3-8b` model via Groq API
- Follows strict move selection priority:
  1. Win in 1 move
  2. Block opponent’s win in 1 move
  3. Create immediate threats (open four)
  4. Create dual threats (open threes)
  5. Extend/bridge longest chain
  6. Prefer central & flexible positions
  7. Fallback to earliest legal move
- Outputs only valid moves from `LEGAL_MOVES`
- Robust JSON parsing with fallback strategy

## 📂 Repository Structure
- my_example.py 
- agent.json


