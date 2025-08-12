# Gomoku-StudentLLMAgent

An **LLM-based Gomoku agent** designed for the [Gomoku AI Framework](https://github.com/sitfoxfly/gomoku-ai-agent-example/).  
This agent uses a large language model to play Gomoku (8×8 board, five-in-a-row wins) with **strict JSON output** and **legal move constraints**.

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


