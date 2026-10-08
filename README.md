# langchain-practice

Hands-on LangChain practice, from a plain model call to a real tool-calling agent. Same format as [`k8s-practice`](https://github.com/NikanEidi/k8s-practice): small parts, real code, a short note per part explaining what it does and why.

## Setup

- Python 3, [Ollama](https://ollama.com) running locally with `llama3.2:3b` pulled
- `python3 -m venv venv && source venv/bin/activate`
- `pip install langchain langchain-ollama`

## Structure

```
langchain-practice/
├── README.md
├── code/     — the actual scripts, one per part
└── notes/    — one note per part: what it does, why, key lines explained
```

## Parts

| # | Part | Code | Note |
|---|---|---|---|
| 1 | First model call | [`code/01_first_call.py`](code/01_first_call.py) | [`notes/01-first-call.md`](notes/01-first-call.md) |
| 2 | System prompt | [`code/02_system_prompt.py`](code/02_system_prompt.py) | [`notes/02-system-prompt.md`](notes/02-system-prompt.md) |

More parts get added as we go: tool calling, a multi-step agent loop, then a real project (a Code Review Agent).
