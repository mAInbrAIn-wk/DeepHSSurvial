---
created: 2026-09-06
last_updated: 2026-09-06
status: active
tags: [environment, python, interpreter, policy]
---

# Python Execution Rule for Antigravity

Whenever executing Python scripts or commands in this workspace:

1. **Interpreter Path:** Always use `C:\GitHub_public\.venv\Scripts\python.exe`.
2. **Never use System Python:** Do NOT use `python` from the default PATH or `C:\Users\wilfr\AppData\Local\Programs\Python\...`. The Windows Application Control Policy blocks scientific native binaries (e.g. Scipy optimization DLLs like `_highspy._core`) when run through the unapproved system Python interpreter. The `.venv` at `C:\GitHub_public\.venv` is whitelisted.
3. **Environment Variable:** Always set `$env:PYTHONPATH = "src"` before running.
4. **Example Command:**
   ```powershell
   $env:PYTHONPATH = "src"; C:\GitHub_public\.venv\Scripts\python.exe path/to/script.py
   ```
