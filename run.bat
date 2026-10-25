@echo off
rem Football Telestrator - first run creates a Python environment and installs the requirements (~70 MB download).
cd /d "%~dp0"
if not exist .venv (
  python -m venv .venv || (echo Python 3.9+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\pip install -r requirements.txt
)
start "" http://localhost:8000
.venv\Scripts\python server\app.py
