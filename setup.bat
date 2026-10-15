@echo off
REM One-time setup for Windows. Double-click or run:  setup.bat
cd /d "%~dp0"
echo Creating virtual environment in .venv ...
python -m venv .venv || (echo Python not found. Install Python 3.10+ from python.org and tick "Add to PATH". & pause & exit /b 1)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if not exist .env copy .env.example .env
echo.
echo Setup complete! Try:
echo    .venv\Scripts\activate
echo    (PowerShell error "running scripts is disabled"? run once: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
echo    python lessons\01_why_rag.py
echo    python app.py
echo    streamlit run web_app.py
pause
