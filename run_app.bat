@echo off
REM Starts the AgriShield website. Open the "Local URL" it prints (usually http://localhost:8501).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe ( echo Run setup_windows.bat first. & pause & exit /b 1 )
call .venv\Scripts\activate.bat
python -m streamlit run app.py --browser.gatherUsageStats false
pause
