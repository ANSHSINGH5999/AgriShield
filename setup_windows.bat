@echo off
REM One-time setup: creates .venv and installs the pinned dependencies (about 1-3 GB download, mostly PyTorch).
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3.12) else (set PY=python)
%PY% --version
if errorlevel 1 (
  echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
REM a .venv copied from another computer (e.g. a Mac) does not work on Windows - recreate it
if exist .venv if not exist .venv\Scripts\python.exe (
  echo Removing a .venv that was created on another operating system ...
  rmdir /s /q .venv
)
if not exist .venv\Scripts\python.exe (
  %PY% -m venv .venv
  if errorlevel 1 ( echo Could not create the virtual environment. & pause & exit /b 1 )
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 ( echo Dependency installation failed - see README "Troubleshooting". & pause & exit /b 1 )
python -m pytest -q tests
echo.
echo Setup finished. Start the website with run_app.bat
pause
