@echo off
REM OPTIONAL: retrains everything from the datasets (see README "Training"). Needs the datasets in the data\ folder.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m scripts.run_pipeline --train
pause
