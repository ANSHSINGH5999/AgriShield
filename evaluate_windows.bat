@echo off
REM OPTIONAL: re-runs the evaluation on the locked test set and PlantDoc. Needs the datasets in the data\ folder.
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python -m scripts.run_pipeline --evaluate
pause
