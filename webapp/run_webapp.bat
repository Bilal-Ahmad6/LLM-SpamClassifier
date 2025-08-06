@echo off
REM Activate the virtual environment and run the Flask app
cd /d "%~dp0"
call ..\.venv\Scripts\activate.bat
python app.py
pause
