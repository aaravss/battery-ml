@echo off
cd /d "%~dp0"
echo Exporting the trained model as a Keras file (no retraining)...
".venv\Scripts\python.exe" "tools\export_keras.py"
echo.
pause
