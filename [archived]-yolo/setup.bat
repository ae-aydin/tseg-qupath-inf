@echo off

echo Setting up Python virtual environment and creating folders...
echo Checking Python 3...
where python >nul 2>nul
if %errorlevel% neq 0 goto PythonNotInstalled
goto contScript

:PythonNotInstalled
echo WARNING: Python 3 not found.
echo Closing window in 5 seconds...
timeout /t 5 >nul
exit

:contScript
echo Python 3 found.
echo Creating virtual environment...
python -m venv .venv
echo Virtual environment successfully created.
call .venv\Scripts\activate
echo Installing requirements...
pip install -r requirements.txt
echo Requirements successfully installed.
call deactivate
echo Creating folders...
mkdir models .roi_tiles .preds
echo Folders successfully created.
echo 
echo Setup completed successfully. Closing window in 5 seconds...
timeout /t 5 >nul
exit
