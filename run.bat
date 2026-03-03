@echo off
REM Gradio has issues with Python 3.13 - use 3.10, 3.11, or 3.12
echo Checking for Python 3.10/3.11/3.12...
py -3.10 --version 2>nul && goto run310
py -3.11 --version 2>nul && goto run311
py -3.12 --version 2>nul && goto run312
echo.
echo Python 3.10, 3.11, or 3.12 not found.
echo Gradio does NOT work with Python 3.13.
echo Install Python 3.10 from https://www.python.org/downloads/
pause
exit /b 1

:run310
echo Using Python 3.10
py -3.10 -m pip install -r requirements.txt -q
py -3.10 app.py
goto end

:run311
echo Using Python 3.11
py -3.11 -m pip install -r requirements.txt -q
py -3.11 app.py
goto end

:run312
echo Using Python 3.12
py -3.12 -m pip install -r requirements.txt -q
py -3.12 app.py
goto end

:end
pause
