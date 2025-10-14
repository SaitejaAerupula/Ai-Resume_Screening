@echo off
:: ============================================================
:: AI-POWERED RESUME SCREENER - BATCH LAUNCHER
:: Automated setup and execution script for Windows
:: ============================================================

echo.
echo ============================================================
echo AI-POWERED RESUME SCREENER
echo Intelligent Candidate Evaluation System
echo ============================================================
echo.

:: Set the script directory as working directory
cd /d "%~dp0"

:: Check if Python is installed
echo [INFO] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH!
    echo [INFO] Please install Python 3.8+ from https://python.org
    echo [INFO] Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

:: Display Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [SUCCESS] Python %PYTHON_VERSION% detected

:: Check if virtual environment exists
if not exist "venv" (
    echo.
    echo [INFO] Virtual environment not found. Creating new environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created
)

:: Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [SUCCESS] Virtual environment activated

:: Check if requirements are installed
echo [INFO] Checking dependencies...
pip show transformers >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing required packages...
    echo [WARNING] This may take several minutes for first-time setup...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies!
        echo [INFO] Check your internet connection and try again
        pause
        exit /b 1
    )
    echo [SUCCESS] All dependencies installed
) else (
    echo [SUCCESS] Dependencies already installed
)

:: Check if .env file exists
if not exist ".env" (
    echo [INFO] Creating environment configuration file...
    copy .env.example .env >nul 2>&1
    echo [SUCCESS] Environment file created (.env)
)

:: Perform syntax check
echo [INFO] Performing syntax validation...
python -c "import main; print('[SUCCESS] Syntax check passed')" 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Syntax validation failed!
    echo [INFO] Please check the main.py file for errors
    pause
    exit /b 1
)

:: Check available disk space (for model downloads)
echo [INFO] Checking system resources...
for /f "tokens=3" %%a in ('dir /-c "%cd%" ^| find "bytes free"') do set FREE_SPACE=%%a
echo [INFO] Available disk space: %FREE_SPACE% bytes

:: Display system information
echo.
echo ============================================================
echo SYSTEM READY - STARTING AI RESUME SCREENER
echo ============================================================
echo [INFO] Python Version: %PYTHON_VERSION%
echo [INFO] Working Directory: %cd%
echo [INFO] Virtual Environment: ACTIVE
echo [INFO] Dependencies: INSTALLED
echo [INFO] Configuration: READY
echo.
echo [WARNING] First run will download AI models (~2GB)
echo [WARNING] This may take 5-10 minutes depending on internet speed
echo.
echo [INFO] The application will open in your default web browser
echo [INFO] Default URL: http://127.0.0.1:7860
echo.
echo Press Ctrl+C to stop the application
echo ============================================================
echo.

:: Start the application
echo [INFO] Launching AI Resume Screener...
python main.py

:: Handle exit
echo.
echo ============================================================
echo APPLICATION STOPPED
echo ============================================================
echo [INFO] AI Resume Screener has been stopped
echo [INFO] Virtual environment is still active
echo [INFO] You can run this script again to restart
echo.
pause