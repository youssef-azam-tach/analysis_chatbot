@echo off
REM Data Analysis & AI Chatbot - Startup Script for Windows

echo 🚀 Starting Data Analysis & AI Chatbot Platform...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.10+
    pause
    exit /b 1
)

REM Create data directory if it doesn't exist
if not exist "data" mkdir data

REM Check if requirements are installed
echo 📦 Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo 📥 Installing dependencies...
    pip install -r requirements.txt
)

REM Start Streamlit
echo.
echo ✅ Starting Streamlit application...
echo 📱 Open your browser at: http://localhost:8501
echo.

cd ui\streamlit
streamlit run app.py

pause
