#!/bin/bash

# Data Analysis & AI Chatbot - Startup Script

echo "🚀 Starting Data Analysis & AI Chatbot Platform..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+"
    exit 1
fi

# Check if Ollama is running
echo "🔍 Checking Ollama connection..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running!"
    echo "Please start Ollama with: ollama serve"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create data directory if it doesn't exist
mkdir -p data

# Check if requirements are installed
echo "📦 Checking dependencies..."
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📥 Installing dependencies..."
    pip install -r requirements.txt
fi

# Start Streamlit
echo ""
echo "✅ Starting Streamlit application..."
echo "📱 Open your browser at: http://localhost:8501"
echo ""

cd ui/streamlit
streamlit run app.py
