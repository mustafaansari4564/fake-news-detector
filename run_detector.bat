@echo off
echo ==============================================
echo 🤖 Starting AI Fake News Detector...
echo ==============================================

cd /d "C:\Users\DELL\.gemini\antigravity\scratch\fake-news-detector"
call .\venv\Scripts\activate.bat
python test.py

pause
