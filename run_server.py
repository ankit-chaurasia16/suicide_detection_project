#!/usr/bin/env python3
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    print("Starting Suicide Detection AI Server...")
    print("Server will be available at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='127.0.0.1', port=5000)
except Exception as e:
    print(f"Error starting server: {e}")
    input("Press Enter to exit...")