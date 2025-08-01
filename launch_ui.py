#!/usr/bin/env python3
"""
Simple launcher for the RAG Chatbot Streamlit UI
Can be run from the project root directory
"""

import subprocess
import sys
from pathlib import Path

def main():
    # Get the project root directory
    script_dir = Path(__file__).parent
    streamlit_app = script_dir / "src" / "ui" / "streamlit_app.py"
    
    if not streamlit_app.exists():
        print(f"Error: Streamlit app not found at {streamlit_app}")
        sys.exit(1)
    
    # Run streamlit
    try:
        print("Starting RAG Chatbot Streamlit UI...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(streamlit_app)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()
