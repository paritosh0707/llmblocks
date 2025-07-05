#!/usr/bin/env python3
"""
Launcher script for LLMBlocks RAG Playground.

This script sets up the Python path and launches the Streamlit playground.
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    """Launch the RAG playground."""
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    # Add the project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Set environment variables
    os.environ['PYTHONPATH'] = str(project_root) + os.pathsep + os.environ.get('PYTHONPATH', '')
    
    # Path to the playground
    playground_path = project_root / "llmblocks" / "playground" / "rag_playground.py"
    
    if not playground_path.exists():
        print(f"❌ Playground not found at: {playground_path}")
        print("Please ensure the playground file exists.")
        return 1
    
    print("🧩 Launching LLMBlocks RAG Playground...")
    print(f"📍 Project root: {project_root}")
    print(f"🎮 Playground: {playground_path}")
    print()
    print("💡 Tips:")
    print("   - Make sure you have Streamlit installed: pip install streamlit")
    print("   - Set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
    print("   - The playground will open in your browser automatically")
    print()
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(playground_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching playground: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Playground stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 