#!/usr/bin/env python3
"""
🔧 Setup Checker - Verify your LLMBlocks installation

Run this first to check if everything is set up correctly.
"""
import sys
import os

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

def check_setup():
    print("🔧 LLMBlocks Setup Checker")
    print("=" * 40)
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 11):
        print("❌ Python 3.11+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check if we're in the right directory
    if os.path.exists('src/llmblocks'):
        print("✅ LLMBlocks source found")
    else:
        print("❌ LLMBlocks source not found. Are you in the project directory?")
        return False
    
    # Check dependencies
    missing_deps = []
    
    try:
        import pydantic
        print("✅ Pydantic available")
    except ImportError:
        missing_deps.append("pydantic")
    
    try:
        import langchain
        print("✅ LangChain available")
    except ImportError:
        missing_deps.append("langchain")
    
    try:
        import langgraph
        print("✅ LangGraph available")
    except ImportError:
        missing_deps.append("langgraph")
    
    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if api_key and len(api_key) > 10:
        print("✅ GOOGLE_API_KEY found")
    else:
        print("⚠️  GOOGLE_API_KEY not set or invalid")
        print("💡 Get your free key: https://aistudio.google.com/app/apikey")
        print("💡 Set it: export GOOGLE_API_KEY='your-key-here'")
    
    # Summary
    print("\n" + "=" * 40)
    
    if missing_deps:
        print("❌ Missing dependencies:", ", ".join(missing_deps))
        print("💡 Install with: uv sync")
        print("💡 Or: pip install -e '.[dev]'")
        return False
    else:
        print("✅ All dependencies available!")
        
    if not api_key:
        print("⚠️  Set up API key to run examples")
        return False
    else:
        print("🚀 Ready to run LLMBlocks examples!")
        return True

if __name__ == "__main__":
    success = check_setup()
    
    if success:
        print("\n🎯 Try these examples:")
        print("python examples/minimal/01_hello_world.py")
        print("python examples/minimal/02_smart_chatbot.py")
    else:
        print("\n🔧 Please fix the issues above first")
    
    sys.exit(0 if success else 1)
