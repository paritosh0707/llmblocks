#!/usr/bin/env python3
"""
🚀 LLMBlocks Hello World - Just 3 lines!

The simplest possible LLM interaction with LLMBlocks.
"""
import asyncio
import sys
import os
sys.path.insert(0, 'src')

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

async def main():
    # Set your API key (get free key from https://aistudio.google.com/app/apikey)
    api_key = os.getenv('GOOGLE_API_KEY', 'your-api-key-here')
    
    print("🚀 LLMBlocks Hello World - Just 3 lines!\n")
    
    # Check setup
    if api_key == 'your-api-key-here':
        print("❌ Please set GOOGLE_API_KEY environment variable")
        print("💡 Get your free key: https://aistudio.google.com/app/apikey")
        print("💡 Run: export GOOGLE_API_KEY='your-key-here'")
        return
    
    try:
        # ✨ THE MAGIC - Only 3 lines needed!
        from llmblocks.blocks.llm_provider import get_provider
        provider = await get_provider("gemini", api_key=api_key)
        response = await provider.generate("Hello! Introduce yourself in one sentence.")
        
        print(f"🤖 AI Response: {response.content}")
        print(f"\n✅ That's it! Just 3 lines of code for AI power!")
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: uv sync")
        print("💡 Or run: python examples/minimal/00_check_setup.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your API key and internet connection")

if __name__ == "__main__":
    asyncio.run(main())
