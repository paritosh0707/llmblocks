#!/usr/bin/env python3
"""
🔄 Multi-Provider AI - Just 6 lines!

Switch between different AI providers seamlessly.
"""
import asyncio
import sys
import os
sys.path.insert(0, 'src')

async def main():
    # Setup (you can use any provider you have API keys for)
    os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-google-key')
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-openai-key')
    
    print("🔄 Multi-Provider AI - Switch providers with ease!")
    print("Just 6 lines to use multiple AI providers!\n")
    
    # ✨ THE MAGIC - Only 6 lines for multi-provider AI!
    from llmblocks.blocks.llm_provider import get_provider
    
    providers = [
        ("Gemini", get_provider("gemini", api_key=os.environ.get('GOOGLE_API_KEY'))),
        ("OpenAI", get_provider("openai", api_key=os.environ.get('OPENAI_API_KEY'))),
    ]
    
    question = "What's the meaning of life in one sentence?"
    
    for name, provider in providers:
        if provider:  # Only use if API key is available
            try:
                response = await provider.generate(question)
                print(f"🤖 {name}: {response.content}\n")
            except Exception as e:
                print(f"⚠️  {name}: API key not configured or error: {e}\n")
    
    print("✅ Multiple AI providers with just 6 lines!")

if __name__ == "__main__":
    asyncio.run(main())
