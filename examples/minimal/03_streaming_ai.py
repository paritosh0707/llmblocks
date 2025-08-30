#!/usr/bin/env python3
"""
🌊 Streaming AI - Just 4 lines!

Real-time AI responses that stream as they're generated.
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
    # Setup
    api_key = os.getenv('GOOGLE_API_KEY', 'your-api-key-here')
    
    print("🌊 Streaming AI - Watch the magic happen in real-time!")
    print("Just 4 lines of code for streaming AI!\n")
    
    if api_key == 'your-api-key-here':
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    # ✨ THE MAGIC - Only 4 lines for streaming AI!
    from llmblocks.blocks.llm_provider import get_provider
    provider = await get_provider("gemini", api_key=api_key)
    print("🤖 AI: ", end="", flush=True)
    async for chunk in provider.generate_stream("Tell me a short story about a robot learning to code"):
        print(chunk.content, end="", flush=True)
    
    print("\n\n✅ Real-time streaming with just 4 lines!")

if __name__ == "__main__":
    asyncio.run(main())
