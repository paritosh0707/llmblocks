#!/usr/bin/env python3
"""
⚡ Zero Config - Just 2 lines!

The absolute minimum code needed for AI - assumes GOOGLE_API_KEY is in environment.
"""
import asyncio
import sys
import os
sys.path.insert(0, 'src')

async def main():
    print("⚡ Zero Config AI - The absolute minimum!")
    print("Assumes GOOGLE_API_KEY is set in your environment\n")
    
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ Please set GOOGLE_API_KEY environment variable")
        print("💡 Run: export GOOGLE_API_KEY='your-api-key'")
        return
    
    # ✨ THE MAGIC - Only 2 lines for AI!
    from llmblocks.blocks.llm_provider import get_provider
    print(f"🤖 AI: {(await get_provider('gemini').generate('Say hello!')).content}")
    
    print(f"\n✅ That's it! Just 2 lines of code!")

if __name__ == "__main__":
    asyncio.run(main())
