#!/usr/bin/env python3
"""
🤖 Smart Chatbot - Just 5 lines!

A fully functional chatbot with conversation memory.
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
    
    print("🤖 Smart Chatbot - Just 5 lines of core code!")
    print("Type 'quit' to exit\n")
    
    if api_key == 'your-api-key-here':
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    # ✨ THE MAGIC - Only 5 lines for a smart chatbot!
    from llmblocks.blocks.llm_provider import get_provider
    provider = await get_provider("gemini", api_key=api_key)
    messages = []  # Simple conversation memory
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        # Add user message and generate response
        messages.append({"role": "user", "content": user_input})
        response = await provider.generate(messages)
        messages.append({"role": "assistant", "content": response.content})
        
        print(f"🤖 Bot: {response.content}\n")
    
    print("✅ Goodbye! That was just 5 lines of core logic!")

if __name__ == "__main__":
    asyncio.run(main())
