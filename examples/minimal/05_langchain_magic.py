#!/usr/bin/env python3
"""
🔗 LangChain Magic - Just 3 lines!

Use LLMBlocks providers directly in LangChain chains.
"""
import asyncio
import sys
import os
sys.path.insert(0, 'src')

async def main():
    # Setup
    os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-api-key-here')
    
    print("🔗 LangChain Magic - LLMBlocks + LangChain = ❤️")
    print("Just 3 lines to integrate with LangChain!\n")
    
    try:
        # ✨ THE MAGIC - Only 3 lines for LangChain integration!
        from llmblocks.blocks.llm_provider import get_provider
        provider = get_provider("gemini", api_key=os.environ['GOOGLE_API_KEY'])
        langchain_llm = provider.as_langchain()  # Convert to LangChain LLM
        
        # Now use it like any LangChain LLM
        from langchain.schema import HumanMessage
        messages = [HumanMessage(content="Explain quantum computing in simple terms")]
        result = langchain_llm.invoke(messages)
        
        print(f"🤖 LangChain Result: {result.content}")
        print(f"\n✅ LangChain integration with just 3 lines!")
        
    except ImportError:
        print("⚠️  LangChain not installed. Run: pip install langchain")
    except Exception as e:
        print(f"⚠️  Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
