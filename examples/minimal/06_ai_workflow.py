#!/usr/bin/env python3
"""
⚡ AI Workflow - Just 7 lines!

Create a complete AI workflow with LangGraph.
"""
import asyncio
import sys
import os
sys.path.insert(0, 'src')

async def main():
    # Setup
    os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-api-key-here')
    
    print("⚡ AI Workflow - Complete workflow in 7 lines!")
    print("Building a LangGraph workflow...\n")
    
    try:
        # ✨ THE MAGIC - Only 7 lines for a complete AI workflow!
        from llmblocks.blocks.llm_provider import get_provider
        provider = get_provider("gemini", api_key=os.environ['GOOGLE_API_KEY'])
        llm_node = provider.create_langgraph_node("assistant")  # Create workflow node
        graph = provider.create_langgraph_graph("ai_workflow")  # Create complete graph
        
        # Run the workflow
        result = await llm_node({"messages": [{"role": "user", "content": "Create a haiku about coding"}]})
        
        print(f"🤖 Workflow Result: {result['messages'][0].content}")
        print(f"\n✅ Complete AI workflow with just 7 lines!")
        
    except ImportError:
        print("⚠️  LangGraph not installed. Run: pip install langgraph")
    except Exception as e:
        print(f"⚠️  Error: {e}")
        print("💡 This might be due to LangGraph version compatibility")

if __name__ == "__main__":
    asyncio.run(main())
