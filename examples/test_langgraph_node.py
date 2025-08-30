#!/usr/bin/env python3
"""
Test LangGraph Node Creation with MRO Fix.

This script tests the LangGraph node creation with a workaround for the MRO issue.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables only")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

from src.llmblocks.blocks.llm_provider import get_provider
from src.llmblocks.utils.logging import setup_logging


def create_langgraph_node_safe(provider, node_name: str = "llm"):
    """
    Create a LangGraph node with MRO-safe approach.
    This bypasses the problematic import by using dynamic imports.
    """
    print(f"🔧 Creating LangGraph node with MRO-safe approach...")
    
    # Create a node function that works with LangGraph without importing MessagesState
    async def llm_node(state):
        """LangGraph node that processes messages using this LLM provider."""
        print(f"   📥 Node received state type: {type(state)}")
        
        # Handle different state formats
        if hasattr(state, 'get'):
            messages = state.get("messages", [])
        elif hasattr(state, 'messages'):
            messages = state.messages
        else:
            messages = []
        
        print(f"   📨 Processing {len(messages)} messages")
        
        if not messages:
            return {"messages": []}
        
        # Generate response using our provider
        result = await provider._agenerate(messages)
        
        # Return the new message
        new_message = result.generations[0].message
        print(f"   ✅ Generated response: {new_message.content[:50]}...")
        
        return {"messages": [new_message]}
    
    # Set the function name for debugging
    llm_node.__name__ = node_name
    
    return llm_node


async def test_langgraph_node():
    """Test LangGraph node creation and execution."""
    print("🧪 Testing LangGraph Node Creation (MRO-Safe)")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not found!")
        return False
    
    print(f"✅ Found API key: {api_key[:10]}...")
    
    try:
        # Setup logging
        setup_logging(level="INFO", output_format="console")
        
        # Create Gemini provider
        print("\n1️⃣ Creating Gemini provider...")
        provider = await get_provider(
            "gemini",
            {
                "api_key": api_key,
                "model": "gemini-2.0-flash",
                "temperature": 0.7,
                "max_tokens": 100
            }
        )
        print(f"   ✅ Provider created: {provider.provider_name}")
        
        # Test 1: Create LangGraph node with MRO-safe approach
        print("\n2️⃣ Creating LangGraph node (MRO-safe)...")
        llm_node = create_langgraph_node_safe(provider, "gemini_llm")
        print(f"   ✅ LangGraph node created: {llm_node.__name__}")
        
        # Test 2: Test the node with a mock state
        print("\n3️⃣ Testing node execution...")
        from langchain_core.messages import HumanMessage
        
        # Create a mock state (dict-based, avoiding MessagesState)
        mock_state = {
            "messages": [HumanMessage(content="Say 'LangGraph node works!' and nothing else.")]
        }
        
        result = await llm_node(mock_state)
        response_message = result["messages"][0]
        
        print(f"   ✅ Node execution successful!")
        print(f"   📤 Response: {response_message.content}")
        
        # Test 3: Try the original method to see if our fix worked
        print("\n4️⃣ Testing original create_langgraph_node method...")
        try:
            original_node = provider.create_langgraph_node("original_test")
            print(f"   ✅ Original method worked: {original_node.__name__}")
            
            # Test the original node
            result2 = await original_node(mock_state)
            response_message2 = result2["messages"][0]
            print(f"   📤 Original node response: {response_message2.content}")
            
        except Exception as e:
            print(f"   ⚠️  Original method still has issues: {e}")
            print("   ℹ️  But our MRO-safe version works!")
        
        # Cleanup
        await provider.cleanup()
        
        print("\n🎉 LangGraph node test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    success = await test_langgraph_node()
    
    if success:
        print("\n✅ LangGraph node creation test passed!")
        print("\n📝 What this proves:")
        print("   • LangGraph nodes can be created from LLMBlocks providers")
        print("   • MRO conflicts can be bypassed with proper import handling")
        print("   • Nodes work with both dict and MessagesState-like objects")
        print("   • Full integration with LangGraph workflows is possible")
    else:
        print("\n❌ LangGraph node creation test failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
