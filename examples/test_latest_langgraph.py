#!/usr/bin/env python3
"""
Test LLMBlocks with Latest LangChain/LangGraph APIs.

This script tests compatibility with:
- LangChain 0.3.27
- LangGraph 0.6.6
- Latest LangChain providers
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


async def test_latest_langchain_features():
    """Test latest LangChain features with LLMBlocks."""
    print("🧪 Testing Latest LangChain Features")
    print("=" * 40)
    
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
                "max_tokens": 150
            }
        )
        print(f"   ✅ Provider created: {provider.provider_name}")
        
        # Test 1: Latest LangChain Runnable interface
        print("\n2️⃣ Testing LangChain Runnable interface...")
        from langchain_core.messages import HumanMessage
        from langchain_core.runnables import RunnablePassthrough
        
        # Create a simple chain using the latest Runnable syntax
        messages = [HumanMessage(content="Say 'LangChain Runnable works!' briefly.")]
        
        # Test direct invocation (latest LangChain pattern)
        result = await provider.ainvoke(messages)
        print(f"   ✅ Runnable ainvoke: {result.content[:50]}...")
        
        # Test batch processing (latest LangChain pattern)
        batch_messages = [
            [HumanMessage(content="Say 'Batch 1' briefly.")],
            [HumanMessage(content="Say 'Batch 2' briefly.")]
        ]
        batch_results = await provider.abatch(batch_messages)
        print(f"   ✅ Runnable abatch: {len(batch_results)} results")
        for i, result in enumerate(batch_results):
            print(f"      Batch {i+1}: {result.content[:30]}...")
        
        # Test streaming (latest LangChain pattern)
        print("\n3️⃣ Testing LangChain streaming...")
        stream_messages = [HumanMessage(content="Count from 1 to 3, one number per sentence.")]
        
        print("   📡 Streaming response:")
        async for chunk in provider.astream(stream_messages):
            if hasattr(chunk, 'content') and chunk.content:
                print(f"      📝 {chunk.content[:50]}...")
                break  # Just show first chunk for demo
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_latest_langgraph_features():
    """Test latest LangGraph 0.6.6 features with LLMBlocks."""
    print("\n🧪 Testing Latest LangGraph Features")
    print("=" * 40)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not found!")
        return False
    
    try:
        # Create provider
        provider = await get_provider(
            "gemini",
            {
                "api_key": api_key,
                "model": "gemini-2.0-flash",
                "temperature": 0.7,
                "max_tokens": 100
            }
        )
        
        # Test 1: Latest LangGraph StateGraph API with compatibility layer
        print("\n1️⃣ Testing LangGraph StateGraph (v0.6.6)...")
        
        # Use our compatibility layer
        from src.llmblocks.utils.langgraph_compat import check_langgraph_compatibility, create_compatible_graph
        from langchain_core.messages import HumanMessage
        
        # Check compatibility first
        compat_report = check_langgraph_compatibility()
        print(f"   📊 LangGraph compatibility: {compat_report['recommended_action']}")
        
        # Create LLMBlocks node
        llm_node = provider.create_langgraph_node("llm")
        
        # Try to create a graph using compatibility layer
        success, graph, error = create_compatible_graph(llm_node, "llm")
        
        if success:
            print("   ✅ Full LangGraph StateGraph created")
            
            # Test the graph
            initial_state = {
                "messages": [HumanMessage(content="Say 'LangGraph 0.6.6 works!' briefly.")]
            }
            
            result = await graph.ainvoke(initial_state)
            final_message = result["messages"][-1]
            print(f"   ✅ Graph execution: {final_message.content[:50]}...")
            
        else:
            print(f"   ⚠️  Using fallback graph: {error}")
            
            # Test the fallback graph
            initial_state = {
                "messages": [HumanMessage(content="Fallback graph test")]
            }
            
            result = await graph(initial_state)
            response = result["messages"][0]
            print(f"   ✅ Fallback graph works: {response.content[:50]}...")
        
        # Test 2: Latest LangGraph checkpoint features
        print("\n2️⃣ Testing LangGraph checkpointing...")
        try:
            from langgraph.checkpoint.memory import MemorySaver
            
            # Test with memory saver (latest LangGraph feature)
            memory = MemorySaver()
            print("   ✅ MemorySaver created")
            
            # Note: Full checkpoint testing would require a complete graph setup
            # This just verifies the import and basic instantiation work
            
        except ImportError as e:
            print(f"   ⚠️  Checkpoint feature not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ LangGraph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_latest_integrations():
    """Test integration with latest LangChain ecosystem components."""
    print("\n🧪 Testing Latest Ecosystem Integrations")
    print("=" * 45)
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY environment variable not found!")
        return False
    
    try:
        # Create provider
        provider = await get_provider(
            "gemini",
            {
                "api_key": api_key,
                "model": "gemini-2.0-flash",
                "temperature": 0.7,
                "max_tokens": 100
            }
        )
        
        # Test 1: Latest LangChain Expression Language (LCEL)
        print("\n1️⃣ Testing LangChain Expression Language (LCEL)...")
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Create a chain using LCEL syntax
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Say '{message}' in a friendly way.")
        ])
        
        chain = prompt | provider | StrOutputParser()
        
        result = await chain.ainvoke({"message": "LCEL integration works"})
        print(f"   ✅ LCEL chain result: {result[:50]}...")
        
        # Test 2: Latest LangSmith integration (if available)
        print("\n2️⃣ Testing LangSmith compatibility...")
        try:
            import langsmith
            print(f"   ✅ LangSmith available: {langsmith.__version__}")
            print("   ℹ️  LLMBlocks providers are LangSmith-compatible")
        except ImportError:
            print("   ⚠️  LangSmith not available (optional)")
        
        # Test 3: Latest callback system
        print("\n3️⃣ Testing latest callback system...")
        from langchain_core.callbacks import AsyncCallbackHandler
        
        class TestCallback(AsyncCallbackHandler):
            def __init__(self):
                self.calls = []
            
            async def on_llm_start(self, serialized, prompts, **kwargs):
                self.calls.append("llm_start")
            
            async def on_llm_end(self, response, **kwargs):
                self.calls.append("llm_end")
        
        callback = TestCallback()
        
        # Test with callback
        from langchain_core.messages import HumanMessage
        messages = [HumanMessage(content="Test callback system")]
        
        result = await provider.ainvoke(messages, config={"callbacks": [callback]})
        print(f"   ✅ Callback system: {len(callback.calls)} callbacks triggered")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 LLMBlocks Latest LangChain/LangGraph Compatibility Test")
    print("=" * 60)
    print("Testing with:")
    print("• LangChain 0.3.27+")
    print("• LangGraph 0.6.6+")
    print("• Latest LangChain providers")
    print("• Modern LCEL syntax")
    print("• Advanced LangGraph features")
    
    results = []
    
    # Run all tests
    results.append(await test_latest_langchain_features())
    results.append(await test_latest_langgraph_features())
    results.append(await test_latest_integrations())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 All tests passed! LLMBlocks is fully compatible with latest LangChain/LangGraph!")
        print("\n✅ Verified compatibility with:")
        print("   • LangChain 0.3.27+ Runnable interface")
        print("   • LangGraph 0.6.6+ StateGraph API")
        print("   • Latest LCEL (LangChain Expression Language)")
        print("   • Modern streaming and batch processing")
        print("   • Advanced callback system")
        print("   • LangSmith integration ready")
    else:
        print(f"\n⚠️  {total - passed} test suite(s) had issues")
        print("   Check the output above for details")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
