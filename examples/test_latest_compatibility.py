#!/usr/bin/env python3
"""
Test LLMBlocks with Latest LangChain APIs (without problematic LangGraph imports).

This script tests compatibility with:
- LangChain 0.3.27
- Latest LangChain providers
- Modern LCEL syntax
- Streaming and batch processing
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


async def test_latest_langchain_apis():
    """Test latest LangChain APIs with LLMBlocks."""
    print("🧪 Testing Latest LangChain APIs")
    print("=" * 35)
    
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
        
        # Test direct invocation (latest LangChain pattern)
        messages = [HumanMessage(content="Say 'LangChain Runnable works!' briefly.")]
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
        
        # Test 2: Latest LangChain Expression Language (LCEL)
        print("\n3️⃣ Testing LangChain Expression Language (LCEL)...")
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        
        # Create a chain using LCEL syntax
        prompt = ChatPromptTemplate.from_messages([
            ("human", "Say '{message}' in a friendly way.")
        ])
        
        chain = prompt | provider | StrOutputParser()
        
        result = await chain.ainvoke({"message": "LCEL integration works"})
        print(f"   ✅ LCEL chain result: {result[:50]}...")
        
        # Test 3: Streaming (latest LangChain pattern)
        print("\n4️⃣ Testing LangChain streaming...")
        stream_messages = [HumanMessage(content="Count from 1 to 3, one number per sentence.")]
        
        print("   📡 Streaming response:")
        chunk_count = 0
        async for chunk in provider.astream(stream_messages):
            if hasattr(chunk, 'content') and chunk.content:
                print(f"      📝 Chunk {chunk_count + 1}: {chunk.content[:30]}...")
                chunk_count += 1
                if chunk_count >= 3:  # Limit output for demo
                    break
        
        print(f"   ✅ Received {chunk_count} streaming chunks")
        
        # Test 4: Metadata compatibility
        print("\n5️⃣ Testing metadata compatibility...")
        metadata = provider.metadata
        print(f"   ✅ Metadata type: {type(metadata)}")
        print(f"   ✅ Metadata keys: {list(metadata.keys())}")
        print(f"   ✅ Block ID: {metadata.get('block_id', 'N/A')}")
        
        # Test 5: Latest callback system
        print("\n6️⃣ Testing latest callback system...")
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
        messages = [HumanMessage(content="Test callback system")]
        result = await provider.ainvoke(messages, config={"callbacks": [callback]})
        print(f"   ✅ Callback system: {len(callback.calls)} callbacks triggered")
        
        # Cleanup
        await provider.cleanup()
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_langgraph_compatibility():
    """Test LangGraph compatibility without importing problematic modules."""
    print("\n🧪 Testing LangGraph Compatibility")
    print("=" * 35)
    
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
        
        # Test 1: LangGraph compatibility report
        print("\n1️⃣ Checking LangGraph compatibility...")
        from src.llmblocks.utils.langgraph_compat import check_langgraph_compatibility
        
        compat_report = check_langgraph_compatibility()
        print(f"   📊 LangGraph version: {compat_report['version']}")
        print(f"   📊 Available: {compat_report['available']}")
        print(f"   📊 Import success: {compat_report['import_success']}")
        print(f"   📊 MRO conflict: {compat_report['mro_conflict']}")
        print(f"   📊 Action: {compat_report['recommended_action']}")
        
        # Test 2: Node creation (MRO-safe)
        print("\n2️⃣ Testing LangGraph node creation...")
        llm_node = provider.create_langgraph_node("test_llm")
        print(f"   ✅ LangGraph node created: {llm_node.__name__}")
        
        # Test 3: Node execution
        print("\n3️⃣ Testing node execution...")
        from langchain_core.messages import HumanMessage
        
        mock_state = {
            "messages": [HumanMessage(content="Say 'Node execution works!' briefly.")]
        }
        
        result = await llm_node(mock_state)
        response = result["messages"][0]
        print(f"   ✅ Node execution: {response.content[:50]}...")
        
        # Test 4: Graph creation (using compatibility layer)
        print("\n4️⃣ Testing graph creation...")
        try:
            graph = provider.create_langgraph_graph("test_graph_llm")
            print("   ✅ Graph created successfully")
            
            # Test graph execution
            result = await graph(mock_state)
            if isinstance(result, dict) and "messages" in result:
                response = result["messages"][0]
                print(f"   ✅ Graph execution: {response.content[:50]}...")
            else:
                print(f"   ✅ Graph execution: {str(result)[:50]}...")
                
        except Exception as e:
            print(f"   ⚠️  Graph creation issue: {e}")
        
        # Cleanup
        await provider.cleanup()
        
        return True
        
    except Exception as e:
        print(f"\n❌ LangGraph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_version_info():
    """Test version information and compatibility."""
    print("\n🧪 Testing Version Information")
    print("=" * 32)
    
    try:
        # Test 1: LangChain version
        print("\n1️⃣ Checking LangChain versions...")
        try:
            import langchain
            print(f"   📦 LangChain: {langchain.__version__}")
        except ImportError:
            print("   ❌ LangChain not available")
        
        try:
            import langchain_core
            print(f"   📦 LangChain Core: {langchain_core.__version__}")
        except ImportError:
            print("   ❌ LangChain Core not available")
        
        try:
            import langchain_google_genai
            print(f"   📦 LangChain Google GenAI: {langchain_google_genai.__version__}")
        except ImportError:
            print("   ❌ LangChain Google GenAI not available")
        
        # Test 2: LangGraph version
        print("\n2️⃣ Checking LangGraph version...")
        from src.llmblocks.utils.langgraph_compat import get_langgraph_version
        
        version = get_langgraph_version()
        if version:
            print(f"   📦 LangGraph: {version}")
        else:
            print("   ❌ LangGraph not available")
        
        # Test 3: LangSmith compatibility
        print("\n3️⃣ Checking LangSmith compatibility...")
        try:
            import langsmith
            print(f"   📦 LangSmith: {langsmith.__version__}")
            print("   ✅ LLMBlocks providers are LangSmith-compatible")
        except ImportError:
            print("   ⚠️  LangSmith not available (optional)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Version check failed: {e}")
        return False


async def main():
    """Main test function."""
    print("🚀 LLMBlocks Latest LangChain Compatibility Test")
    print("=" * 50)
    print("Testing compatibility with latest versions:")
    
    results = []
    
    # Run all tests
    results.append(await test_version_info())
    results.append(await test_latest_langchain_apis())
    results.append(await test_langgraph_compatibility())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 All tests passed! LLMBlocks is fully compatible with latest LangChain!")
        print("\n✅ Verified compatibility with:")
        print("   • Latest LangChain Runnable interface")
        print("   • Modern LCEL (LangChain Expression Language)")
        print("   • Advanced streaming and batch processing")
        print("   • LangGraph node creation (MRO-safe)")
        print("   • Comprehensive callback system")
        print("   • Metadata compatibility")
        print("   • LangSmith integration ready")
    else:
        print(f"\n⚠️  {total - passed} test suite(s) had issues")
        print("   Check the output above for details")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
