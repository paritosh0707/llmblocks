#!/usr/bin/env python3
"""
Test script for LLM providers in LLMBlocks.

This script demonstrates how to use different LLM providers
and tests the basic functionality.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from llmblocks.blocks.llm_provider import (
    get_provider,
    list_available_providers,
    LLMMessage,
    LLMRole
)
from llmblocks.utils.logging import setup_logging


async def test_provider(provider_name: str, config: dict):
    """Test a specific LLM provider."""
    print(f"\n🧪 Testing {provider_name} provider...")
    
    try:
        # Create provider
        provider = await get_provider(provider_name, config)
        
        # Test 1: Basic generation with our interface
        print(f"   1️⃣ Testing LLMBlocks interface...")
        messages = [
            LLMMessage(role=LLMRole.USER, content="Hello! Please respond with just 'Hi there!'")
        ]
        
        response = await provider.generate(messages, max_tokens=10)
        
        print(f"   ✅ Response: {response.content}")
        print(f"   📊 Model: {response.model}")
        print(f"   🔢 Tokens: {response.usage.get('total_tokens', 'N/A') if response.usage else 'N/A'}")
        
        # Test 2: LangChain compatibility
        print(f"   2️⃣ Testing LangChain compatibility...")
        from langchain_core.messages import HumanMessage
        
        langchain_messages = [HumanMessage(content="Say 'LangChain works!'")]
        langchain_result = await provider._agenerate(langchain_messages)
        langchain_response = langchain_result.generations[0][0].message
        
        print(f"   ✅ LangChain response: {langchain_response.content}")
        
        # Test 3: Streaming (if supported)
        if provider.is_streaming_enabled:
            print(f"   3️⃣ Testing streaming...")
            stream_content = ""
            async for chunk in provider.generate_stream(messages, max_tokens=20):
                if chunk.content:
                    stream_content += chunk.content
                    print(f"      📦 Chunk: {chunk.content}")
            print(f"   ✅ Full stream: {stream_content}")
        
        # Test 4: Health check
        health = await provider.health_check()
        print(f"   4️⃣ Health check: {'✅' if health.get('is_healthy') else '❌'}")
        
        # Test 5: LangGraph node creation
        print(f"   5️⃣ Testing LangGraph node creation...")
        llm_node = provider.create_langgraph_node("test_llm")
        print(f"   ✅ LangGraph node created: {llm_node.__name__}")
        
        # Cleanup
        await provider.cleanup()
        
        return True
        
    except Exception as e:
        print(f"❌ {provider_name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("🚀 LLMBlocks Provider Test Suite")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level="INFO", output_format="console")
    
    # List available providers
    providers = list_available_providers()
    print(f"📋 Available providers: {', '.join(providers)}")
    
    # Test configurations
    test_configs = {}
    
    # OpenAI configuration
    if os.getenv("OPENAI_API_KEY"):
        test_configs["openai"] = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 50
        }
    
    # Gemini configuration
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        test_configs["gemini"] = {
            "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            "model": "gemini-2.0-flash",
            "temperature": 0.7,
            "max_tokens": 50
        }
    
    # Anthropic configuration
    if os.getenv("ANTHROPIC_API_KEY"):
        test_configs["anthropic"] = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "model": "claude-3-5-haiku-20241022",
            "temperature": 0.7,
            "max_tokens": 50
        }
    
    if not test_configs:
        print("⚠️  No API keys found in environment variables.")
        print("   Set OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY to test providers.")
        return
    
    # Test each configured provider
    results = {}
    for provider_name, config in test_configs.items():
        results[provider_name] = await test_provider(provider_name, config)
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    for provider_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{provider_name:12} {status}")
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    print(f"\nOverall: {successful_tests}/{total_tests} providers working")


if __name__ == "__main__":
    asyncio.run(main())
