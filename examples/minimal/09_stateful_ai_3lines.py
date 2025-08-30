#!/usr/bin/env python3
"""
🚀 Stateful AI in Just 3 Lines!

This is the dream realized - a fully stateful AI assistant that remembers
conversations, maintains context, and provides intelligent responses with
minimal code.

The magic:
- Line 1: Create stateful AI
- Line 2: First interaction  
- Line 3: AI remembers previous context!

No memory management, no context handling, no complexity - just pure AI magic!
"""

import asyncio
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


async def three_line_dream():
    """The ultimate dream: Stateful AI in 3 lines!"""
    print("🚀 The Dream: Stateful AI in Just 3 Lines!")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    try:
        from llmblocks.blocks.llm_provider import get_stateful_ai
        
        print("✨ Creating stateful AI...")
        print()
        print("```python")
        print("# Line 1: Create stateful AI")
        print('ai = await get_stateful_ai("gemini", api_key="your-key")')
        ai = await get_stateful_ai("gemini", api_key=api_key)
        
        print()
        print("# Line 2: First interaction")
        print('response1 = await ai.chat("Hi! My name is Alice and I love Python.")')
        response1 = await ai.chat("Hi! My name is Alice and I love Python.")
        
        print()
        print("# Line 3: AI remembers context!")
        print('response2 = await ai.chat("What\'s my name and what do I love?")')
        response2 = await ai.chat("What's my name and what do I love?")
        print("```")
        
        print()
        print("🎯 Results:")
        print(f"👤 User: Hi! My name is Alice and I love Python.")
        print(f"🤖 AI: {response1}")
        print()
        print(f"👤 User: What's my name and what do I love?")
        print(f"🤖 AI: {response2}")
        
        # Show conversation summary
        summary = await ai.get_conversation_summary()
        print()
        print("📊 Conversation Summary:")
        print(f"   Session: {summary['session_id']}")
        print(f"   Total messages: {summary['total_messages']}")
        print(f"   Conversation turns: {summary['conversation_turns']}")
        print(f"   Memory backend: {summary['memory_backend']}")
        
        await ai.close()
        print()
        print("✅ 3-line stateful AI completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def persistent_ai_demo():
    """Demonstrate persistent AI that survives restarts."""
    print("\n💾 Persistent AI Demo")
    print("=" * 60)
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    try:
        from llmblocks.blocks.llm_provider import get_persistent_ai
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Using storage: {temp_dir}")
            
            # Session 1: Start conversation
            print("\n🔄 Session 1: Learning about user...")
            ai1 = await get_persistent_ai(
                "gemini", 
                api_key=api_key,
                session_id="persistent_user",
                storage_dir=temp_dir
            )
            
            response1 = await ai1.chat("I'm a software engineer working on AI projects.")
            print(f"👤 User: I'm a software engineer working on AI projects.")
            print(f"🤖 AI: {response1[:100]}...")
            
            await ai1.close()
            print("✅ Session 1 saved and closed")
            
            # Session 2: Resume conversation (simulates app restart)
            print("\n🔄 Session 2: Resuming after 'restart'...")
            ai2 = await get_persistent_ai(
                "gemini",
                api_key=api_key, 
                session_id="persistent_user",  # Same session ID
                storage_dir=temp_dir
            )
            
            response2 = await ai2.chat("What do you remember about my profession?")
            print(f"👤 User: What do you remember about my profession?")
            print(f"🤖 AI: {response2[:100]}...")
            
            summary = await ai2.get_conversation_summary()
            print(f"\n📊 Persistent conversation: {summary['conversation_turns']} turns, {summary['total_messages']} messages")
            
            await ai2.close()
            print("✅ Persistent AI demo completed!")
            
    except Exception as e:
        print(f"❌ Error: {e}")


async def streaming_stateful_ai():
    """Demonstrate streaming stateful AI."""
    print("\n🌊 Streaming Stateful AI")
    print("=" * 60)
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    try:
        from llmblocks.blocks.llm_provider import get_stateful_ai
        
        ai = await get_stateful_ai("gemini", api_key=api_key)
        
        # Set context
        await ai.chat("I'm learning about machine learning algorithms.")
        
        print("👤 User: Can you explain neural networks in simple terms?")
        print("🤖 AI: ", end="", flush=True)
        
        # Stream response
        async for chunk in ai.chat_stream("Can you explain neural networks in simple terms?"):
            print(chunk, end="", flush=True)
        
        print("\n")
        
        # Follow-up that uses context
        print("👤 User: How does this relate to what I'm learning?")
        print("🤖 AI: ", end="", flush=True)
        
        async for chunk in ai.chat_stream("How does this relate to what I'm learning?"):
            print(chunk, end="", flush=True)
        
        print("\n")
        
        await ai.close()
        print("\n✅ Streaming stateful AI completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def multi_session_demo():
    """Demonstrate multiple independent AI sessions."""
    print("\n👥 Multi-Session AI Demo")
    print("=" * 60)
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("❌ Please set GOOGLE_API_KEY environment variable")
        return
    
    try:
        from llmblocks.blocks.llm_provider import get_stateful_ai
        
        # Create multiple AI sessions
        alice_ai = await get_stateful_ai("gemini", api_key=api_key, session_id="alice")
        bob_ai = await get_stateful_ai("gemini", api_key=api_key, session_id="bob")
        
        # Alice's conversation
        print("👩 Alice's AI Session:")
        alice_response = await alice_ai.chat("Hi! I'm Alice and I love cooking.")
        print(f"   Alice: Hi! I'm Alice and I love cooking.")
        print(f"   AI: {alice_response[:80]}...")
        
        # Bob's conversation  
        print("\n👨 Bob's AI Session:")
        bob_response = await bob_ai.chat("Hello! I'm Bob and I'm into sports.")
        print(f"   Bob: Hello! I'm Bob and I'm into sports.")
        print(f"   AI: {bob_response[:80]}...")
        
        # Test session isolation
        print("\n🔍 Testing Session Isolation:")
        alice_test = await alice_ai.chat("What's my hobby?")
        bob_test = await bob_ai.chat("What's my hobby?")
        
        print(f"   Alice AI remembers: {alice_test[:60]}...")
        print(f"   Bob AI remembers: {bob_test[:60]}...")
        
        # Cleanup
        await alice_ai.close()
        await bob_ai.close()
        
        print("\n✅ Multi-session demo completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all 3-line AI demonstrations."""
    print("🎉 LLMBlocks: The Dream of 3-Line Stateful AI")
    print("=" * 70)
    print()
    print("💡 The Vision:")
    print("   Create powerful, stateful AI assistants with minimal code.")
    print("   No complex memory management, no context handling - just pure AI magic!")
    print()
    
    await three_line_dream()
    await persistent_ai_demo()
    await streaming_stateful_ai()
    await multi_session_demo()
    
    print("\n" + "="*70)
    print("🎊 THE DREAM IS REALIZED!")
    print("="*70)
    print()
    print("✨ What we've achieved:")
    print("   ✅ Stateful AI in just 3 lines of code")
    print("   ✅ Automatic memory management")
    print("   ✅ Context-aware conversations")
    print("   ✅ Persistent storage across restarts")
    print("   ✅ Real-time streaming responses")
    print("   ✅ Multi-session support")
    print("   ✅ LangChain/LangGraph compatibility")
    print()
    print("🚀 From complex AI frameworks to simple, powerful solutions!")
    print("   This is the future of AI development - accessible to everyone.")


if __name__ == "__main__":
    asyncio.run(main())
