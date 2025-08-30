#!/usr/bin/env python3
"""
Basic Conversation Memory Example

This example demonstrates how to use LLMBlocks conversation memory
to maintain chat history across interactions.

Features demonstrated:
- Creating conversation memory
- Adding messages
- Retrieving conversation history
- Context window management
- LangChain compatibility
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


async def basic_memory_example():
    """Demonstrate basic conversation memory usage."""
    print("🧠 Basic Conversation Memory Example")
    print("=" * 50)
    
    try:
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Create conversation memory with in-memory backend
        memory = await get_conversation_memory(
            backend_type="in_memory",
            session_id="demo_chat",
            max_context_messages=10
        )
        
        print(f"✅ Created conversation memory (Session: {memory.get_session_id()})")
        
        # Simulate a conversation
        conversation_turns = [
            ("user", "Hello! My name is Alice."),
            ("assistant", "Hello Alice! Nice to meet you. How can I help you today?"),
            ("user", "I'm working on a Python project and need help with async programming."),
            ("assistant", "I'd be happy to help with async programming! What specific aspect are you struggling with?"),
            ("user", "How do I handle multiple async operations concurrently?"),
            ("assistant", "Great question! You can use asyncio.gather() or asyncio.create_task() for concurrent operations. Here's an example..."),
            ("user", "That's helpful! Can you also explain async context managers?"),
            ("assistant", "Absolutely! Async context managers use __aenter__ and __aexit__ methods..."),
        ]
        
        # Add messages to memory
        print("\n💬 Adding conversation messages...")
        for role, content in conversation_turns:
            if role == "user":
                await memory.add_user_message(content)
            else:
                await memory.add_assistant_message(content)
            print(f"   {role.title()}: {content[:50]}{'...' if len(content) > 50 else ''}")
        
        # Get conversation history
        print("\n📜 Retrieving conversation history...")
        history = await memory.get_conversation_history()
        print(f"   Total messages in history: {len(history)}")
        
        # Get context window (optimized for current conversation)
        print("\n🪟 Getting context window...")
        context = await memory.get_context_window()
        print(f"   Messages in context window: {len(context)}")
        
        # Display recent context
        print("\n📋 Recent Context:")
        for i, msg in enumerate(context[-3:], 1):  # Show last 3 messages
            timestamp = msg.timestamp.strftime("%H:%M:%S")
            print(f"   {i}. [{timestamp}] {msg.role.value}: {msg.content}")
        
        # Get conversation stats
        print("\n📊 Conversation Statistics:")
        stats = await memory.get_conversation_stats()
        for key, value in stats.items():
            if key not in ["session_id", "backend_type"]:
                print(f"   {key}: {value}")
        
        # Search messages
        print("\n🔍 Searching for 'async'...")
        search_results = await memory.search_messages("async", limit=3)
        print(f"   Found {len(search_results)} messages containing 'async'")
        for msg in search_results:
            print(f"   - {msg.role.value}: {msg.content[:60]}...")
        
        # Clean up
        await memory.close()
        print("\n✅ Memory closed successfully")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def langchain_integration_example():
    """Demonstrate LangChain integration."""
    print("\n🔗 LangChain Integration Example")
    print("=" * 50)
    
    try:
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Create memory
        memory = await get_conversation_memory(
            backend_type="in_memory",
            session_id="langchain_demo"
        )
        
        # Convert to LangChain format
        langchain_memory = memory.to_langchain_memory()
        langchain_chat_history = memory.to_langchain_chat_history()
        
        print("✅ Created LangChain-compatible memory adapters")
        
        # Add some messages through LangChain interface
        await langchain_chat_history.aadd_message(
            type("HumanMessage", (), {"content": "Hello from LangChain!"})()
        )
        
        # Get messages in LangChain format
        lc_messages = await memory.get_langchain_messages()
        print(f"📨 LangChain messages: {len(lc_messages)}")
        
        # Clean up
        await memory.close()
        print("✅ LangChain integration demo completed")
        
    except ImportError:
        print("⚠️  LangChain not available - skipping integration demo")
    except Exception as e:
        print(f"❌ Error in LangChain integration: {e}")


async def file_backend_example():
    """Demonstrate file-based persistence."""
    print("\n💾 File Backend Example")
    print("=" * 50)
    
    try:
        from llmblocks.blocks.memory import get_conversation_memory
        import tempfile
        
        # Create temporary directory for demo
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Using temporary storage: {temp_dir}")
            
            # Create memory with file backend
            memory = await get_conversation_memory(
                backend_type="file",
                session_id="persistent_chat",
                backend_config={"storage_dir": temp_dir}
            )
            
            # Add some messages
            await memory.add_user_message("This message will be saved to disk!")
            await memory.add_assistant_message("Yes, it will persist across restarts.")
            
            print("💬 Added messages to file-backed memory")
            
            # Close and reopen to test persistence
            await memory.close()
            
            # Create new memory instance with same session
            memory2 = await get_conversation_memory(
                backend_type="file",
                session_id="persistent_chat",
                backend_config={"storage_dir": temp_dir}
            )
            
            # Check if messages persisted
            history = await memory2.get_conversation_history()
            print(f"📜 Retrieved {len(history)} messages after restart")
            
            for msg in history:
                print(f"   {msg.role.value}: {msg.content}")
            
            await memory2.close()
            print("✅ File persistence demo completed")
            
    except Exception as e:
        print(f"❌ Error in file backend demo: {e}")


async def main():
    """Run all memory examples."""
    print("🚀 LLMBlocks Memory System Examples")
    print("=" * 60)
    
    await basic_memory_example()
    await langchain_integration_example()
    await file_backend_example()
    
    print("\n🎉 All memory examples completed!")
    print("\n💡 Next steps:")
    print("   - Try different backends (Redis, database)")
    print("   - Experiment with context strategies")
    print("   - Integrate with LLM providers for full conversations")
    print("   - Use in production applications")


if __name__ == "__main__":
    asyncio.run(main())
