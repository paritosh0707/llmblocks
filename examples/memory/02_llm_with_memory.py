#!/usr/bin/env python3
"""
LLM Provider with Memory Integration

This example shows how to combine LLM providers with conversation memory
to create a stateful chatbot that remembers previous interactions.

Features demonstrated:
- LLM + Memory integration
- Persistent conversation history
- Context-aware responses
- Session management
- Memory-enhanced chatbot
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


class MemoryEnhancedChatbot:
    """
    A chatbot that combines LLM providers with conversation memory.
    
    This demonstrates how to create a stateful AI assistant that
    remembers previous conversations and maintains context.
    """
    
    def __init__(self, llm_provider, memory, system_prompt=None):
        self.llm_provider = llm_provider
        self.memory = memory
        self.system_prompt = system_prompt or (
            "You are a helpful AI assistant. You have access to the conversation "
            "history and should use it to provide contextual responses. "
            "Remember details from previous messages and refer to them when relevant."
        )
    
    async def initialize(self):
        """Initialize the chatbot."""
        # Add system message if provided
        if self.system_prompt:
            await self.memory.add_system_message(self.system_prompt)
    
    async def chat(self, user_message: str) -> str:
        """
        Process a user message and return AI response.
        
        This method:
        1. Adds user message to memory
        2. Gets conversation context
        3. Generates AI response with context
        4. Adds AI response to memory
        5. Returns the response
        """
        # Add user message to memory
        await self.memory.add_user_message(user_message)
        
        # Get conversation context
        context_messages = await self.memory.get_context_window()
        
        # Convert to format expected by LLM provider
        llm_messages = []
        for msg in context_messages:
            llm_messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Generate response
        response = await self.llm_provider.generate(llm_messages)
        ai_response = response.content
        
        # Add AI response to memory
        await self.memory.add_assistant_message(ai_response)
        
        return ai_response
    
    async def get_conversation_summary(self) -> dict:
        """Get a summary of the current conversation."""
        stats = await self.memory.get_conversation_stats()
        history = await self.memory.get_conversation_history()
        
        return {
            "session_id": self.memory.get_session_id(),
            "total_messages": len(history),
            "context_messages": stats.get("context_messages", 0),
            "conversation_turns": len([m for m in history if m.role.value == "user"]),
            "memory_backend": stats.get("backend_type", "unknown")
        }
    
    async def clear_conversation(self):
        """Clear the conversation history."""
        await self.memory.clear_memory()
        # Re-add system prompt
        if self.system_prompt:
            await self.memory.add_system_message(self.system_prompt)
    
    async def close(self):
        """Close the chatbot and cleanup resources."""
        await self.memory.close()
        await self.llm_provider.close()


async def basic_chatbot_demo():
    """Demonstrate basic memory-enhanced chatbot."""
    print("🤖 Memory-Enhanced Chatbot Demo")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("⚠️  Please set GOOGLE_API_KEY environment variable")
        return
    
    try:
        from llmblocks.blocks.llm_provider import get_provider
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Create LLM provider
        print("🧠 Creating LLM provider...")
        llm_provider = await get_provider("gemini", api_key=api_key)
        
        # Create conversation memory
        print("💾 Creating conversation memory...")
        memory = await get_conversation_memory(
            backend_type="in_memory",
            session_id="demo_chatbot",
            max_context_messages=20
        )
        
        # Create memory-enhanced chatbot
        chatbot = MemoryEnhancedChatbot(
            llm_provider=llm_provider,
            memory=memory,
            system_prompt=(
                "You are Alice, a helpful AI assistant. You remember our conversation "
                "and can refer to previous topics. Be friendly and conversational."
            )
        )
        
        await chatbot.initialize()
        print("✅ Chatbot initialized with memory")
        
        # Simulate a conversation with memory
        conversation = [
            "Hi! My name is Bob and I'm a software developer.",
            "I'm working on a Python project about machine learning.",
            "Can you help me understand neural networks?",
            "What did I say my name was?",  # Test memory
            "What was I working on again?",  # Test memory
            "Thanks for remembering! Can you suggest some ML libraries?"
        ]
        
        print("\n💬 Starting conversation...")
        for i, user_input in enumerate(conversation, 1):
            print(f"\n--- Turn {i} ---")
            print(f"👤 User: {user_input}")
            
            # Get AI response
            ai_response = await chatbot.chat(user_input)
            print(f"🤖 Alice: {ai_response}")
            
            # Show conversation summary every few turns
            if i % 3 == 0:
                summary = await chatbot.get_conversation_summary()
                print(f"📊 Conversation: {summary['conversation_turns']} turns, "
                      f"{summary['total_messages']} total messages")
        
        # Final conversation summary
        print("\n📋 Final Conversation Summary:")
        summary = await chatbot.get_conversation_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Clean up
        await chatbot.close()
        print("\n✅ Chatbot demo completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def persistent_chatbot_demo():
    """Demonstrate persistent chatbot with file storage."""
    print("\n💾 Persistent Chatbot Demo")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key or api_key == 'your-api-key-here':
        print("⚠️  Please set GOOGLE_API_KEY environment variable")
        return
    
    try:
        from llmblocks.blocks.llm_provider import get_provider
        from llmblocks.blocks.memory import get_conversation_memory
        import tempfile
        
        # Create temporary directory for persistence
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"📁 Using storage directory: {temp_dir}")
            
            # Session 1: Start conversation
            print("\n🔄 Session 1: Starting conversation...")
            llm_provider1 = await get_provider("gemini", api_key=api_key)
            memory1 = await get_conversation_memory(
                backend_type="file",
                session_id="persistent_user",
                backend_config={"storage_dir": temp_dir}
            )
            
            chatbot1 = MemoryEnhancedChatbot(llm_provider1, memory1)
            await chatbot1.initialize()
            
            # Have initial conversation
            response1 = await chatbot1.chat("Hi! I'm learning about AI and machine learning.")
            print(f"🤖 Response 1: {response1[:100]}...")
            
            response2 = await chatbot1.chat("Can you recommend some beginner resources?")
            print(f"🤖 Response 2: {response2[:100]}...")
            
            await chatbot1.close()
            print("✅ Session 1 completed and saved")
            
            # Session 2: Resume conversation
            print("\n🔄 Session 2: Resuming conversation...")
            llm_provider2 = await get_provider("gemini", api_key=api_key)
            memory2 = await get_conversation_memory(
                backend_type="file",
                session_id="persistent_user",  # Same session ID
                backend_config={"storage_dir": temp_dir}
            )
            
            chatbot2 = MemoryEnhancedChatbot(llm_provider2, memory2)
            
            # Check if previous conversation was loaded
            history = await memory2.get_conversation_history()
            print(f"📜 Loaded {len(history)} messages from previous session")
            
            # Continue conversation
            response3 = await chatbot2.chat("What did we discuss in our last conversation?")
            print(f"🤖 Response 3: {response3[:100]}...")
            
            await chatbot2.close()
            print("✅ Session 2 completed")
            
            print("\n🎉 Persistent conversation demo completed!")
            print("   The chatbot successfully remembered the previous conversation!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def context_strategy_demo():
    """Demonstrate different context management strategies."""
    print("\n🪟 Context Strategy Demo")
    print("=" * 50)
    
    try:
        from llmblocks.blocks.memory import get_conversation_memory
        
        # Test different context strategies
        strategies = ["sliding_window", "summarized", "priority_based"]
        
        for strategy in strategies:
            print(f"\n📋 Testing {strategy} strategy...")
            
            memory = await get_conversation_memory(
                backend_type="in_memory",
                session_id=f"strategy_{strategy}",
                max_context_messages=5,  # Small window for demo
                context_strategy=strategy
            )
            
            # Add many messages to test context management
            for i in range(10):
                await memory.add_user_message(f"User message {i+1}: This is a test message.")
                await memory.add_assistant_message(f"Assistant response {i+1}: I understand your message.")
            
            # Get context window
            context = await memory.get_context_window()
            print(f"   Context window size: {len(context)} messages")
            
            # Show context summary
            user_msgs = len([m for m in context if m.role.value == "user"])
            assistant_msgs = len([m for m in context if m.role.value == "assistant"])
            system_msgs = len([m for m in context if m.role.value == "system"])
            
            print(f"   Messages: {user_msgs} user, {assistant_msgs} assistant, {system_msgs} system")
            
            await memory.close()
        
        print("\n✅ Context strategy demo completed")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all memory + LLM examples."""
    print("🚀 LLM Provider + Memory Integration Examples")
    print("=" * 60)
    
    await basic_chatbot_demo()
    await persistent_chatbot_demo()
    await context_strategy_demo()
    
    print("\n🎉 All integration examples completed!")
    print("\n💡 Key takeaways:")
    print("   - Memory enables stateful conversations")
    print("   - Different backends provide different persistence levels")
    print("   - Context strategies optimize memory usage")
    print("   - Integration is seamless and powerful")


if __name__ == "__main__":
    asyncio.run(main())
