#!/usr/bin/env python3
"""
LangChain/LangGraph Compatibility Demo for LLMBlocks.

This script demonstrates how LLMBlocks providers are fully compatible
with LangChain and LangGraph, allowing seamless integration with the
broader LangChain ecosystem.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.llmblocks.blocks.llm_provider import get_provider
from src.llmblocks.utils.logging import setup_logging

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# LangGraph imports
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver


async def demo_langchain_compatibility():
    """Demonstrate LangChain compatibility."""
    print("🔗 LangChain Compatibility Demo")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY to run this demo")
        return
    
    # Create LLMBlocks provider
    provider = await get_provider(
        "openai",
        {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    
    print(f"✅ Created LLMBlocks provider: {provider.provider_name}")
    
    # 1. Direct LangChain usage
    print("\n1️⃣ Direct LangChain Message Usage:")
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="What is the capital of France?")
    ]
    
    # Use LangChain's generate method
    result = await provider._agenerate(messages)
    response = result.generations[0][0].message
    print(f"   Response: {response.content}")
    
    # 2. LangChain Chain usage
    print("\n2️⃣ LangChain Chain Usage:")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers in one sentence."),
        ("human", "{question}")
    ])
    
    # Create a chain using our provider
    chain = prompt | provider | StrOutputParser()
    
    # Use the chain
    chain_response = await chain.ainvoke({"question": "What is machine learning?"})
    print(f"   Chain Response: {chain_response}")
    
    # 3. LangChain Runnable interface
    print("\n3️⃣ LangChain Runnable Interface:")
    runnable = provider.get_langchain_runnable()
    
    # Use as a runnable
    runnable_response = await runnable.ainvoke([
        HumanMessage(content="Tell me a fun fact about space.")
    ])
    print(f"   Runnable Response: {runnable_response.content}")
    
    print("\n✅ LangChain compatibility demo completed!")


async def demo_langgraph_compatibility():
    """Demonstrate LangGraph compatibility."""
    print("\n🕸️ LangGraph Compatibility Demo")
    print("=" * 40)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY to run this demo")
        return
    
    # Create LLMBlocks provider
    provider = await get_provider(
        "openai",
        {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 150
        }
    )
    
    # Create a LangGraph workflow using our provider
    workflow = StateGraph(MessagesState)
    
    # Add our LLM provider as a node
    llm_node = provider.create_langgraph_node("llm")
    workflow.add_node("llm", llm_node)
    
    # Add a simple processing node
    def process_input(state: MessagesState):
        """Add a system message to the conversation."""
        messages = state.get("messages", [])
        if not any(msg.type == "system" for msg in messages):
            from langchain_core.messages import SystemMessage
            system_msg = SystemMessage(content="You are a creative storyteller. Keep responses concise.")
            return {"messages": [system_msg] + messages}
        return {"messages": messages}
    
    workflow.add_node("process", process_input)
    
    # Define the flow
    workflow.add_edge(START, "process")
    workflow.add_edge("process", "llm")
    workflow.add_edge("llm", END)
    
    # Compile the workflow
    app = workflow.compile(checkpointer=MemorySaver())
    
    # Test the workflow
    print("🚀 Running LangGraph workflow...")
    
    config = {"configurable": {"thread_id": "demo-thread"}}
    
    # First interaction
    result1 = await app.ainvoke(
        {"messages": [HumanMessage(content="Tell me a short story about a robot.")]},
        config=config
    )
    
    print(f"   Story: {result1['messages'][-1].content}")
    
    # Follow-up interaction (with memory)
    result2 = await app.ainvoke(
        {"messages": [HumanMessage(content="What was the robot's name?")]},
        config=config
    )
    
    print(f"   Follow-up: {result2['messages'][-1].content}")
    
    print("\n✅ LangGraph compatibility demo completed!")


async def demo_ecosystem_integration():
    """Demonstrate integration with LangChain ecosystem tools."""
    print("\n🌐 LangChain Ecosystem Integration Demo")
    print("=" * 45)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Set OPENAI_API_KEY to run this demo")
        return
    
    # Create LLMBlocks provider
    provider = await get_provider(
        "openai",
        {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": "gpt-4o-mini",
            "temperature": 0.3,
            "max_tokens": 200
        }
    )
    
    # Example: Using with LangChain's ConversationChain equivalent
    from langchain_core.memory import ChatMessageHistory
    from langchain_core.runnables.history import RunnableWithMessageHistory
    
    # Create a simple chain
    def get_session_history(session_id: str) -> ChatMessageHistory:
        return ChatMessageHistory()
    
    # Create a runnable with memory
    chain_with_memory = RunnableWithMessageHistory(
        provider,
        get_session_history,
        input_messages_key="messages",
        history_messages_key="messages",
    )
    
    # Test conversation with memory
    print("💬 Conversation with Memory:")
    
    config = {"configurable": {"session_id": "demo-session"}}
    
    response1 = await chain_with_memory.ainvoke(
        {"messages": [HumanMessage(content="My name is Alice. Remember this.")]},
        config=config
    )
    print(f"   Response 1: {response1.content}")
    
    response2 = await chain_with_memory.ainvoke(
        {"messages": [HumanMessage(content="What's my name?")]},
        config=config
    )
    print(f"   Response 2: {response2.content}")
    
    print("\n✅ Ecosystem integration demo completed!")


async def main():
    """Main demo function."""
    print("🚀 LLMBlocks LangChain/LangGraph Compatibility Demos")
    print("=" * 60)
    
    # Setup logging
    setup_logging(level="INFO", output_format="console")
    
    try:
        # Run all demos
        await demo_langchain_compatibility()
        await demo_langgraph_compatibility()
        await demo_ecosystem_integration()
        
        print("\n🎉 All compatibility demos completed successfully!")
        print("\n📝 Key Benefits:")
        print("   • Full LangChain compatibility - use any LangChain feature")
        print("   • LangGraph integration - build complex workflows")
        print("   • Ecosystem compatibility - works with all LangChain tools")
        print("   • Enhanced features - connection pooling, error handling, metrics")
        print("   • Unified interface - consistent API across all providers")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
