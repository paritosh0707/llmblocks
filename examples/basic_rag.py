"""
Basic RAG Example for LLMBlocks

This example demonstrates how to use the different RAG pipelines:
- BasicRAG: Simple retrieval and generation
- StreamingRAG: Streaming response generation
- MemoryRAG: RAG with conversation memory

Now with LangChain Runnable compatibility:
- .invoke() method for single inputs
- .batch() method for multiple inputs
- .stream() method for streaming responses
- | operator for chaining
"""

import os
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

from llmblocks.blocks.rag import BasicRAG, StreamingRAG, MemoryRAG, RAGConfig, create_rag
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

def create_sample_documents() -> List[Document]:
    """Create sample documents for the RAG pipeline."""
    documents = [
        Document(
            page_content="""
            LLMBlocks is a modular, scalable Python library that provides pre-cooked, 
            production-ready building blocks for developing LLM-powered systems using 
            LangChain, LangGraph, and related frameworks. The focus is on abstraction, 
            modularity, and rapid prototyping of agentic workflows, RAG pipelines, 
            memory systems, evaluation tools, and more.
            """,
            metadata={"source": "project_overview", "type": "introduction"}
        ),
        Document(
            page_content="""
            The project structure includes blocks/ for high-level pluggable building 
            blocks (RAG, agents, memory, prompts, etc.), core/ for core utilities, 
            config loaders, base classes, tracing, logging, playground/ for Streamlit 
            UIs or visualization tools, cli/ for CLI interface, config/ for prompt 
            templates and tool configs, examples/ for end-to-end examples, tests/ for 
            test suites, scripts/ for dev/debug scripts, and docs/ for documentation.
            """,
            metadata={"source": "project_structure", "type": "architecture"}
        ),
        Document(
            page_content="""
            Priority modules for Level 1 MVP include blocks.rag with BasicRAG, 
            StreamingRAG, and MemoryRAG; blocks.agents with MultiToolAgent and 
            SelfCriticAgent; blocks.memory with InMemory, RedisMemory, and ChromaMemory; 
            blocks.prompts with PromptTemplateFactory and prompt registry; and core 
            utilities including utils, config_loader, tracing, and logger.
            """,
            metadata={"source": "mvp_modules", "type": "features"}
        ),
        Document(
            page_content="""
            The tech stack includes Python 3.12+, uv as the Python package manager, 
            LangChain, LangGraph, OpenAI, ChromaDB, Redis, and Streamlit selectively. 
            The project aims to wrap complex LangChain/LangGraph constructs into 
            developer-friendly interfaces and provide plug-and-play components for 
            various LLM-powered functionalities.
            """,
            metadata={"source": "tech_stack", "type": "technology"}
        )
    ]
    return documents

def basic_rag_example():
    """Demonstrate basic RAG functionality with both invoke() and query() methods."""
    print("=== Basic RAG Example ===")

    # Create RAG configuration
    config = RAGConfig(
        name="basic_rag_demo",
        chunk_size=500,
        chunk_overlap=100,
        top_k=3,
        llm_model="gpt-3.5-turbo",
        temperature=0.1
    )

    # Create RAG pipeline
    rag = BasicRAG(config)

    # Initialize and add documents
    with rag:
        documents = create_sample_documents()
        rag.add_documents(documents)

        # Query the RAG pipeline using both methods
        questions = [
            "What is LLMBlocks?",
            "What are the main components of the project structure?",
            "What is the tech stack used in LLMBlocks?"
        ]

        print("\n--- Using .invoke() method (new Runnable interface) ---")
        for question in questions:
            print(f"\nQuestion: {question}")
            try:
                # New invoke() method
                answer = rag.invoke(question)
                print(f"Answer: {answer}")
            except NotImplementedError as e:
                print(f"Error: {e}")
                print("Note: This example requires proper embedding setup")

        print("\n--- Using .query() method (backward compatibility) ---")
        for question in questions:
            print(f"\nQuestion: {question}")
            try:
                # Backward compatible query() method
                answer = rag.query(question)
                print(f"Answer: {answer}")
            except NotImplementedError as e:
                print(f"Error: {e}")
                print("Note: This example requires proper embedding setup")

def streaming_rag_example():
    """Demonstrate streaming RAG functionality with new stream() method."""
    print("\n=== Streaming RAG Example ===")

    # Create streaming RAG
    rag = create_rag("streaming", 
                     chunk_size=500,
                     chunk_overlap=100,
                     top_k=3,
                     llm_model="gpt-3.5-turbo",
                     temperature=0.1)

    # Initialize and add documents
    with rag:
        documents = create_sample_documents()
        rag.add_documents(documents)

        # Query with streaming using new stream() method
        question = "Explain the priority modules for the MVP"
        print(f"\nQuestion: {question}")
        print("Answer (streaming with .stream()): ", end="", flush=True)

        try:
            # New stream() method
            for chunk in rag.stream(question):
                print(chunk, end="", flush=True)
            print()  # New line after streaming
        except NotImplementedError as e:
            print(f"\nError: {e}")
            print("Note: This example requires proper embedding setup")

        # Also demonstrate backward compatibility
        print("\nAnswer (streaming with .query_stream()): ", end="", flush=True)
        try:
            # Backward compatible query_stream() method
            for chunk in rag.query_stream(question):
                print(chunk, end="", flush=True)
            print()  # New line after streaming
        except NotImplementedError as e:
            print(f"\nError: {e}")
            print("Note: This example requires proper embedding setup")

def memory_rag_example():
    """Demonstrate RAG with memory functionality."""
    print("\n=== Memory RAG Example ===")

    # Create memory RAG
    rag = create_rag("memory",
                     chunk_size=500,
                     chunk_overlap=100,
                     top_k=3,
                     llm_model="gpt-3.5-turbo",
                     temperature=0.1)

    # Initialize and add documents
    with rag:
        documents = create_sample_documents()
        rag.add_documents(documents)

        # Conversation flow using invoke() method
        conversation = [
            "What is LLMBlocks?",
            "What are the main components mentioned in the project structure?",
            "Can you tell me more about the blocks.rag module specifically?",
            "What about the memory systems you mentioned earlier?"
        ]

        try:
            for i, question in enumerate(conversation, 1):
                print(f"\nTurn {i}:")
                print(f"Question: {question}")
                # Using new invoke() method
                answer = rag.invoke(question)
                print(f"Answer: {answer}")

                # Show memory state
                print(f"Memory entries: {len(rag.conversation_history)}")
        except NotImplementedError as e:
            print(f"Error: {e}")
            print("Note: This example requires proper embedding setup")

def batch_processing_example():
    """Demonstrate batch processing with the new batch() method."""
    print("\n=== Batch Processing Example ===")

    # Create RAG pipeline
    rag = create_rag("basic",
                     chunk_size=500,
                     chunk_overlap=100,
                     top_k=3,
                     llm_model="gpt-3.5-turbo",
                     temperature=0.1)

    # Initialize and add documents
    with rag:
        documents = create_sample_documents()
        rag.add_documents(documents)

        # Batch questions
        questions = [
            "What is LLMBlocks?",
            "What are the main components?",
            "What is the tech stack?",
            "What are the priority modules?"
        ]

        try:
            print("Processing questions in batch...")
            # New batch() method
            answers = rag.batch(questions)

            for question, answer in zip(questions, answers):
                print(f"\nQ: {question}")
                print(f"A: {answer}")
        except NotImplementedError as e:
            print(f"Error: {e}")
            print("Note: This example requires proper embedding setup")

def chaining_example():
    """Demonstrate chaining capabilities with the | operator."""
    print("\n=== Chaining Example ===")

    # Create RAG pipeline
    rag = create_rag("basic",
                     chunk_size=500,
                     chunk_overlap=100,
                     top_k=3,
                     llm_model="gpt-3.5-turbo",
                     temperature=0.1)

    # Initialize and add documents
    with rag:
        documents = create_sample_documents()
        rag.add_documents(documents)

        # Create a simple post-processor (example of what could be chained)
        class SimplePostProcessor:
            def invoke(self, input: str, config: Optional[Dict[str, Any]] = None) -> str:
                """Add a prefix to the response."""
                return f"[Processed] {input}"

            def __or__(self, other):
                """Enable chaining."""
                return self.chain(other)

        post_processor = SimplePostProcessor()

        try:
            # Demonstrate chaining (this would work with proper LangChain integration)
            print("Chaining RAG with post-processor...")
            # pipeline = rag | post_processor  # This would work with proper integration
            # result = pipeline.invoke("What is LLMBlocks?")

            # For now, demonstrate the concept
            question = "What is LLMBlocks?"
            rag_result = rag.invoke(question)
            final_result = post_processor.invoke(rag_result)

            print(f"Original: {rag_result}")
            print(f"Chained: {final_result}")

        except NotImplementedError as e:
            print(f"Error: {e}")
            print("Note: This example requires proper embedding setup")

def factory_example():
    """Demonstrate the factory function for creating RAG pipelines."""
    print("\n=== Factory Function Example ===")

    # Create different types of RAG using the factory
    rag_types = ["basic", "streaming", "memory"]

    for rag_type in rag_types:
        print(f"\nCreating {rag_type} RAG...")
        rag = create_rag(rag_type, 
                        name=f"{rag_type}_demo",
                        chunk_size=500,
                        top_k=2)

        print(f"RAG type: {type(rag).__name__}")
        print(f"Config: {rag.config.name}")
        print(f"Streaming enabled: {rag.config.streaming_enabled}")
        print(f"Memory enabled: {rag.config.memory_enabled}")

        # Test invoke method availability
        print(f"Has invoke() method: {hasattr(rag, 'invoke')}")
        print(f"Has batch() method: {hasattr(rag, 'batch')}")
        print(f"Has stream() method: {hasattr(rag, 'stream')}")

if __name__ == "__main__":
    print("LLMBlocks RAG Examples with LangChain Runnable Compatibility")
    print("=" * 70)

    # Note: These examples require proper setup of embeddings and API keys
    print("Note: These examples require:")
    print("- OpenAI API key set in environment variables")
    print("- Proper embedding model setup")
    print("- Internet connection for API calls")
    print("\nNew Features:")
    print("- .invoke() method for Runnable compatibility")
    print("- .batch() method for batch processing")
    print("- .stream() method for streaming responses")
    print("- | operator support for chaining (with proper integration)")
    print("- Backward compatibility with .query() and .query_stream() methods")

    # Run examples
    basic_rag_example()
    streaming_rag_example()
    memory_rag_example()
    batch_processing_example()
    chaining_example()
    factory_example()