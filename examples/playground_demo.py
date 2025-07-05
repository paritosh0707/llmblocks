"""
Playground Demo - Programmatic usage of RAG playground components.

This demo shows how to use the playground components programmatically,
which is useful for understanding the underlying functionality and
for integration into other applications.

Now with LangChain Runnable interface support:
- .invoke() for single inputs
- .batch() for multiple inputs  
- .stream() for streaming responses
- Chaining with | operator
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llmblocks.blocks.rag import BasicRAG, StreamingRAG, MemoryRAG, RAGConfig, create_rag
from llmblocks.core.config_loader import ConfigLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration."""
    documents = [
        Document(
            page_content="""
            LLMBlocks is a modular, scalable Python library that provides pre-cooked, 
            production-ready building blocks for developing LLM-powered systems using 
            LangChain, LangGraph, and related frameworks. The focus is on abstraction, 
            modularity, and rapid prototyping of agentic workflows, RAG pipelines, 
            memory systems, evaluation tools, and more.
            
            The project structure includes blocks/ for high-level pluggable building 
            blocks (RAG, agents, memory, prompts, etc.), core/ for core utilities, 
            config loaders, base classes, tracing, logging, playground/ for Streamlit 
            UIs or visualization tools, cli/ for CLI interface, config/ for prompt 
            templates and tool configs, examples/ for end-to-end examples, tests/ for 
            test suites, scripts/ for dev/debug scripts, and docs/ for documentation.
            """,
            metadata={"source": "project_overview", "type": "introduction"}
        ),
        Document(
            page_content="""
            Priority modules for Level 1 MVP include blocks.rag with BasicRAG, 
            StreamingRAG, and MemoryRAG; blocks.agents with MultiToolAgent and 
            SelfCriticAgent; blocks.memory with InMemory, RedisMemory, and ChromaMemory; 
            blocks.prompts with PromptTemplateFactory and prompt registry; and core 
            utilities including utils, config_loader, tracing, and logger.
            
            The tech stack includes Python 3.12+, uv as the Python package manager, 
            LangChain, LangGraph, OpenAI, ChromaDB, Redis, and Streamlit selectively. 
            The project aims to wrap complex LangChain/LangGraph constructs into 
            developer-friendly interfaces and provide plug-and-play components for 
            various LLM-powered functionalities.
            """,
            metadata={"source": "mvp_modules", "type": "features"}
        )
    ]
    return documents


def process_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Process documents with text splitting (same as playground)."""
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    processed_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            processed_docs.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            ))

    return processed_docs


def demo_basic_rag():
    """Demonstrate basic RAG functionality with new invoke() method."""
    print("=== Basic RAG Demo (with invoke()) ===")

    # Create configuration
    config = RAGConfig(
        name="demo_basic_rag",
        chunk_size=500,
        chunk_overlap=100,
        top_k=3,
        llm_model="gpt-3.5-turbo",
        temperature=0.1
    )

    # Create documents
    documents = create_sample_documents()
    processed_docs = process_documents(documents, config.chunk_size, config.chunk_overlap)

    print(f"Created {len(documents)} original documents")
    print(f"Processed into {len(processed_docs)} chunks")

    # Create and use RAG
    try:
        rag = BasicRAG(config)

        with rag:
            rag.add_documents(processed_docs)

            questions = [
                "What is LLMBlocks?",
                "What are the main components?",
                "What is the tech stack?"
            ]

            print("\n--- Using new invoke() method ---")
            for question in questions:
                print(f"\nQ: {question}")
                try:
                    answer = rag.invoke(question)
                    print(f"A: {answer}")
                except NotImplementedError as e:
                    print(f"Error: {e} (requires embedding setup)")

            print("\n--- Using backward compatible query() method ---")
            for question in questions:
                print(f"\nQ: {question}")
                try:
                    answer = rag.query(question)
                    print(f"A: {answer}")
                except NotImplementedError as e:
                    print(f"Error: {e} (requires embedding setup)")

            print("\n--- Batch processing with .batch() ---")
            try:
                responses = rag.batch(questions)
                for question, response in zip(questions, responses):
                    print(f"\nQ: {question}")
                    print(f"A: {response}")
            except NotImplementedError as e:
                print(f"Error: {e} (requires embedding setup)")

    except Exception as e:
        print(f"Error: {e}")


def demo_streaming_rag():
    """Demonstrate streaming RAG functionality with new stream() method."""
    print("\n=== Streaming RAG Demo (with stream()) ===")

    # Create configuration
    config = RAGConfig(
        name="demo_streaming_rag",
        chunk_size=500,
        chunk_overlap=100,
        top_k=3,
        llm_model="gpt-3.5-turbo",
        temperature=0.1,
        streaming_enabled=True
    )

    # Create documents
    documents = create_sample_documents()
    processed_docs = process_documents(documents, config.chunk_size, config.chunk_overlap)

    # Create and use streaming RAG
    try:
        rag = StreamingRAG(config)

        with rag:
            rag.add_documents(processed_docs)

            question = "Explain the project structure and main features"
            print(f"\nQ: {question}")

            print("--- Using new stream() method ---")
            print("A: ", end="", flush=True)
            try:
                for chunk in rag.stream(question):
                    print(chunk, end="", flush=True)
                print()  # New line after streaming
            except NotImplementedError as e:
                print(f"Error: {e} (requires embedding setup)")

            print("\n--- Using backward compatible query_stream() method ---")
            print("A: ", end="", flush=True)
            try:
                for chunk in rag.query_stream(question):
                    print(chunk, end="", flush=True)
                print()  # New line after streaming
            except NotImplementedError as e:
                print(f"Error: {e} (requires embedding setup)")

    except Exception as e:
        print(f"Error: {e}")


def demo_memory_rag():
    """Demonstrate memory RAG functionality with new invoke() method."""
    print("\n=== Memory RAG Demo (with invoke()) ===")

    # Create configuration
    config = RAGConfig(
        name="demo_memory_rag",
        chunk_size=500,
        chunk_overlap=100,
        top_k=3,
        llm_model="gpt-3.5-turbo",
        temperature=0.1,
        memory_enabled=True
    )

    # Create documents
    documents = create_sample_documents()
    processed_docs = process_documents(documents, config.chunk_size, config.chunk_overlap)

    # Create and use memory RAG
    try:
        rag = MemoryRAG(config)

        with rag:
            rag.add_documents(processed_docs)

            conversation = [
                "What is LLMBlocks?",
                "What are the main components mentioned?",
                "Can you tell me more about the RAG modules specifically?",
                "What about the memory systems you mentioned?"
            ]

            print("--- Using new invoke() method ---")
            for i, question in enumerate(conversation, 1):
                print(f"\nTurn {i}:")
                print(f"Q: {question}")

                try:
                    answer = rag.invoke(question)
                    print(f"A: {answer}")
                    print(f"Memory entries: {len(rag.conversation_history)}")
                except NotImplementedError as e:
                    print(f"Error: {e} (requires embedding setup)")
                    break

            print("\n--- Using backward compatible query() method ---")
            rag.clear_memory()  # Clear memory for fair comparison
            for i, question in enumerate(conversation, 1):
                print(f"\nTurn {i}:")
                print(f"Q: {question}")

                try:
                    answer = rag.query(question)
                    print(f"A: {answer}")
                    print(f"Memory entries: {len(rag.conversation_history)}")
                except NotImplementedError as e:
                    print(f"Error: {e} (requires embedding setup)")
                    break

    except Exception as e:
        print(f"Error: {e}")


def demo_runnable_interface():
    """Demonstrate the new Runnable interface capabilities."""
    print("\n=== Runnable Interface Demo ===")

    # Create configuration
    config = RAGConfig(
        name="demo_runnable",
        chunk_size=500,
        chunk_overlap=100,
        top_k=3,
        llm_model="gpt-3.5-turbo",
        temperature=0.1
    )

    # Create documents
    documents = create_sample_documents()
    processed_docs = process_documents(documents, config.chunk_size, config.chunk_overlap)

    try:
        rag = BasicRAG(config)

        with rag:
            rag.add_documents(processed_docs)

            print("--- Testing invoke() with different input types ---")
            try:
                # String input
                response1 = rag.invoke("What is LLMBlocks?")
                print(f"String input: {response1[:100]}...")

                # Dict input
                response2 = rag.invoke({"question": "What is LLMBlocks?"})
                print(f"Dict input: {response2[:100]}...")

                # Both should be the same
                print(f"Responses match: {response1 == response2}")

            except NotImplementedError as e:
                print(f"Error: {e} (requires embedding setup)")

            print("\n--- Testing batch() method ---")
            try:
                questions = ["What is LLMBlocks?", "What are the main components?", "What is the tech stack?"]
                responses = rag.batch(questions)

                print(f"Processed {len(questions)} questions in batch:")
                for i, (q, r) in enumerate(zip(questions, responses), 1):
                    print(f"  {i}. Q: {q[:50]}...")
                    print(f"     A: {r[:100]}...")

            except NotImplementedError as e:
                print(f"Error: {e} (requires embedding setup)")

            print("\n--- Testing stream() method ---")
            try:
                print("Streaming response: ", end="", flush=True)
                for chunk in rag.stream("Explain the project structure"):
                    print(chunk, end="", flush=True)
                print()  # New line

            except NotImplementedError as e:
                print(f"Error: {e} (requires embedding setup)")

    except Exception as e:
        print(f"Error: {e}")


def demo_config_loader():
    """Demonstrate configuration loader functionality."""
    print("\n=== Configuration Loader Demo ===")

    try:
        # Test config loader
        config_path = ConfigLoader.get_default_config_path()
        print(f"Default config path: {config_path}")

        config_files = ConfigLoader.list_config_files()
        print(f"Found {len(config_files)} config files")

        # Create a sample config
        sample_config = RAGConfig(
            name="sample_config",
            chunk_size=800,
            chunk_overlap=150,
            llm_model="gpt-4",
            temperature=0.2
        )

        # Save config
        temp_config_path = "temp_rag_config.yaml"
        ConfigLoader.save_config(sample_config, temp_config_path)
        print(f"Saved sample config to: {temp_config_path}")

        # Load config
        loaded_config = ConfigLoader.load_rag_config(temp_config_path)
        print(f"Loaded config: {loaded_config.name}")
        print(f"Chunk size: {loaded_config.chunk_size}")
        print(f"LLM model: {loaded_config.llm_model}")

        # Clean up
        Path(temp_config_path).unlink(missing_ok=True)
        print("Cleaned up temporary config file")

    except Exception as e:
        print(f"Error: {e}")


def demo_factory_function():
    """Demonstrate the factory function for creating RAG pipelines."""
    print("\n=== Factory Function Demo ===")

    # Test different RAG types
    rag_types = ["basic", "streaming", "memory"]

    for rag_type in rag_types:
        print(f"\nCreating {rag_type} RAG...")
        try:
            rag = create_rag(rag_type, name=f"demo_{rag_type}")
            print(f"✅ Created {type(rag).__name__}")
            print(f"   Config: {rag.config.name}")
            print(f"   Streaming: {rag.config.streaming_enabled}")
            print(f"   Memory: {rag.config.memory_enabled}")
            print(f"   Has invoke(): {hasattr(rag, 'invoke')}")
            print(f"   Has batch(): {hasattr(rag, 'batch')}")
            print(f"   Has stream(): {hasattr(rag, 'stream')}")
        except Exception as e:
            print(f"❌ Error creating {rag_type} RAG: {e}")


def main():
    """Run all demos."""
    print("🧩 LLMBlocks Playground Components Demo")
    print("=" * 60)
    print("This demo shows how to use playground components programmatically.")
    print("Now with LangChain Runnable interface support!")
    print("Note: Some features require proper embedding setup.")
    print("=" * 60)

    demos = [
        demo_basic_rag,
        demo_streaming_rag,
        demo_memory_rag,
        demo_runnable_interface,
        demo_config_loader,
        demo_factory_function
    ]

    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"Demo failed: {e}")
        print("\n" + "-" * 40)

    print("\n🎉 Demo completed!")
    print("\n💡 To see the interactive playground, run:")
    print("   python run_playground.py")
    print("\n🔄 New Runnable Interface Features:")
    print("   - .invoke() for single inputs")
    print("   - .batch() for multiple inputs")
    print("   - .stream() for streaming responses")
    print("   - | operator for chaining")


if __name__ == "__main__":
    main() 