"""
Multi-Provider Demo - Showcase different LLM providers in LLMBlocks.

This demo shows how to use different LLM providers including:
- OpenAI
- Google Gemini
- Hugging Face
- Groq
- Anthropic
- Ollama

Each provider has different configuration requirements and capabilities.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llmblocks.blocks.rag import BasicRAG, RAGConfig, create_rag
from llmblocks.core.config_loader import ConfigLoader
from llmblocks.core.llm_providers import LLMProviderFactory
from langchain.schema import Document


def create_sample_documents() -> List[Document]:
    """Create sample documents for demonstration."""
    return [
        Document(
            page_content="LLMBlocks is a modular framework for building LLM-powered applications. It provides pre-built components for RAG, agents, and other AI workflows.",
            metadata={"source": "introduction", "type": "overview"}
        ),
        Document(
            page_content="The framework supports multiple LLM providers including OpenAI, Google, Hugging Face, Groq, Anthropic, and Ollama. Each provider has different capabilities and pricing.",
            metadata={"source": "providers", "type": "technical"}
        ),
        Document(
            page_content="RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation to provide accurate, contextual responses based on your knowledge base.",
            metadata={"source": "rag", "type": "concept"}
        )
    ]


def demo_provider_info():
    """Display information about available providers."""
    print("\n=== Available LLM Providers ===")
    
    for provider_name in LLMProviderFactory.list_providers():
        try:
            info = LLMProviderFactory.get_provider_info(provider_name)
            status = "✅ Available" if info['available'] else "❌ Not Available"
            print(f"\n{provider_name.upper()}:")
            print(f"  Status: {status}")
            print(f"  Required Env Vars: {info['required_env_vars']}")
        except Exception as e:
            print(f"\n{provider_name.upper()}: Error getting info - {e}")


def demo_openai_rag():
    """Demonstrate OpenAI RAG pipeline."""
    print("\n=== OpenAI RAG Demo ===")
    
    try:
        # Check if OpenAI API key is available
        if not os.getenv('OPENAI_API_KEY'):
            print("⚠️  OPENAI_API_KEY not set. Skipping OpenAI demo.")
            return
        
        config = RAGConfig(
            name="openai_demo",
            llm_provider="openai",
            llm_model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=500
        )
        
        rag = create_rag("basic", config)
        documents = create_sample_documents()
        
        with rag:
            rag.add_documents(documents)
            
            question = "What is LLMBlocks and what providers does it support?"
            print(f"Question: {question}")
            
            answer = rag.query(question)
            print(f"Answer: {answer}")
            
    except Exception as e:
        print(f"❌ OpenAI demo failed: {e}")


def demo_google_rag():
    """Demonstrate Google Gemini RAG pipeline."""
    print("\n=== Google Gemini RAG Demo ===")
    
    try:
        # Check if Google API key is available
        if not os.getenv('GOOGLE_API_KEY'):
            print("⚠️  GOOGLE_API_KEY not set. Skipping Google demo.")
            return
        
        config = RAGConfig(
            name="google_demo",
            llm_provider="google",
            llm_model="gemini-pro",
            temperature=0.1,
            max_tokens=500
        )
        
        rag = create_rag("basic", config)
        documents = create_sample_documents()
        
        with rag:
            rag.add_documents(documents)
            
            question = "Explain RAG and how it works in LLMBlocks."
            print(f"Question: {question}")
            
            answer = rag.query(question)
            print(f"Answer: {answer}")
            
    except Exception as e:
        print(f"❌ Google demo failed: {e}")


def demo_huggingface_rag():
    """Demonstrate Hugging Face RAG pipeline."""
    print("\n=== Hugging Face RAG Demo ===")
    
    try:
        # Check if Hugging Face API key is available
        if not os.getenv('HUGGINGFACE_API_KEY'):
            print("⚠️  HUGGINGFACE_API_KEY not set. Skipping Hugging Face demo.")
            return
        
        config = RAGConfig(
            name="huggingface_demo",
            llm_provider="huggingface",
            llm_model="microsoft/DialoGPT-medium",
            llm_task="text-generation",
            temperature=0.1,
            max_tokens=500
        )
        
        rag = create_rag("basic", config)
        documents = create_sample_documents()
        
        with rag:
            rag.add_documents(documents)
            
            question = "What are the main features of LLMBlocks?"
            print(f"Question: {question}")
            
            answer = rag.query(question)
            print(f"Answer: {answer}")
            
    except Exception as e:
        print(f"❌ Hugging Face demo failed: {e}")


def demo_groq_rag():
    """Demonstrate Groq RAG pipeline."""
    print("\n=== Groq RAG Demo ===")
    
    try:
        # Check if Groq API key is available
        if not os.getenv('GROQ_API_KEY'):
            print("⚠️  GROQ_API_KEY not set. Skipping Groq demo.")
            return
        
        config = RAGConfig(
            name="groq_demo",
            llm_provider="groq",
            llm_model="llama3-8b-8192",
            temperature=0.1,
            max_tokens=500
        )
        
        rag = create_rag("basic", config)
        documents = create_sample_documents()
        
        with rag:
            rag.add_documents(documents)
            
            question = "How does LLMBlocks handle different providers?"
            print(f"Question: {question}")
            
            answer = rag.query(question)
            print(f"Answer: {answer}")
            
    except Exception as e:
        print(f"❌ Groq demo failed: {e}")


def demo_anthropic_rag():
    """Demonstrate Anthropic RAG pipeline."""
    print("\n=== Anthropic RAG Demo ===")
    
    try:
        # Check if Anthropic API key is available
        if not os.getenv('ANTHROPIC_API_KEY'):
            print("⚠️  ANTHROPIC_API_KEY not set. Skipping Anthropic demo.")
            return
        
        config = RAGConfig(
            name="anthropic_demo",
            llm_provider="anthropic",
            llm_model="claude-3-haiku-20240307",
            temperature=0.1,
            max_tokens=500
        )
        
        rag = create_rag("basic", config)
        documents = create_sample_documents()
        
        with rag:
            rag.add_documents(documents)
            
            question = "What makes LLMBlocks modular and extensible?"
            print(f"Question: {question}")
            
            answer = rag.query(question)
            print(f"Answer: {answer}")
            
    except Exception as e:
        print(f"❌ Anthropic demo failed: {e}")


def demo_ollama_rag():
    """Demonstrate Ollama RAG pipeline."""
    print("\n=== Ollama RAG Demo ===")
    
    try:
        config = RAGConfig(
            name="ollama_demo",
            llm_provider="ollama",
            llm_model="llama2",
            llm_base_url="http://localhost:11434",
            temperature=0.1,
            max_tokens=500
        )
        
        rag = create_rag("basic", config)
        documents = create_sample_documents()
        
        with rag:
            rag.add_documents(documents)
            
            question = "What is the purpose of the LLMBlocks framework?"
            print(f"Question: {question}")
            
            answer = rag.query(question)
            print(f"Answer: {answer}")
            
    except Exception as e:
        print(f"❌ Ollama demo failed: {e}")
        print("💡 Make sure Ollama is running locally with: ollama serve")


def demo_config_loading():
    """Demonstrate loading configurations for different providers."""
    print("\n=== Configuration Loading Demo ===")
    
    try:
        # Load different provider configurations
        config_files = [
            "llmblocks/config/openai_rag.yaml",
            "llmblocks/config/google_rag.yaml",
            "llmblocks/config/huggingface_rag.yaml",
            "llmblocks/config/groq_rag.yaml",
            "llmblocks/config/ollama_rag.yaml"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                print(f"\nLoading {config_file}...")
                config = ConfigLoader.load_rag_config(config_file)
                print(f"  Provider: {config.llm_provider}")
                print(f"  Model: {config.llm_model}")
                print(f"  Temperature: {config.temperature}")
            else:
                print(f"\n⚠️  Config file not found: {config_file}")
                
    except Exception as e:
        print(f"❌ Config loading demo failed: {e}")


def main():
    """Run all provider demonstrations."""
    print("🧩 LLMBlocks Multi-Provider Demo")
    print("=" * 50)
    
    # Show provider information
    demo_provider_info()
    
    # Run provider demos
    demos = [
        demo_openai_rag,
        demo_google_rag,
        demo_huggingface_rag,
        demo_groq_rag,
        demo_anthropic_rag,
        demo_ollama_rag
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    # Configuration loading demo
    demo_config_loading()
    
    print("\n" + "=" * 50)
    print("🎉 Multi-provider demo completed!")
    print("\n💡 Tips:")
    print("  - Set environment variables for API keys")
    print("  - Install required packages: pip install -r requirements_providers.txt")
    print("  - For Ollama: run 'ollama serve' and pull models with 'ollama pull llama2'")


if __name__ == "__main__":
    main() 