# RAG Usage Guide

## Quick Start

### 1. Basic Setup

LLMBlocks RAG components now support LangChain's Runnable interface with `.invoke()`, `.batch()`, and `.stream()` methods, while maintaining backward compatibility.

```python
from llmblocks.blocks.rag import BasicRAG, RAGConfig
from langchain.schema import Document

# Create configuration
config = RAGConfig(
    name="my_first_rag",
    llm_provider="openai",
    llm_model="gpt-3.5-turbo",
    chunk_size=1000,
    top_k=4
)

# Create RAG pipeline
rag = BasicRAG(config)

# Prepare documents
documents = [
    Document(
        page_content="LLMBlocks is a modular framework for building LLM-powered applications.",
        metadata={"source": "introduction"}
    )
]

# Use the RAG pipeline with new invoke() method
with rag:
    rag.add_documents(documents)
    
    # New Runnable interface
    response = rag.invoke("What is LLMBlocks?")
    print(response)
    
    # Backward compatibility
    response = rag.query("What is LLMBlocks?")
    print(response)
    
    # Batch processing
    questions = ["What is LLMBlocks?", "What are its features?"]
    responses = rag.batch(questions)
    for q, r in zip(questions, responses):
        print(f"Q: {q}\nA: {r}\n")
```

### 2. Environment Setup

Set your API keys:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# Google Gemini
export GOOGLE_API_KEY="your-google-api-key"

# Hugging Face
export HUGGINGFACE_API_KEY="your-huggingface-api-key"

# Groq
export GROQ_API_KEY="your-groq-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Install dependencies:

```bash
pip install -r requirements_providers.txt
```

## Document Processing

### Loading Documents

```python
from langchain.schema import Document
from pathlib import Path

# From text
documents = [
    Document(
        page_content="Your document content here",
        metadata={"source": "manual_input", "type": "text"}
    )
]

# From files
def load_documents_from_directory(directory: str) -> List[Document]:
    docs = []
    for file_path in Path(directory).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            docs.append(Document(
                page_content=content,
                metadata={"source": str(file_path), "type": "file"}
            ))
    return docs

# Load documents
documents = load_documents_from_directory("./documents")
```

### Document Chunking

The RAG pipeline automatically chunks documents, but you can control the process:

```python
config = RAGConfig(
    chunk_size=500,      # Smaller chunks for precise answers
    chunk_overlap=100,   # 20% overlap to maintain context
    text_splitter_type="recursive"
)
```

**Chunk Size Guidelines:**
- **500-800**: For specific, factual questions
- **1000-1500**: For general questions (default)
- **1500-2000**: For comprehensive answers

## LLM Provider Selection

### Provider Comparison

| Provider | Best For | Pros | Cons |
|----------|----------|------|------|
| **OpenAI** | Production, reliability | High quality, reliable, good performance | Expensive |
| **Google Gemini** | Cost-effective, creative | Good value, creative responses | Less reliable than OpenAI |
| **Hugging Face** | Open-source, customizable | Free models, customizable | Requires more setup |
| **Groq** | Fast inference | Very fast, good for real-time | Limited model selection |
| **Anthropic** | Reasoning, analysis | Strong reasoning, good for analysis | Expensive |
| **Ollama** | Privacy, local deployment | Free, private, offline | Requires local setup |

### Provider-Specific Examples

```python
# OpenAI (Production)
openai_config = RAGConfig(
    llm_provider="openai",
    llm_model="gpt-4",
    temperature=0.1,
    max_tokens=1000
)

# Google Gemini (Cost-effective)
google_config = RAGConfig(
    llm_provider="google",
    llm_model="gemini-pro",
    temperature=0.2,
    max_tokens=1000
)

# Hugging Face (Open-source)
hf_config = RAGConfig(
    llm_provider="huggingface",
    llm_model="meta-llama/Llama-2-7b-chat-hf",
    llm_task="text-generation",
    temperature=0.1,
    max_tokens=1000
)

# Groq (Fast)
groq_config = RAGConfig(
    llm_provider="groq",
    llm_model="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000
)

# Anthropic (Reasoning)
anthropic_config = RAGConfig(
    llm_provider="anthropic",
    llm_model="claude-3-sonnet-20240229",
    temperature=0.1,
    max_tokens=1000
)

# Ollama (Local)
ollama_config = RAGConfig(
    llm_provider="ollama",
    llm_model="llama2",
    llm_base_url="http://localhost:11434",
    temperature=0.1,
    max_tokens=1000
)
```

## LangChain Runnable Interface

LLMBlocks RAG components now implement LangChain's Runnable interface, providing standardized execution methods and enabling composition with other LangChain components.

### Core Methods

```python
# Single execution
response = rag.invoke("What is LLMBlocks?")

# Batch processing
questions = ["Q1", "Q2", "Q3"]
responses = rag.batch(questions)

# Streaming
for chunk in rag.stream("Explain in detail"):
    print(chunk, end="", flush=True)
```

### Input Formats

The `invoke()` method accepts multiple input formats:

```python
# String input (most common)
response = rag.invoke("What is LLMBlocks?")

# Dictionary input
response = rag.invoke({"question": "What is LLMBlocks?"})

# With configuration overrides
response = rag.invoke("What is LLMBlocks?", {"temperature": 0.5})
```

### Chaining Capabilities

RAG components can be chained with other Runnable components:

```python
# Example chaining (requires proper LangChain integration)
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Create a post-processor
post_processor = PromptTemplate(
    input_variables=["answer"],
    template="Summarize this answer: {answer}"
)

# Chain RAG with post-processor
pipeline = rag | post_processor | OpenAI()
result = pipeline.invoke("What is LLMBlocks?")
```

## RAG Pipeline Types

### 1. Basic RAG

Use for simple Q&A applications:

```python
from llmblocks.blocks.rag import BasicRAG

config = RAGConfig(
    name="basic_qa",
    llm_provider="openai",
    llm_model="gpt-4"
)

rag = BasicRAG(config)

with rag:
    rag.add_documents(documents)
    
    # New invoke() method
    answer = rag.invoke("What is the main topic?")
    print(answer)
    
    # Backward compatibility
    answer = rag.query("What is the main topic?")
    print(answer)
```

### 2. Streaming RAG

Use for better user experience:

```python
from llmblocks.blocks.rag import StreamingRAG

config = RAGConfig(
    name="streaming_qa",
    llm_provider="openai",
    llm_model="gpt-4",
    streaming_enabled=True
)

rag = StreamingRAG(config)

with rag:
    rag.add_documents(documents)
    
    # New stream() method
    print("Answer (stream): ", end="", flush=True)
    for chunk in rag.stream("Explain in detail"):
        print(chunk, end="", flush=True)
    print()
    
    # Backward compatibility
    print("Answer (query_stream): ", end="", flush=True)
    for chunk in rag.query_stream("Explain in detail"):
        print(chunk, end="", flush=True)
    print()
    
    # Non-streaming invoke() method
    response = rag.invoke("Explain in detail")
    print(f"Complete response: {response}")
```

### 3. Memory RAG

Use for conversational applications:

```python
from llmblocks.blocks.rag import MemoryRAG

config = RAGConfig(
    name="conversation_qa",
    llm_provider="openai",
    llm_model="gpt-4",
    memory_enabled=True,
    memory_type="in_memory"
)

rag = MemoryRAG(config)

with rag:
    rag.add_documents(documents)
    
    # Multi-turn conversation using invoke()
    response1 = rag.invoke("What is the main topic?")
    print(f"Q1: What is the main topic?")
    print(f"A1: {response1}")
    
    response2 = rag.invoke("Can you elaborate on that?")
    print(f"Q2: Can you elaborate on that?")
    print(f"A2: {response2}")
    
    # Backward compatibility
    response3 = rag.query("What else should I know?")
    print(f"Q3: What else should I know?")
    print(f"A3: {response3}")
    
    # Clear memory when needed
    rag.clear_memory()
```

## Real-World Examples

### 1. Document Q&A System

```python
from llmblocks.blocks.rag import BasicRAG, RAGConfig
from langchain.schema import Document
from pathlib import Path

def create_document_qa_system(documents_dir: str):
    """Create a document Q&A system."""
    
    # Load documents
    documents = []
    for file_path in Path(documents_dir).glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append(Document(
                page_content=content,
                metadata={"source": str(file_path)}
            ))
    
    # Create RAG configuration
    config = RAGConfig(
        name="document_qa",
        llm_provider="openai",
        llm_model="gpt-4",
        chunk_size=1000,
        chunk_overlap=200,
        top_k=4,
        temperature=0.1
    )
    
    # Create and initialize RAG
    rag = BasicRAG(config)
    
    with rag:
        rag.add_documents(documents)
        
        # Interactive Q&A
        while True:
            question = input("\nAsk a question (or 'quit' to exit): ")
            if question.lower() == 'quit':
                break
                
            try:
                answer = rag.query(question)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                print(f"Error: {e}")

# Usage
create_document_qa_system("./documents")
```

### 2. Customer Support Chatbot

```python
from llmblocks.blocks.rag import MemoryRAG, RAGConfig

def create_support_chatbot(knowledge_base: List[Document]):
    """Create a customer support chatbot with memory."""
    
    config = RAGConfig(
        name="support_chatbot",
        llm_provider="openai",
        llm_model="gpt-4",
        memory_enabled=True,
        memory_type="in_memory",
        temperature=0.2,  # Slightly more conversational
        max_tokens=800
    )
    
    rag = MemoryRAG(config)
    
    with rag:
        rag.add_documents(knowledge_base)
        
        print("Customer Support Chatbot")
        print("Type 'quit' to exit, 'clear' to clear memory")
        print("-" * 40)
        
        while True:
            user_input = input("\nCustomer: ")
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                rag.clear_memory()
                print("Memory cleared.")
                continue
            
            try:
                response = rag.query(user_input)
                print(f"Support: {response}")
            except Exception as e:
                print(f"Error: {e}")

# Usage
support_docs = [
    Document(page_content="Our return policy allows returns within 30 days..."),
    Document(page_content="To contact support, email support@company.com..."),
    # ... more support documents
]

create_support_chatbot(support_docs)
```

## Best Practices

### 1. Performance Optimization

```python
# Optimize for speed
fast_config = RAGConfig(
    llm_provider="groq",
    llm_model="llama3-8b-8192",
    chunk_size=800,
    top_k=3
)

# Optimize for accuracy
accurate_config = RAGConfig(
    llm_provider="openai",
    llm_model="gpt-4",
    chunk_size=1200,
    top_k=6,
    similarity_threshold=0.8
)

# Optimize for cost
cost_effective_config = RAGConfig(
    llm_provider="google",
    llm_model="gemini-pro",
    chunk_size=1000,
    top_k=4
)
```

### 2. Error Handling

```python
import logging
from llmblocks.blocks.rag import BasicRAG, RAGConfig

# Setup logging
logging.basicConfig(level=logging.INFO)

def create_safe_rag(config: RAGConfig, documents: List[Document]):
    """Create RAG pipeline with comprehensive error handling."""
    
    try:
        rag = BasicRAG(config)
        
        with rag:
            # Add documents
            if not documents:
                raise ValueError("No documents provided")
            
            rag.add_documents(documents)
            
            # Test query
            test_response = rag.query("Test question")
            if not test_response:
                raise ValueError("RAG pipeline not responding")
            
            return rag
            
    except NotImplementedError as e:
        print(f"Embedding setup required: {e}")
        print("Please implement the _get_embeddings method")
        return None
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
```

### 3. Testing

```python
import pytest
from llmblocks.blocks.rag import BasicRAG, RAGConfig

def test_basic_rag():
    """Test basic RAG functionality."""
    
    config = RAGConfig(
        name="test_rag",
        llm_provider="openai",
        llm_model="gpt-3.5-turbo"
    )
    
    rag = BasicRAG(config)
    
    documents = [
        Document(
            page_content="LLMBlocks is a modular framework for building LLM applications.",
            metadata={"source": "test"}
        )
    ]
    
    with rag:
        rag.add_documents(documents)
        
        response = rag.query("What is LLMBlocks?")
        
        assert response is not None
        assert len(response) > 0
        assert "LLMBlocks" in response or "framework" in response.lower()
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Import errors | Missing dependencies | `pip install -r requirements_providers.txt` |
| API key errors | Missing environment variables | Set appropriate API keys |
| Embedding errors | Not implemented | Implement `_get_embeddings()` method |
| Memory issues | Context overflow | Clear memory periodically |
| Slow responses | Large chunks or high top_k | Optimize chunk size and top_k |
| Poor quality answers | Wrong chunk size | Adjust chunk size for your use case |
| Ollama connection | Service not running | Run `ollama serve` |

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Your RAG code here
```

This comprehensive usage guide covers all aspects of using the RAG block effectively, from basic setup to advanced configurations and real-world applications.
