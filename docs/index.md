# LLMBlocks Documentation

Welcome to the LLMBlocks documentation! This guide will help you understand and use the LLMBlocks framework effectively.

## 📚 Documentation Structure

### Core Documentation

- **[RAG Block Documentation](blocks/rag.md)** - Comprehensive guide to the RAG (Retrieval-Augmented Generation) block
- **[RAG Usage Guide](usage/rag_guide.md)** - Practical examples and best practices
- **[RAG API Reference](api/rag.md)** - Detailed API documentation

### Quick Links

- [Getting Started](#getting-started)
- [RAG Overview](#rag-overview)
- [LLM Providers](#llm-providers)
- [Configuration](#configuration)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/llmblocks/llmblocks.git
cd llmblocks

# Install dependencies
pip install -r requirements_providers.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
```

### Quick Example

```python
from llmblocks.blocks.rag import BasicRAG, RAGConfig
from langchain.schema import Document

# Create configuration
config = RAGConfig(
    name="quick_start",
    llm_provider="openai",
    llm_model="gpt-3.5-turbo"
)

# Create RAG pipeline
rag = BasicRAG(config)

# Add documents
documents = [
    Document(page_content="LLMBlocks is a modular framework for building LLM-powered applications.")
]

# Query the RAG pipeline
with rag:
    rag.add_documents(documents)
    response = rag.query("What is LLMBlocks?")
    print(response)
```

## 🧩 RAG Overview

The RAG (Retrieval-Augmented Generation) block is a core component of LLMBlocks that combines document retrieval with language model generation to provide accurate, contextual responses based on your knowledge base.

### Key Features

- **Multiple LLM Providers**: Support for OpenAI, Google, Hugging Face, Groq, Anthropic, and Ollama
- **Three RAG Types**: Basic, Streaming, and Memory-enabled RAG pipelines
- **Flexible Configuration**: YAML-based configuration with programmatic overrides
- **Production Ready**: Error handling, logging, and performance optimization
- **Extensible**: Easy to add new providers and customize behavior

### Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Documents     │    │   Text          │    │   Vector        │
│   (Input)       │───▶│   Splitter      │───▶│   Store         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM           │◀───│   RAG           │◀───│   Retriever     │
│   Provider      │    │   Pipeline      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Response      │◀───│   Prompt        │◀───│   Retrieved     │
│   (Output)      │    │   Template      │    │   Documents     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🤖 LLM Providers

LLMBlocks supports multiple LLM providers through a unified interface:

### Supported Providers

| Provider | Models | Environment Variable | Best For |
|----------|--------|---------------------|----------|
| **OpenAI** | GPT-3.5, GPT-4, GPT-4o | `OPENAI_API_KEY` | Production, reliability |
| **Google** | Gemini Pro, Gemini 1.5 | `GOOGLE_API_KEY` | Cost-effective, creative |
| **Hugging Face** | Llama, DialoGPT, GPT-2 | `HUGGINGFACE_API_KEY` | Open-source, customizable |
| **Groq** | Llama3, Mixtral | `GROQ_API_KEY` | Fast inference, real-time |
| **Anthropic** | Claude 3 Sonnet, Haiku | `ANTHROPIC_API_KEY` | Reasoning, analysis |
| **Ollama** | Llama2, Mistral, CodeLlama | None | Local deployment, privacy |

### Provider Selection Guide

- **Production Applications**: OpenAI or Anthropic
- **Cost-Sensitive**: Google Gemini or Hugging Face
- **Real-Time Applications**: Groq
- **Privacy-Focused**: Ollama (local)
- **Custom Models**: Hugging Face

## ⚙️ Configuration

### YAML Configuration

```yaml
rag:
  name: "production_rag"
  description: "Production RAG pipeline for document Q&A"
  
  # Document processing
  chunk_size: 1000
  chunk_overlap: 200
  text_splitter_type: "recursive"
  
  # Vector store configuration
  vector_store_type: "chroma"
  embedding_model: "text-embedding-ada-002"
  persist_directory: "./data/vectorstore"
  
  # LLM configuration
  llm_provider: "openai"
  llm_model: "gpt-4"
  llm_api_key: null  # Set via environment variable
  temperature: 0.1
  max_tokens: 1000
  
  # Retrieval configuration
  top_k: 4
  similarity_threshold: 0.7
  
  # Memory configuration
  memory_enabled: true
  memory_type: "in_memory"
  
  # Streaming configuration
  streaming_enabled: false
```

### Programmatic Configuration

```python
from llmblocks.blocks.rag import RAGConfig

config = RAGConfig(
    name="my_rag",
    llm_provider="openai",
    llm_model="gpt-4",
    chunk_size=1000,
    chunk_overlap=200,
    top_k=4,
    temperature=0.1,
    memory_enabled=True,
    streaming_enabled=False
)
```

## 📖 Examples

### Basic RAG

```python
from llmblocks.blocks.rag import BasicRAG, RAGConfig

config = RAGConfig(
    name="basic_rag",
    llm_provider="openai",
    llm_model="gpt-4"
)

rag = BasicRAG(config)

with rag:
    rag.add_documents(documents)
    response = rag.query("What is the main topic?")
    print(response)
```

### Streaming RAG

```python
from llmblocks.blocks.rag import StreamingRAG, RAGConfig

config = RAGConfig(
    name="streaming_rag",
    llm_provider="openai",
    llm_model="gpt-4",
    streaming_enabled=True
)

rag = StreamingRAG(config)

with rag:
    rag.add_documents(documents)
    
    for chunk in rag.query_stream("Explain in detail"):
        print(chunk, end="", flush=True)
```

### Memory RAG

```python
from llmblocks.blocks.rag import MemoryRAG, RAGConfig

config = RAGConfig(
    name="memory_rag",
    llm_provider="openai",
    llm_model="gpt-4",
    memory_enabled=True
)

rag = MemoryRAG(config)

with rag:
    rag.add_documents(documents)
    
    # Multi-turn conversation
    response1 = rag.query("What is the main topic?")
    response2 = rag.query("Can you elaborate?")  # Uses conversation history
```

### Multi-Provider Example

```python
# OpenAI
openai_config = RAGConfig(
    llm_provider="openai",
    llm_model="gpt-4"
)

# Google Gemini
google_config = RAGConfig(
    llm_provider="google",
    llm_model="gemini-pro"
)

# Hugging Face
hf_config = RAGConfig(
    llm_provider="huggingface",
    llm_model="meta-llama/Llama-2-7b-chat-hf",
    llm_task="text-generation"
)

# Groq
groq_config = RAGConfig(
    llm_provider="groq",
    llm_model="llama3-8b-8192"
)

# Ollama (Local)
ollama_config = RAGConfig(
    llm_provider="ollama",
    llm_model="llama2",
    llm_base_url="http://localhost:11434"
)
```

## 🔧 Best Practices

### 1. Chunk Size Optimization

- **Small chunks (500-1000)**: Better for specific questions, higher precision
- **Large chunks (1000-2000)**: Better for comprehensive answers, higher recall
- **Overlap (10-20% of chunk size)**: Maintains context across chunks

### 2. Provider Selection

- **OpenAI**: Best for production, reliable, good performance
- **Google Gemini**: Cost-effective, good for creative tasks
- **Hugging Face**: Open-source models, customizable
- **Groq**: Fast inference, good for real-time applications
- **Anthropic**: Strong reasoning capabilities
- **Ollama**: Local deployment, privacy-focused

### 3. Performance Optimization

- Use appropriate `top_k` values (2-8 for most cases)
- Enable streaming for better user experience
- Persist vector store for large document collections
- Use similarity thresholds to filter low-quality matches

### 4. Error Handling

```python
try:
    with rag:
        rag.add_documents(documents)
        response = rag.query("Your question")
except NotImplementedError as e:
    print(f"Embedding setup required: {e}")
except Exception as e:
    print(f"RAG error: {e}")
```

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements_providers.txt
   ```

2. **API Key Issues**
   ```bash
   export OPENAI_API_KEY="your-key"
   export GOOGLE_API_KEY="your-key"
   # etc.
   ```

3. **Embedding Errors**
   ```python
   # Implement embedding setup
   def _get_embeddings(self):
       # Return your embedding model
       pass
   ```

4. **Memory Issues**
   ```python
   rag.clear_memory()
   ```

5. **Ollama Connection**
   ```bash
   ollama serve
   ollama pull llama2
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your RAG code here
```

## 📚 Additional Resources

### Documentation

- **[RAG Block Documentation](blocks/rag.md)** - Complete RAG block guide
- **[RAG Usage Guide](usage/rag_guide.md)** - Practical examples and patterns
- **[RAG API Reference](api/rag.md)** - Detailed API documentation

### Examples

- **[Basic RAG Example](../../examples/basic_rag.py)** - Simple RAG implementation
- **[Multi-Provider Demo](../../examples/multi_provider_demo.py)** - All providers demonstration
- **[Playground Demo](../../examples/playground_demo.py)** - Interactive playground usage

### Configuration Files

- **[OpenAI Config](../../llmblocks/config/openai_rag.yaml)** - OpenAI configuration
- **[Google Config](../../llmblocks/config/google_rag.yaml)** - Google Gemini configuration
- **[Hugging Face Config](../../llmblocks/config/huggingface_rag.yaml)** - Hugging Face configuration
- **[Groq Config](../../llmblocks/config/groq_rag.yaml)** - Groq configuration
- **[Ollama Config](../../llmblocks/config/ollama_rag.yaml)** - Ollama configuration

### Dependencies

- **[Provider Requirements](../../requirements_providers.txt)** - Additional dependencies for LLM providers

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/llmblocks/llmblocks.git
cd llmblocks
pip install -r requirements_providers.txt
pip install -e .
```

### Running Tests

```bash
pytest tests/
pytest tests/core/test_llm_providers.py
```

## 📞 Support

- 📧 Email: team@llmblocks.dev
- 🐛 Issues: [GitHub Issues](https://github.com/llmblocks/llmblocks/issues)
- 📖 Documentation: [llmblocks.dev](https://llmblocks.dev)

---

**Made with ❤️ by the LLMBlocks Team**