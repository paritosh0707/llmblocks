# RAG (Retrieval-Augmented Generation) Block

## Overview

The RAG (Retrieval-Augmented Generation) block is a core component of LLMBlocks that combines document retrieval with language model generation to provide accurate, contextual responses based on your knowledge base. This implementation supports multiple LLM providers and provides various RAG pipeline types to suit different use cases.

## Architecture

The RAG block follows a modular architecture with the following key components:

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

### Components

1. **Text Splitter**: Breaks documents into manageable chunks
2. **Vector Store**: Stores document embeddings for similarity search
3. **Retriever**: Finds relevant documents based on query
4. **LLM Provider**: Generates responses using various language models
5. **Prompt Template**: Structures the input for the LLM
6. **Memory System**: (Optional) Maintains conversation history

## Runnable Interface & Composability

LLMBlocks RAG blocks implement the LangChain `Runnable` interface, enabling:
- `.invoke(input)`: Standard single input execution (main method)
- `.batch(inputs)`: Batch processing of multiple inputs
- `.stream(input)`: Streaming output for real-time applications
- Chaining with the `|` operator for composable pipelines

**Example:**
```python
response = rag.invoke("What is LLMBlocks?")
responses = rag.batch(["Q1", "Q2"])
for chunk in rag.stream("Explain in detail"): print(chunk, end="")

# Chaining (with other Runnable components)
pipeline = rag | some_post_processor
result = pipeline.invoke("What is LLMBlocks?")
```

> **Note:** `.query()` and `.query_stream()` are still available for backward compatibility, but `.invoke()` and `.stream()` are now the recommended interfaces.

## RAG Pipeline Types

### 1. BasicRAG

Simple retrieval and generation pipeline suitable for most use cases.

```python
from llmblocks.blocks.rag import BasicRAG, RAGConfig

config = RAGConfig(
    name="basic_rag",
    llm_provider="openai",
    llm_model="gpt-4",
    chunk_size=1000,
    chunk_overlap=200,
    top_k=4
)

rag = BasicRAG(config)

with rag:
    rag.add_documents(documents)
    # Main method
    response = rag.invoke("What is LLMBlocks?")
    print(response)
    # Backward compatibility
    response = rag.query("What is LLMBlocks?")
    print(response)
```

**Features:**
- Document chunking and embedding
- Similarity-based retrieval
- Context-aware response generation
- Configurable chunk size and overlap
- Adjustable number of retrieved documents
- **Composable:** Supports chaining with `|` and integration with other LangChain Runnables

### 2. StreamingRAG

Provides real-time streaming responses for better user experience.

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
    # Main streaming method
    print("Response: ", end="", flush=True)
    for chunk in rag.stream("What is LLMBlocks?"):
        print(chunk, end="", flush=True)
    print()
    # Backward compatibility
    for chunk in rag.query_stream("What is LLMBlocks?"):
        print(chunk, end="", flush=True)
    print()
```

**Features:**
- Real-time response streaming
- Progressive response generation
- Better user experience for long responses
- Compatible with all supported LLM providers
- **Composable:** Can be chained with other streaming or post-processing blocks

### 3. MemoryRAG

Maintains conversation history for contextual multi-turn conversations.

```python
from llmblocks.blocks.rag import MemoryRAG, RAGConfig

config = RAGConfig(
    name="memory_rag",
    llm_provider="openai",
    llm_model="gpt-4",
    memory_enabled=True,
    memory_type="in_memory"
)

rag = MemoryRAG(config)

with rag:
    rag.add_documents(documents)
    # Multi-turn conversation
    answer1 = rag.invoke("What is the main topic?")
    answer2 = rag.invoke("Can you elaborate on that?")
    print(answer1, answer2)
    # Backward compatibility
    answer3 = rag.query("What else should I know?")
    print(answer3)
```

**Features:**
- Conversation memory management
- Context-aware follow-up questions
- Memory persistence across sessions
- Configurable memory types
- **Composable:** Can be part of a larger conversational pipeline

## Supported LLM Providers

### OpenAI

```python
config = RAGConfig(
    llm_provider="openai",
    llm_model="gpt-4",
    temperature=0.1,
    max_tokens=1000
)
```

**Available Models:**
- `gpt-3.5-turbo`
- `gpt-4`
- `gpt-4-turbo`
- `gpt-4o`

**Environment Variable:** `OPENAI_API_KEY`

### Google Gemini

```python
config = RAGConfig(
    llm_provider="google",
    llm_model="gemini-pro",
    temperature=0.1,
    max_tokens=1000
)
```

**Available Models:**
- `gemini-pro`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

**Environment Variable:** `GOOGLE_API_KEY`

### Hugging Face

```python
config = RAGConfig(
    llm_provider="huggingface",
    llm_model="meta-llama/Llama-2-7b-chat-hf",
    llm_task="text-generation",
    temperature=0.1,
    max_tokens=1000
)
```

**Available Models:**
- `meta-llama/Llama-2-7b-chat-hf`
- `microsoft/DialoGPT-medium`
- `gpt2`

**Tasks:**
- `text-generation`
- `conversational`

**Environment Variable:** `HUGGINGFACE_API_KEY`

### Groq

```python
config = RAGConfig(
    llm_provider="groq",
    llm_model="llama3-8b-8192",
    temperature=0.1,
    max_tokens=1000
)
```

**Available Models:**
- `llama3-8b-8192`
- `llama3-70b-8192`
- `mixtral-8x7b-32768`

**Environment Variable:** `GROQ_API_KEY`

### Anthropic

```python
config = RAGConfig(
    llm_provider="anthropic",
    llm_model="claude-3-sonnet-20240229",
    temperature=0.1,
    max_tokens=1000
)
```

**Available Models:**
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- `claude-3-opus-20240229`

**Environment Variable:** `ANTHROPIC_API_KEY`

### Ollama (Local)

```python
config = RAGConfig(
    llm_provider="ollama",
    llm_model="llama2",
    llm_base_url="http://localhost:11434",
    temperature=0.1,
    max_tokens=1000
)
```

**Available Models:**
- `llama2`
- `llama2:13b`
- `llama2:70b`
- `mistral`
- `codellama`

**No API key required** - runs locally with Ollama

## Configuration

### RAGConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | - | Name of the RAG pipeline |
| `description` | str | None | Description of the pipeline |
| `chunk_size` | int | 1000 | Size of document chunks |
| `chunk_overlap` | int | 200 | Overlap between chunks |
| `text_splitter_type` | str | "recursive" | Type of text splitter |
| `vector_store_type` | str | "chroma" | Vector store backend |
| `embedding_model` | str | "text-embedding-ada-002" | Embedding model |
| `persist_directory` | str | None | Directory to persist vector store |
| `llm_provider` | str | "openai" | LLM provider to use |
| `llm_model` | str | "gpt-3.5-turbo" | Model name |
| `llm_api_key` | str | None | API key (optional, can use env vars) |
| `llm_base_url` | str | None | Base URL for local providers |
| `llm_task` | str | None | Task type for Hugging Face models |
| `temperature` | float | 0.0 | Response creativity (0-1) |
| `max_tokens` | int | 1000 | Maximum response length |
| `top_k` | int | 4 | Number of documents to retrieve |
| `similarity_threshold` | float | 0.7 | Minimum similarity score |
| `memory_enabled` | bool | False | Enable conversation memory |
| `memory_type` | str | "in_memory" | Memory storage type |
| `streaming_enabled` | bool | False | Enable streaming responses |

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

## Usage Examples

### Basic Usage

```python
from llmblocks.blocks.rag import BasicRAG, RAGConfig
from langchain.schema import Document

config = RAGConfig(
    name="my_rag",
    llm_provider="openai",
    llm_model="gpt-4",
    chunk_size=1000,
    top_k=4
)

rag = BasicRAG(config)

documents = [
    Document(
        page_content="LLMBlocks is a modular framework for building LLM-powered applications.",
        metadata={"source": "introduction"}
    ),
    Document(
        page_content="The framework supports multiple LLM providers and RAG pipelines.",
        metadata={"source": "features"}
    )
]

with rag:
    rag.add_documents(documents)
    # Main method
    response = rag.invoke("What is LLMBlocks?")
    print(response)
```

### Advanced Usage with Memory

```python
from llmblocks.blocks.rag import MemoryRAG, RAGConfig

config = RAGConfig(
    name="conversation_rag",
    llm_provider="openai",
    llm_model="gpt-4",
    memory_enabled=True,
    memory_type="in_memory"
)

rag = MemoryRAG(config)

with rag:
    rag.add_documents(documents)
    # Multi-turn conversation
    response1 = rag.invoke("What is the main topic?")
    print(f"Response 1: {response1}")
    response2 = rag.invoke("Can you provide more details?")
    print(f"Response 2: {response2}")
    # Clear memory if needed
    rag.clear_memory()
```

### Streaming Responses

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
    print("Response: ", end="", flush=True)
    for chunk in rag.stream("Explain the framework in detail"):
        print(chunk, end="", flush=True)
    print()
```

### Using Different Providers

```python
# Google Gemini
google_config = RAGConfig(
    name="gemini_rag",
    llm_provider="google",
    llm_model="gemini-pro",
    temperature=0.2
)

# Hugging Face
hf_config = RAGConfig(
    name="hf_rag",
    llm_provider="huggingface",
    llm_model="meta-llama/Llama-2-7b-chat-hf",
    llm_task="text-generation"
)

# Groq
groq_config = RAGConfig(
    name="groq_rag",
    llm_provider="groq",
    llm_model="llama3-8b-8192"
)

# Ollama (Local)
ollama_config = RAGConfig(
    name="ollama_rag",
    llm_provider="ollama",
    llm_model="llama2",
    llm_base_url="http://localhost:11434"
)
```

## Factory Function

Use the factory function for easy RAG creation:

```python
from llmblocks.blocks.rag import create_rag, RAGConfig

config = RAGConfig(
    name="factory_rag",
    llm_provider="openai",
    llm_model="gpt-4"
)

# Create different RAG types
basic_rag = create_rag("basic", config)
streaming_rag = create_rag("streaming", config)
memory_rag = create_rag("memory", config)
```

## Best Practices

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

### 3. Memory Management

- Use memory for conversational applications
- Clear memory periodically to prevent context overflow
- Consider memory persistence for long-running applications

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

### 5. Performance Optimization

- Use appropriate `top_k` values (2-8 for most cases)
- Enable streaming for better user experience
- Persist vector store for large document collections
- Use similarity thresholds to filter low-quality matches

## Troubleshooting

### Common Issues

1. **Import Errors**: Install required provider packages
   ```bash
   pip install -r requirements_providers.txt
   ```

2. **API Key Issues**: Set environment variables
   ```bash
   export OPENAI_API_KEY="your-key"
   export GOOGLE_API_KEY="your-key"
   # etc.
   ```

3. **Embedding Errors**: Implement embedding setup
   ```python
   # This needs to be implemented based on your embedding provider
   def _get_embeddings(self):
       # Return your embedding model
       pass
   ```

4. **Memory Issues**: Clear conversation history
   ```python
   rag.clear_memory()
   ```

5. **Ollama Connection**: Ensure Ollama is running
   ```bash
   ollama serve
   ollama pull llama2
   ```

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Your RAG code here
```

## API Reference

For detailed API documentation, see [RAG API Reference](../api/rag.md).

## Examples

For more examples, see:
- [Basic RAG Example](../../examples/basic_rag.py)
- [Multi-Provider Demo](../../examples/multi_provider_demo.py)
- [Playground Demo](../../examples/playground_demo.py)
