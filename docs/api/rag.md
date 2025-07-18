# RAG API Reference

## Classes

### RAGConfig

Configuration class for RAG pipelines.

```python
@dataclass
class RAGConfig(ComponentConfig):
    """Configuration for RAG pipelines."""
    
    # Document processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    text_splitter_type: str = "recursive"
    
    # Vector store
    vector_store_type: str = "chroma"
    embedding_model: str = "text-embedding-ada-002"
    persist_directory: Optional[str] = None
    
    # LLM Provider Configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_task: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1000
    
    # Retrieval
    top_k: int = 4
    similarity_threshold: float = 0.7
    
    # Memory
    memory_enabled: bool = False
    memory_type: str = "in_memory"
    
    # Streaming
    streaming_enabled: bool = False
```

#### Parameters

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
