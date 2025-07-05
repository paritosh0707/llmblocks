"""
RAG (Retrieval-Augmented Generation) Pipelines for LLMBlocks.

This module provides pre-built RAG pipelines that can be easily composed
and extended for different use cases.
"""

from typing import List, Dict, Any, Optional, Iterator, Union, Generator
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
from abc import abstractmethod

from langchain.schema import Document, BaseRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import Chroma
from langchain.llms.base import LLM
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate

from ..core.base_component import BaseComponent, ComponentConfig, BaseBlock
from ..core.llm_providers import LLMProviderFactory


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
    llm_provider: str = "openai"  # openai, google, huggingface, groq, anthropic, ollama
    llm_model: str = "gpt-3.5-turbo"
    llm_api_key: Optional[str] = None  # Can be set directly or via environment variables
    llm_base_url: Optional[str] = None  # For local providers like Ollama
    llm_task: Optional[str] = None  # For Hugging Face models
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


class BaseRAG(BaseBlock):
    """Base class for all RAG implementations that extends BaseBlock for Runnable compatibility."""
    
    def __init__(self, config: Optional[RAGConfig] = None, **kwargs):
        super().__init__(config or RAGConfig(**kwargs))
        self.retriever: Optional[BaseRetriever] = None
        self.llm: Optional[LLM] = None
        self.vector_store: Optional[VectorStore] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        
    def initialize(self) -> None:
        """Initialize the RAG pipeline components."""
        self.logger.info("Initializing RAG pipeline...")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Initialize LLM
        self._setup_llm()
        
        self.logger.info("RAG pipeline initialized successfully")
    
    def _setup_llm(self) -> None:
        """Setup the language model using the provider factory."""
        try:
            # Prepare configuration for the provider
            llm_config = {
                'model': self.config.llm_model,
                'temperature': self.config.temperature,
                'max_tokens': self.config.max_tokens,
                'streaming': self.config.streaming_enabled,
                'api_key': self.config.llm_api_key,
            }
            
            # Add provider-specific configurations
            if self.config.llm_base_url:
                llm_config['base_url'] = self.config.llm_base_url
            
            if self.config.llm_task:
                llm_config['task'] = self.config.llm_task
            
            # Create LLM using the provider factory
            self.llm = LLMProviderFactory.create_llm(
                self.config.llm_provider, 
                llm_config
            )
            
            self.logger.info(f"Initialized {self.config.llm_provider} LLM with model: {self.config.llm_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM provider '{self.config.llm_provider}': {e}")
            raise
    
    def _setup_vector_store(self, documents: List[Document]) -> None:
        """Setup vector store with documents."""
        if not documents:
            self.logger.warning("No documents provided for vector store setup")
            return
            
        # For now, we'll use Chroma as the default vector store
        # In a production system, you'd want to support multiple backends
        embeddings = self._get_embeddings()
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=self.config.persist_directory
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.config.top_k}
        )
    
    def _get_embeddings(self) -> Embeddings:
        """Get embeddings model."""
        # This is a placeholder - you'd implement actual embedding logic here
        # For now, we'll raise an error to indicate this needs to be implemented
        raise NotImplementedError("Embedding setup needs to be implemented")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        if not self.vector_store:
            self._setup_vector_store(documents)
        else:
            # Add to existing vector store
            self.vector_store.add_documents(documents)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add raw texts to the vector store."""
        documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(texts, metadatas or [{}] * len(texts))
        ]
        self.add_documents(documents)
    
    @abstractmethod
    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute the RAG pipeline with a single input.
        
        Args:
            input: The input to process. Can be a string (question) or dict with 'question' key.
            config: Optional configuration overrides.
            
        Returns:
            The generated response as a string.
        """
        pass
    
    def query(self, question: str, **kwargs) -> str:
        """
        Query the RAG pipeline (backward compatibility method).
        
        Args:
            question: The question to ask.
            **kwargs: Additional arguments.
            
        Returns:
            The generated response.
        """
        return self.invoke(question, kwargs)
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self.vector_store and hasattr(self.vector_store, 'persist'):
            self.vector_store.persist()
        self.logger.info("RAG pipeline cleaned up")


class BasicRAG(BaseRAG):
    """Basic RAG pipeline with simple retrieval and generation."""
    
    def __init__(self, config: Optional[RAGConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following context to answer the question. 
            If you cannot answer the question based on the context, say so.

            Context: {context}
            Question: {question}
            
            Answer:"""
        )
    
    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute the basic RAG pipeline.
        
        Args:
            input: The input to process. Can be a string (question) or dict with 'question' key.
            config: Optional configuration overrides.
            
        Returns:
            The generated response as a string.
        """
        # Handle different input types
        if isinstance(input, dict):
            question = input.get('question', '')
        elif isinstance(input, str):
            question = input
        else:
            raise ValueError(f"Unsupported input type: {type(input)}. Expected string or dict with 'question' key.")
        
        if not self.retriever:
            raise ValueError("No documents have been added to the RAG pipeline")
        
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response
        prompt = self.prompt_template.format(context=context, question=question)
        
        if isinstance(self.llm, ChatOpenAI):
            messages = [
                SystemMessage(content="You are a helpful assistant that answers questions based on the provided context."),
                HumanMessage(content=prompt)
            ]
            response = self.llm(messages)
            return response.content
        else:
            return self.llm(prompt)


class StreamingRAG(BasicRAG):
    """RAG pipeline with streaming response generation."""
    
    def __init__(self, config: Optional[RAGConfig] = None, **kwargs):
        if config:
            config.streaming_enabled = True
        else:
            kwargs['streaming_enabled'] = True
        super().__init__(config, **kwargs)
    
    def stream(self, input: Any, config: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """
        Execute the streaming RAG pipeline.
        
        Args:
            input: The input to process. Can be a string (question) or dict with 'question' key.
            config: Optional configuration overrides.
            
        Yields:
            Streamed response chunks.
        """
        # Handle different input types
        if isinstance(input, dict):
            question = input.get('question', '')
        elif isinstance(input, str):
            question = input
        else:
            raise ValueError(f"Unsupported input type: {type(input)}. Expected string or dict with 'question' key.")
        
        if not self.retriever:
            raise ValueError("No documents have been added to the RAG pipeline")
        
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate streaming response
        prompt = self.prompt_template.format(context=context, question=question)
        
        if isinstance(self.llm, ChatOpenAI):
            messages = [
                SystemMessage(content="You are a helpful assistant that answers questions based on the provided context."),
                HumanMessage(content=prompt)
            ]
            
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
        else:
            # Fallback for non-streaming LLMs
            response = self.llm(prompt)
            yield response
    
    def query_stream(self, question: str, **kwargs) -> Iterator[str]:
        """
        Query the streaming RAG pipeline (backward compatibility method).
        
        Args:
            question: The question to ask.
            **kwargs: Additional arguments.
            
        Yields:
            Streamed response chunks.
        """
        yield from self.stream(question, kwargs)
    
    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute the streaming RAG pipeline (non-streaming interface).
        
        Args:
            input: The input to process. Can be a string (question) or dict with 'question' key.
            config: Optional configuration overrides.
            
        Returns:
            The complete response as a string.
        """
        return "".join(self.stream(input, config))


class MemoryRAG(BasicRAG):
    """RAG pipeline with conversation memory."""
    
    def __init__(self, config: Optional[RAGConfig] = None, **kwargs):
        if config:
            config.memory_enabled = True
        else:
            kwargs['memory_enabled'] = True
        super().__init__(config, **kwargs)
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_to_memory(self, question: str, answer: str) -> None:
        """Add a Q&A pair to conversation memory."""
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "timestamp": asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else None
        })
    
    def get_relevant_history(self, question: str, top_k: int = 3) -> str:
        """Get relevant conversation history for the current question."""
        # Simple implementation - in production you'd want semantic search
        if not self.conversation_history:
            return ""
        
        # For now, just return the last few conversations
        recent_history = self.conversation_history[-top_k:]
        history_text = "\n\n".join([
            f"Q: {item['question']}\nA: {item['answer']}"
            for item in recent_history
        ])
        return history_text
    
    def invoke(self, input: Any, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute the memory-enabled RAG pipeline.
        
        Args:
            input: The input to process. Can be a string (question) or dict with 'question' key.
            config: Optional configuration overrides.
            
        Returns:
            The generated response as a string.
        """
        # Handle different input types
        if isinstance(input, dict):
            question = input.get('question', '')
        elif isinstance(input, str):
            question = input
        else:
            raise ValueError(f"Unsupported input type: {type(input)}. Expected string or dict with 'question' key.")
        
        if not self.retriever:
            raise ValueError("No documents have been added to the RAG pipeline")
        
        # Retrieve relevant documents
        docs = self.retriever.get_relevant_documents(question)
        
        # Combine context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Get relevant conversation history
        history = self.get_relevant_history(question)
        
        # Create enhanced prompt with history
        if history:
            prompt = f"""Previous conversation:
{history}

Current context:
{context}

Question: {question}

Answer:"""
        else:
            prompt = f"""Context: {context}
Question: {question}
Answer:"""
        
        # Generate response
        if isinstance(self.llm, ChatOpenAI):
            messages = [
                SystemMessage(content="You are a helpful assistant that answers questions based on the provided context and conversation history."),
                HumanMessage(content=prompt)
            ]
            response = self.llm(messages)
            answer = response.content
        else:
            answer = self.llm(prompt)
        
        # Add to memory
        self.add_to_memory(question, answer)
        
        return answer
    
    def clear_memory(self) -> None:
        """Clear conversation memory."""
        self.conversation_history.clear()
        self.logger.info("Conversation memory cleared")


def create_rag(rag_type: str = "basic", config: Optional[RAGConfig] = None, **kwargs) -> BaseRAG:
    """
    Create a RAG pipeline of the specified type.
    
    Args:
        rag_type: Type of RAG pipeline ("basic", "streaming", "memory").
        config: Optional RAG configuration.
        **kwargs: Additional configuration parameters.
        
    Returns:
        A RAG pipeline instance.
        
    Raises:
        ValueError: If rag_type is not supported.
    """
    rag_classes = {
        "basic": BasicRAG,
        "streaming": StreamingRAG,
        "memory": MemoryRAG
    }
    
    if rag_type not in rag_classes:
        raise ValueError(f"Unknown RAG type: {rag_type}. Available types: {list(rag_classes.keys())}")
    
    return rag_classes[rag_type](config, **kwargs)