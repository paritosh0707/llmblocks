"""
Tests for RAG blocks with LangChain Runnable compatibility.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from llmblocks.blocks.rag import (
    BasicRAG, 
    StreamingRAG, 
    MemoryRAG, 
    RAGConfig, 
    BaseRAG,
    create_rag
)
from langchain_core.documents import Document


class TestRAGConfig:
    """Test RAG configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RAGConfig(name="test")

        assert config.name == "test"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.llm_model == "gpt-3.5-turbo"
        assert config.temperature == 0.0
        assert config.top_k == 4
        assert not config.memory_enabled
        assert not config.streaming_enabled

    def test_custom_config(self):
        """Test custom configuration values."""
        config = RAGConfig(
            name="custom",
            chunk_size=500,
            chunk_overlap=100,
            llm_model="gpt-4",
            temperature=0.5,
            top_k=8,
            memory_enabled=True,
            streaming_enabled=True
        )

        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.llm_model == "gpt-4"
        assert config.temperature == 0.5
        assert config.top_k == 8
        assert config.memory_enabled
        assert config.streaming_enabled


class TestBaseRAG:
    """Test base RAG functionality."""

    def test_initialization(self):
        """Test RAG initialization."""
        config = RAGConfig(name="test")
        rag = BaseRAG(config)

        assert rag.config == config
        assert rag.retriever is None
        assert rag.llm is None
        assert rag.vector_store is None
        assert rag.text_splitter is None

    def test_context_manager(self):
        """Test context manager functionality."""
        config = RAGConfig(name="test")
        rag = BaseRAG(config)

        with patch.object(rag, 'initialize') as mock_init:
            with patch.object(rag, 'cleanup') as mock_cleanup:
                with rag:
                    mock_init.assert_called_once()
                mock_cleanup.assert_called_once()

    def test_add_texts(self):
        """Test adding texts to RAG."""
        config = RAGConfig(name="test")
        rag = BaseRAG(config)

        texts = ["Text 1", "Text 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]

        with patch.object(rag, 'add_documents') as mock_add_docs:
            rag.add_texts(texts, metadatas)

            mock_add_docs.assert_called_once()
            docs = mock_add_docs.call_args[0][0]
            assert len(docs) == 2
            assert docs[0].page_content == "Text 1"
            assert docs[0].metadata == {"source": "test1"}
            assert docs[1].page_content == "Text 2"
            assert docs[1].metadata == {"source": "test2"}


class TestBasicRAG:
    """Test BasicRAG functionality."""

    def test_initialization(self):
        """Test BasicRAG initialization."""
        config = RAGConfig(name="test")
        rag = BasicRAG(config)

        assert rag.prompt_template is not None
        assert "context" in rag.prompt_template.input_variables
        assert "question" in rag.prompt_template.input_variables

    def test_invoke_without_documents(self):
        """Test invoking without documents raises error."""
        config = RAGConfig(name="test")
        rag = BasicRAG(config)

        with pytest.raises(ValueError, match="No documents have been added"):
            rag.invoke("test question")

    def test_query_without_documents(self):
        """Test querying without documents raises error (backward compatibility)."""
        config = RAGConfig(name="test")
        rag = BasicRAG(config)

        with pytest.raises(ValueError, match="No documents have been added"):
            rag.query("test question")

    @patch('llmblocks.blocks.rag.ChatOpenAI')
    def test_invoke_with_documents(self, mock_chat_openai):
        """Test invoking with documents using new invoke() method."""
        # Setup mock
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test answer"
        mock_llm.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        config = RAGConfig(name="test")
        rag = BasicRAG(config)

        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        rag.retriever = mock_retriever

        # Mock LLM
        rag.llm = mock_llm

        # Test invoke
        result = rag.invoke("test question")

        assert result == "Test answer"
        mock_retriever.get_relevant_documents.assert_called_once_with("test question")
        mock_llm.assert_called_once()

    @patch('llmblocks.blocks.rag.ChatOpenAI')
    def test_query_with_documents(self, mock_chat_openai):
        """Test querying with documents using backward compatible query() method."""
        # Setup mock
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test answer"
        mock_llm.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        config = RAGConfig(name="test")
        rag = BasicRAG(config)

        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        rag.retriever = mock_retriever

        # Mock LLM
        rag.llm = mock_llm

        # Test query (backward compatibility)
        result = rag.query("test question")

        assert result == "Test answer"
        mock_retriever.get_relevant_documents.assert_called_once_with("test question")
        mock_llm.assert_called_once()

    def test_invoke_with_dict_input(self):
        """Test invoke with dictionary input."""
        config = RAGConfig(name="test")
        rag = BasicRAG(config)

        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        rag.retriever = mock_retriever

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test answer"
        mock_llm.return_value = mock_response
        rag.llm = mock_llm

        # Test with dict input
        result = rag.invoke({"question": "test question"})

        assert result == "Test answer"
        mock_retriever.get_relevant_documents.assert_called_once_with("test question")

    def test_invoke_with_invalid_input(self):
        """Test invoke with invalid input type."""
        config = RAGConfig(name="test")
        rag = BasicRAG(config)

        with pytest.raises(ValueError, match="Unsupported input type"):
            rag.invoke(123)  # Invalid input type

    def test_batch_processing(self):
        """Test batch processing functionality."""
        config = RAGConfig(name="test")
        rag = BasicRAG(config)

        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        rag.retriever = mock_retriever

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test answer"
        mock_llm.return_value = mock_response
        rag.llm = mock_llm

        # Test batch processing
        questions = ["Question 1", "Question 2", "Question 3"]
        results = rag.batch(questions)

        assert len(results) == 3
        assert all(result == "Test answer" for result in results)
        assert mock_retriever.get_relevant_documents.call_count == 3


class TestStreamingRAG:
    """Test StreamingRAG functionality."""

    def test_initialization_enables_streaming(self):
        """Test that StreamingRAG enables streaming by default."""
        config = RAGConfig(name="test")
        rag = StreamingRAG(config)

        assert rag.config.streaming_enabled

    def test_initialization_with_kwargs(self):
        """Test StreamingRAG initialization with kwargs."""
        rag = StreamingRAG(chunk_size=500)

        assert rag.config.streaming_enabled
        assert rag.config.chunk_size == 500

    @patch('llmblocks.blocks.rag.ChatOpenAI')
    def test_stream_method(self, mock_chat_openai):
        """Test new stream() method."""
        # Setup mock
        mock_llm = Mock()
        mock_chunk1 = Mock()
        mock_chunk1.content = "Hello"
        mock_chunk2 = Mock()
        mock_chunk2.content = " World"
        mock_llm.stream.return_value = [mock_chunk1, mock_chunk2]
        mock_chat_openai.return_value = mock_llm

        config = RAGConfig(name="test")
        rag = StreamingRAG(config)

        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        rag.retriever = mock_retriever

        # Mock LLM
        rag.llm = mock_llm

        # Test new stream() method
        chunks = list(rag.stream("test question"))

        assert chunks == ["Hello", " World"]
        mock_retriever.get_relevant_documents.assert_called_once_with("test question")

    @patch('llmblocks.blocks.rag.ChatOpenAI')
    def test_query_stream_method(self, mock_chat_openai):
        """Test backward compatible query_stream() method."""
        # Setup mock
        mock_llm = Mock()
        mock_chunk1 = Mock()
        mock_chunk1.content = "Hello"
        mock_chunk2 = Mock()
        mock_chunk2.content = " World"
        mock_llm.stream.return_value = [mock_chunk1, mock_chunk2]
        mock_chat_openai.return_value = mock_llm

        config = RAGConfig(name="test")
        rag = StreamingRAG(config)

        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        rag.retriever = mock_retriever

        # Mock LLM
        rag.llm = mock_llm

        # Test backward compatible query_stream() method
        chunks = list(rag.query_stream("test question"))

        assert chunks == ["Hello", " World"]
        mock_retriever.get_relevant_documents.assert_called_once_with("test question")

    def test_stream_without_documents(self):
        """Test streaming without documents raises error."""
        config = RAGConfig(name="test")
        rag = StreamingRAG(config)

        with pytest.raises(ValueError, match="No documents have been added"):
            list(rag.stream("test question"))

    def test_query_stream_without_documents(self):
        """Test query streaming without documents raises error (backward compatibility)."""
        config = RAGConfig(name="test")
        rag = StreamingRAG(config)

        with pytest.raises(ValueError, match="No documents have been added"):
            list(rag.query_stream("test question"))

    def test_invoke_method(self):
        """Test invoke() method returns concatenated stream."""
        config = RAGConfig(name="test")
        rag = StreamingRAG(config)

        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        rag.retriever = mock_retriever

        # Mock LLM
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test answer"
        mock_llm.return_value = mock_response
        rag.llm = mock_llm

        # Test invoke() method
        result = rag.invoke("test question")

        assert result == "Test answer"


class TestMemoryRAG:
    """Test MemoryRAG functionality."""

    def test_initialization_enables_memory(self):
        """Test that MemoryRAG enables memory by default."""
        config = RAGConfig(name="test")
        rag = MemoryRAG(config)

        assert rag.config.memory_enabled
        assert len(rag.conversation_history) == 0

    def test_initialization_with_kwargs(self):
        """Test MemoryRAG initialization with kwargs."""
        rag = MemoryRAG(chunk_size=500)

        assert rag.config.memory_enabled
        assert rag.config.chunk_size == 500

    def test_add_to_memory(self):
        """Test adding to memory."""
        config = RAGConfig(name="test")
        rag = MemoryRAG(config)

        rag.add_to_memory("What is X?", "X is a thing")

        assert len(rag.conversation_history) == 1
        assert rag.conversation_history[0]["question"] == "What is X?"
        assert rag.conversation_history[0]["answer"] == "X is a thing"

    def test_get_relevant_history_empty(self):
        """Test getting relevant history when empty."""
        config = RAGConfig(name="test")
        rag = MemoryRAG(config)

        history = rag.get_relevant_history("test question")
        assert history == ""

    def test_get_relevant_history_with_entries(self):
        """Test getting relevant history with entries."""
        config = RAGConfig(name="test")
        rag = MemoryRAG(config)

        # Add some conversation history
        rag.add_to_memory("What is A?", "A is the first letter")
        rag.add_to_memory("What is B?", "B is the second letter")
        rag.add_to_memory("What is C?", "C is the third letter")

        history = rag.get_relevant_history("test question", top_k=2)

        # Should contain the last 2 conversations
        assert "What is B?" in history
        assert "B is the second letter" in history
        assert "What is C?" in history
        assert "C is the third letter" in history
        # Should not contain the first conversation
        assert "What is A?" not in history

    def test_clear_memory(self):
        """Test clearing memory."""
        config = RAGConfig(name="test")
        rag = MemoryRAG(config)

        # Add some memory
        rag.add_to_memory("What is X?", "X is a thing")
        rag.add_to_memory("What is Y?", "Y is another thing")

        assert len(rag.conversation_history) == 2

        # Clear memory
        rag.clear_memory()

        assert len(rag.conversation_history) == 0

    @patch('llmblocks.blocks.rag.ChatOpenAI')
    def test_invoke_with_memory(self, mock_chat_openai):
        """Test invoke() method with memory functionality."""
        # Setup mock
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test answer"
        mock_llm.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        config = RAGConfig(name="test")
        rag = MemoryRAG(config)

        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_retriever.get_relevant_documents.return_value = [mock_doc]
        rag.retriever = mock_retriever

        # Mock LLM
        rag.llm = mock_llm

        # Test invoke with memory
        result = rag.invoke("test question")

        assert result == "Test answer"
        assert len(rag.conversation_history) == 1
        assert rag.conversation_history[0]["question"] == "test question"
        assert rag.conversation_history[0]["answer"] == "Test answer"


class TestCreateRAG:
    """Test RAG factory function."""

    def test_create_basic_rag(self):
        """Test creating basic RAG."""
        rag = create_rag("basic")
        assert isinstance(rag, BasicRAG)

    def test_create_streaming_rag(self):
        """Test creating streaming RAG."""
        rag = create_rag("streaming")
        assert isinstance(rag, StreamingRAG)
        assert rag.config.streaming_enabled

    def test_create_memory_rag(self):
        """Test creating memory RAG."""
        rag = create_rag("memory")
        assert isinstance(rag, MemoryRAG)
        assert rag.config.memory_enabled

    def test_create_unknown_rag_type(self):
        """Test creating unknown RAG type raises error."""
        with pytest.raises(ValueError, match="Unknown RAG type"):
            create_rag("unknown")

    def test_create_rag_with_config(self):
        """Test creating RAG with config."""
        config = RAGConfig(name="test", chunk_size=500)
        rag = create_rag("basic", config)

        assert isinstance(rag, BasicRAG)
        assert rag.config.name == "test"
        assert rag.config.chunk_size == 500


class TestRAGIntegration:
    """Test RAG integration scenarios."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                page_content="LLMBlocks is a modular framework for building LLM applications.",
                metadata={"source": "test"}
            ),
            Document(
                page_content="The framework supports multiple LLM providers and RAG pipelines.",
                metadata={"source": "test"}
            )
        ]

    @patch('llmblocks.blocks.rag.ChatOpenAI')
    def test_basic_rag_integration(self, mock_chat_openai, sample_documents):
        """Test basic RAG integration with both invoke() and query() methods."""
        # Setup mock
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "LLMBlocks is a modular framework"
        mock_llm.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        config = RAGConfig(name="test")
        rag = BasicRAG(config)

        with rag:
            rag.add_documents(sample_documents)

            # Test invoke() method
            result1 = rag.invoke("What is LLMBlocks?")
            assert "LLMBlocks" in result1

            # Test query() method (backward compatibility)
            result2 = rag.query("What is LLMBlocks?")
            assert "LLMBlocks" in result2

            # Both should return the same result
            assert result1 == result2

    @patch('llmblocks.blocks.rag.ChatOpenAI')
    def test_memory_rag_conversation(self, mock_chat_openai, sample_documents):
        """Test memory RAG conversation flow."""
        # Setup mock
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "Test answer"
        mock_llm.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        config = RAGConfig(name="test")
        rag = MemoryRAG(config)

        with rag:
            rag.add_documents(sample_documents)

            # First question
            result1 = rag.invoke("What is LLMBlocks?")
            assert len(rag.conversation_history) == 1

            # Second question (should use memory)
            result2 = rag.invoke("Can you tell me more?")
            assert len(rag.conversation_history) == 2

            # Clear memory
            rag.clear_memory()
            assert len(rag.conversation_history) == 0 