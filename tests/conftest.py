"""
Pytest configuration and shared fixtures for LLMBlocks tests.
"""

import os
import sys
import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock

# Add src to Python path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test-api-key-12345"


@pytest.fixture
def sample_llm_config():
    """Provide sample LLM configuration for testing."""
    return {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 100,
        "timeout": 30.0
    }


@pytest.fixture
def sample_messages():
    """Provide sample messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "What's the weather like?"}
    ]


@pytest.fixture
def mock_llm_response():
    """Provide a mock LLM response for testing."""
    from llmblocks.blocks.llm_provider.base import LLMResponse
    return LLMResponse(
        content="This is a test response from the AI assistant.",
        metadata={
            "model": "test-model",
            "tokens_used": 25,
            "finish_reason": "stop"
        }
    )


@pytest.fixture
def mock_streaming_response():
    """Provide a mock streaming response for testing."""
    from llmblocks.blocks.llm_provider.base import LLMResponse
    
    async def mock_stream():
        chunks = ["Hello", " there", "! How", " can I", " help you", " today?"]
        for chunk in chunks:
            yield LLMResponse(
                content=chunk,
                metadata={"chunk_index": chunks.index(chunk)}
            )
    
    return mock_stream()


@pytest.fixture
def skip_if_no_api_key():
    """Skip test if required API keys are not available."""
    def _skip_if_no_key(provider: str):
        key_map = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GOOGLE_API_KEY", 
            "anthropic": "ANTHROPIC_API_KEY"
        }
        
        env_key = key_map.get(provider)
        if not env_key or not os.getenv(env_key):
            pytest.skip(f"Skipping {provider} test - {env_key} not set")
    
    return _skip_if_no_key


@pytest.fixture
def temp_env_var():
    """Temporarily set environment variables for testing."""
    original_values = {}
    
    def _set_env(key: str, value: str):
        original_values[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield _set_env
    
    # Cleanup
    for key, original_value in original_values.items():
        if original_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original_value


@pytest.fixture
def mock_langchain_client():
    """Provide a mock LangChain client for testing."""
    mock_client = AsyncMock()
    
    # Mock the agenerate method
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    
    mock_message = AIMessage(content="Test response from mock client")
    mock_generation = ChatGeneration(message=mock_message)
    mock_result = ChatResult(generations=[mock_generation])
    
    mock_client.agenerate.return_value = mock_result
    
    # Mock the astream method
    async def mock_stream(*args, **kwargs):
        chunks = ["Hello", " world", "!"]
        for chunk in chunks:
            yield AIMessage(content=chunk)
    
    mock_client.astream.return_value = mock_stream()
    
    return mock_client


@pytest.fixture
def disable_logging():
    """Disable logging during tests for cleaner output."""
    import logging
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# Test markers for different test types
pytestmark = [
    pytest.mark.asyncio,  # All tests are async by default
]


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Ensure we're using the src layout
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interaction"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests for complete workflows"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests that may take longer to run"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add e2e marker to tests in e2e/ directory
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
