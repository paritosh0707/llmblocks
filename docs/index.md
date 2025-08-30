# 📚 LLMBlocks Documentation

Welcome to the comprehensive documentation for LLMBlocks - a modular, enterprise-grade AI application framework.

## 🎯 **Overview**

LLMBlocks is designed as a "block-based" architecture where you can mix and match different AI components to build sophisticated applications. Think of it as "Lego blocks for AI applications."

## 🏗️ **Architecture**

### **Core Principles**
1. **Modularity** - Each component is a self-contained block
2. **Composability** - Blocks can be combined in flexible ways
3. **Async-First** - Built for high-performance concurrent applications
4. **Configuration-Driven** - Everything configurable via YAML/JSON
5. **Production-Ready** - Enterprise-grade security and monitoring

### **Current Architecture (Phase 1 & 2 Complete)**

```
src/llmblocks/
├── core/                   # ✅ Foundation layer
│   ├── base_block.py      # Base class for all blocks
│   ├── registry.py        # Block discovery and management
│   ├── config.py          # Configuration management
│   ├── utils.py           # Utility functions (20+ helpers)
│   └── tracing.py         # Observability and monitoring
├── blocks/                # ✅ Core blocks
│   └── llm_provider/      # ✅ LLM provider system
│       ├── base.py        # BaseLLMProvider with LangChain integration
│       ├── factory.py     # Provider factory and registry
│       ├── gemini_provider.py    # Google Gemini (Production Ready)
│       ├── openai_provider.py    # OpenAI GPT (Production Ready)
│       └── anthropic_provider.py # Anthropic Claude (Production Ready)
├── utils/                 # ✅ Utilities
│   ├── exceptions.py      # Custom exception hierarchy
│   ├── logging.py         # Structured logging with structlog
│   └── langgraph_compat.py # LangGraph compatibility layer
└── components/            # 🚧 High-level components (Future phases)
    ├── chatbot/          # 🚧 Phase 4
    ├── rag/              # 🚧 Phase 4
    └── agent/            # 🚧 Phase 4
```

## 🔥 **LLM Provider System** (Production Ready)

The LLM Provider system is the cornerstone of LLMBlocks, providing a unified interface to multiple LLM providers.

### **Key Features**
- **Multi-Provider Support** - OpenAI, Gemini, Anthropic, and more
- **LangChain Integration** - Full compatibility with LangChain ecosystem
- **LangGraph Ready** - Create nodes and graphs easily
- **Async/Sync Support** - Both synchronous and asynchronous operations
- **Streaming Support** - Real-time response streaming
- **Error Handling** - Comprehensive retry logic and error management
- **Observability** - Built-in tracing and monitoring

### **Usage Examples**

#### **Basic Usage**
```python
from src.llmblocks.blocks.llm_provider import get_provider

# Create a provider
provider = get_provider(
    provider_name="gemini",
    api_key="your-api-key",
    model="gemini-2.0-flash-exp"
)

# Generate response
response = await provider.generate("Hello, world!")
print(response.content)
```

#### **LangChain Integration**
```python
# Use as LangChain LLM
langchain_llm = provider.as_langchain()
result = langchain_llm.invoke("Tell me about AI")

# Use in LangChain chains
from langchain.schema import HumanMessage
messages = [HumanMessage(content="Hello")]
result = langchain_llm.invoke(messages)
```

#### **LangGraph Integration**
```python
# Create LangGraph node
llm_node = provider.create_langgraph_node("assistant")

# Create complete graph
graph = provider.create_langgraph_graph("llm_graph")
```

#### **Streaming**
```python
async for chunk in provider.generate_stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

## 🔧 **Configuration System**

LLMBlocks uses a powerful configuration system that supports:
- **Environment Variables** - Secure API key management
- **YAML/JSON Files** - Human-readable configuration
- **Pydantic Validation** - Type-safe configuration with validation
- **Hierarchical Configs** - Override and merge configurations

### **Example Configuration**
```python
from src.llmblocks.blocks.llm_provider import GeminiProviderConfig

config = GeminiProviderConfig(
    api_key="your-api-key",
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    max_tokens=1000,
    timeout=30.0
)
```

## 📊 **Observability & Monitoring**

Built-in observability features:

### **Structured Logging**
```python
from src.llmblocks.utils.logging import get_logger

logger = get_logger("MyComponent")
logger.info("Processing request", user_id="123", request_type="chat")
```

### **Tracing**
```python
from src.llmblocks.core.tracing import trace_span, trace_llm_call

# Trace code blocks
with trace_span("llm_generation"):
    response = await provider.generate("Hello")

# Trace LLM calls automatically
trace_llm_call("gemini", "gemini-2.0-flash", prompt, response.content)
```

## 🧪 **Testing**

Comprehensive test suite with 7 test files:

- `test_gemini_simple.py` - Basic Gemini provider testing
- `test_latest_compatibility.py` - LangChain/LangGraph compatibility
- `test_langgraph_node.py` - LangGraph node creation
- `test_latest_langgraph.py` - Advanced LangGraph features
- `test_llm_providers.py` - Multi-provider testing
- `langchain_compatibility_demo.py` - Integration examples

### **Running Tests**
```bash
# Run specific test
python examples/test_gemini_simple.py

# Run all tests
python -m pytest tests/
```

## 🚀 **Development Roadmap**

### **✅ Phase 1: Foundation & Core Architecture** (Complete)
- Base block system
- Registry and configuration
- Logging and error handling
- Utility functions

### **✅ Phase 2: Enhanced LLM Provider System** (Complete)
- LangChain/LangGraph integration
- Multi-provider support (OpenAI, Gemini, Anthropic)
- Async/sync operations
- Streaming support
- Comprehensive testing

### **🚧 Phase 3: Memory & State Management** (Next)
- In-memory storage
- Redis integration
- PostgreSQL support
- Vector memory for embeddings
- Session management

### **🚧 Phase 4: Advanced Components** (Planned)
- RAG system with document loaders
- Multi-tool agent framework
- Conversation management
- Workflow orchestration

### **🚧 Phase 5: Developer Experience** (Planned)
- CLI framework for project scaffolding
- Web-based playground
- Deployment tools
- Performance optimization

## 🔗 **API Reference**

### **Core Classes**

#### **BaseBlock**
```python
from src.llmblocks.core import BaseBlock, BlockStatus

class MyBlock(BaseBlock):
    async def initialize(self) -> None:
        # Initialize your block
        pass
```

#### **BaseLLMProvider**
```python
from src.llmblocks.blocks.llm_provider import BaseLLMProvider

class CustomProvider(BaseLLMProvider):
    async def _generate_impl(self, messages, **kwargs):
        # Implement your provider
        pass
```

### **Utility Functions**
```python
from src.llmblocks.core.utils import (
    generate_id,
    get_timestamp,
    load_yaml_file,
    retry_async,
    measure_time
)
```

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Follow the coding standards** (black, isort, mypy)
4. **Add tests** for your changes
5. **Update documentation** as needed
6. **Submit a pull request**

### **Development Setup**
```bash
git clone https://github.com/paritosh0707/llmblocks.git
cd llmblocks
uv sync
pre-commit install
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🆘 **Support**

- **Documentation**: This documentation
- **Issues**: [GitHub Issues](https://github.com/paritosh0707/llmblocks/issues)
- **Discussions**: [GitHub Discussions](https://github.com/paritosh0707/llmblocks/discussions)

---

*LLMBlocks - Building the future of AI applications, one block at a time.* 🚀