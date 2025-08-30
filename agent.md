# 🤖 Agent Knowledge Base - LLMBlocks Project

This document contains comprehensive instructions, project knowledge, and important decisions for AI agents working on the LLMBlocks project.

## 📋 **Project Overview**

**LLMBlocks** is a modular, enterprise-grade AI application framework designed as "Lego blocks for AI applications." The project follows a phased development approach with a block-based architecture.

### **Current Status (January 2025)**
- **Phase 1 & 2**: ✅ **COMPLETE** - Foundation + LLM Provider System
- **Phase 3**: 🚧 **NEXT** - Memory & State Management  
- **Phase 4**: 🚧 **PLANNED** - RAG & Agent Systems
- **Phase 5**: 🚧 **PLANNED** - Developer Experience & Playground

## 🏗️ **Architecture Decisions**

### **1. Directory Structure - `src/` Layout**
**CRITICAL**: Always use the `src/llmblocks/` structure, NOT `llmblocks/`.

```
src/llmblocks/           # ✅ CORRECT - Modern Python packaging
├── core/               # Foundation layer
├── blocks/             # Core functional blocks
├── components/         # High-level components  
├── utils/              # Utilities and helpers
├── cli/                # Command-line interface
└── playground/         # Interactive development
```

**Why `src/` layout?**
- Prevents accidental imports from source during development
- Forces proper package installation
- Industry standard (pytest, requests, setuptools use this)
- Ensures tests use installed package, not source

### **2. Import Patterns**
```python
# ✅ CORRECT - Production imports (after pip install -e .)
from llmblocks.blocks.llm_provider import get_provider
from llmblocks.core import BaseBlock

# ✅ CORRECT - Development imports (direct source access)
from src.llmblocks.blocks.llm_provider import get_provider
from src.llmblocks.core import BaseBlock

# ❌ WRONG - Never use old structure
from llmblocks.blocks.llm_provider import get_provider  # Old structure
```

### **3. LangChain/LangGraph Integration Strategy**

**CRITICAL DECISION**: Use **composition over inheritance** to avoid Pydantic conflicts.

#### **The Problem**
- `BaseLLMProvider` needs to inherit from `langchain.BaseChatModel` (Pydantic model)
- `BaseBlock` is also a Pydantic model
- Multiple Pydantic inheritance causes MRO (Method Resolution Order) conflicts

#### **The Solution**
```python
class BaseLLMProvider(BaseChatModel):  # Inherit from LangChain
    def __init__(self, config):
        # Use composition for BaseBlock functionality
        self._block_id = generate_id("llm_provider")
        self._status = BlockStatus.UNINITIALIZED
        self._created_at = datetime.utcnow()
        # ... other BaseBlock attributes as private fields
    
    @property
    def block_id(self) -> str:
        return self._block_id  # Expose via properties
```

### **4. LangGraph MRO Compatibility**

**CRITICAL ISSUE**: LangGraph has MRO conflicts in certain versions.

#### **The Problem**
```python
# This fails in some LangGraph versions
from langgraph.graph import StateGraph, MessagesState
# TypeError: Cannot create a consistent method resolution order (MRO) for bases ABC, Generic
```

#### **The Solution**
Created `src/llmblocks/utils/langgraph_compat.py`:
```python
def create_compatible_graph(llm_node, node_name):
    """Creates StateGraph or fallback if MRO conflict."""
    try:
        from langgraph.graph import StateGraph, MessagesState
        # ... create StateGraph
        return True, app, None
    except (ImportError, TypeError) as e:
        # Fallback to simple callable
        return False, fallback_graph, str(e)
```

### **5. Metadata Compatibility**

**CRITICAL ISSUE**: LangChain's callback system expects `metadata` to be a mutable dict.

#### **The Problem**
```python
# This fails
@property
def metadata(self):
    return {"status": self.status}  # Returns new dict each time

# LangChain tries: self.metadata.update({...})  # Fails on property
```

#### **The Solution**
```python
def __init__(self, config):
    # Make metadata a regular dict attribute
    self.metadata = {
        "block_id": self._block_id,
        "provider": self.provider_name,
        "status": self.status.value,
    }

def update_metadata(self):
    """Keep metadata synchronized."""
    self.metadata.update({
        "status": self.status.value,
        "updated_at": self.updated_at.isoformat()
    })
```

## 🔧 **Development Guidelines**

### **1. Code Standards**
- **Python 3.11+** minimum
- **Type hints** everywhere (`from typing import ...`)
- **Async-first** design (use `async def` when possible)
- **Pydantic models** for configuration
- **Structured logging** with context
- **Comprehensive error handling**

### **2. Testing Strategy**
```python
# Test file naming
test_*.py           # Unit tests
examples/*.py       # Integration tests and demos

# Key test files
test_gemini_simple.py          # Basic provider functionality
test_latest_compatibility.py   # LangChain/LangGraph compatibility
test_langgraph_node.py        # LangGraph node creation
langchain_compatibility_demo.py # Integration examples
```

### **3. Error Handling Patterns**
```python
# Custom exception hierarchy
from src.llmblocks.utils.exceptions import (
    LLMBlocksError,           # Base exception
    LLMProviderError,         # Provider-specific errors
    LLMConnectionError,       # Connection issues
    LLMAuthenticationError,   # Auth failures
    LLMRateLimitError,       # Rate limiting
    LLMTimeoutError          # Timeout issues
)

# Usage pattern
try:
    response = await provider.generate(prompt)
except LLMAuthenticationError:
    logger.error("Invalid API key")
except LLMRateLimitError:
    logger.warning("Rate limit hit, retrying...")
except LLMProviderError as e:
    logger.error(f"Provider error: {e}")
```

### **4. Configuration Patterns**
```python
# Always use Pydantic for configuration
class MyProviderConfig(LLMProviderConfig):
    api_key: SecretStr = Field(..., description="API key")
    model: str = Field(default="default-model")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    @validator('model')
    def validate_model(cls, v):
        # Custom validation logic
        return v
```

### **5. Logging Patterns**
```python
from src.llmblocks.utils.logging import get_logger

class MyProvider:
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    async def process(self, data):
        self.logger.info(
            "Processing request",
            request_id=generate_id(),
            data_size=len(data)
        )
```

## 🚨 **Critical Issues & Solutions**

### **1. Environment Setup**
```bash
# ✅ CORRECT - Use uv for dependency management
uv sync

# ✅ CORRECT - Load environment variables
# Create .env file with API keys
echo "GOOGLE_API_KEY=your-key" > .env

# ✅ CORRECT - Test setup
python examples/test_gemini_simple.py
```

### **2. Common Import Errors**
```python
# ❌ WRONG - This will fail if dependencies not installed
from llmblocks.blocks.llm_provider import get_provider

# ✅ CORRECT - Development pattern
import sys
sys.path.insert(0, 'src')
from llmblocks.blocks.llm_provider import get_provider

# ✅ CORRECT - Production pattern (after pip install -e .)
from llmblocks.blocks.llm_provider import get_provider
```

### **3. LangChain Version Compatibility**
```toml
# pyproject.toml - Always use these versions or newer
"langchain>=0.3.26",
"langchain-core>=0.3.75", 
"langchain-openai>=0.3.32",
"langchain-anthropic>=0.3.19",
"langchain-google-genai>=2.1.10",
"langgraph>=0.6.6",
```

### **4. Pydantic Conflicts**
```python
# ❌ WRONG - Multiple Pydantic inheritance
class MyProvider(BaseBlock, BaseChatModel):  # MRO conflict!

# ✅ CORRECT - Composition pattern
class MyProvider(BaseChatModel):
    def __init__(self, config):
        super().__init__(**config.model_dump())
        # Compose BaseBlock functionality
        self._block_id = generate_id()
        self._status = BlockStatus.UNINITIALIZED
```

## 📚 **Key Files & Their Purpose**

### **Core Files**
```
src/llmblocks/core/
├── base_block.py      # BaseBlock class, BlockStatus enum
├── config.py          # ConfigManager, LLMBlocksConfig  
├── registry.py        # BlockRegistry for discovery
├── utils.py           # 20+ utility functions (NEW)
└── tracing.py         # Observability system (NEW)
```

### **LLM Provider System**
```
src/llmblocks/blocks/llm_provider/
├── base.py            # BaseLLMProvider (LangChain compatible)
├── factory.py         # LLMProviderFactory, get_provider()
├── gemini_provider.py # Google Gemini (PRODUCTION READY)
├── openai_provider.py # OpenAI GPT (PRODUCTION READY)
└── anthropic_provider.py # Anthropic Claude (PRODUCTION READY)
```

### **Utilities**
```
src/llmblocks/utils/
├── exceptions.py      # Custom exception hierarchy
├── logging.py         # Structured logging with structlog
└── langgraph_compat.py # LangGraph MRO compatibility (NEW)
```

### **Test Files**
```
examples/
├── test_gemini_simple.py          # Basic Gemini testing
├── test_latest_compatibility.py   # LangChain/LangGraph compatibility
├── test_langgraph_node.py        # LangGraph node creation
├── test_latest_langgraph.py      # Advanced LangGraph features
├── test_llm_providers.py         # Multi-provider testing
└── langchain_compatibility_demo.py # Integration examples
```

## 🎯 **Development Phases**

### **✅ Phase 1: Foundation & Core Architecture** (COMPLETE)
- Base block system (`BaseBlock`, `BlockStatus`, `BlockType`)
- Registry system (`BlockRegistry`)
- Configuration management (`ConfigManager`)
- Logging system (`get_logger`, structured logging)
- Error handling (`LLMBlocksError` hierarchy)
- Utility functions (20+ helpers in `utils.py`)
- Tracing system (`TraceCollector`, observability)

### **✅ Phase 2: Enhanced LLM Provider System** (COMPLETE)
- `BaseLLMProvider` with LangChain integration
- Multi-provider support (OpenAI, Gemini, Anthropic)
- Async/sync operations
- Streaming support
- LangGraph compatibility with MRO-safe approach
- Comprehensive error handling and retry logic
- Factory pattern for provider creation
- Full test suite (7 test files)

### **🚧 Phase 3: Memory & State Management** (NEXT)
**Planned Components:**
- `BaseMemoryProvider` abstract class
- `InMemoryProvider` - Fast, temporary storage
- `RedisMemoryProvider` - Persistent, scalable storage  
- `PostgreSQLMemoryProvider` - Enterprise-grade persistence
- `VectorMemoryProvider` - Embeddings and semantic search
- Session management and state persistence
- Memory factory and registry system

**Implementation Notes:**
- Follow same patterns as LLM providers
- Use composition over inheritance
- Async-first design
- Comprehensive error handling
- Full LangChain integration

### **🚧 Phase 4: Advanced Components** (PLANNED)
**RAG System:**
- Document loaders (PDF, text, web, etc.)
- Text splitters and chunking strategies
- Vector store integrations (ChromaDB, Pinecone, etc.)
- Retrieval strategies (similarity, MMR, etc.)
- RAG chain orchestration

**Agent System:**
- Multi-tool agent framework
- Conversation agents with memory
- Task-oriented agents
- Agent orchestration and coordination
- Tool integration and management

### **🚧 Phase 5: Developer Experience** (PLANNED)
**CLI Framework:**
- Project scaffolding (`llmblocks init`)
- Configuration management (`llmblocks config`)
- Development tools (`llmblocks dev`)
- Deployment helpers (`llmblocks deploy`)

**Web Playground:**
- Interactive development environment
- Agent testing interface
- RAG experimentation tools
- Workflow visualization
- Real-time debugging and monitoring

## 🔄 **Git Workflow**

### **Branch Strategy**
- `main` - Stable releases
- `implementation-plan` - Current development branch
- `feature/*` - Feature branches
- `hotfix/*` - Critical fixes

### **Commit Message Format**
```
feat: Add new LLM provider support
fix: Resolve LangGraph MRO conflict  
docs: Update API documentation
test: Add comprehensive provider tests
refactor: Improve error handling
```

### **Development Workflow**
1. Create feature branch from `implementation-plan`
2. Implement changes following guidelines
3. Add/update tests
4. Update documentation
5. Run linting and tests
6. Submit PR to `implementation-plan`

## 🧪 **Testing Guidelines**

### **Test Categories**
1. **Unit Tests** - Individual component testing
2. **Integration Tests** - Component interaction testing
3. **Compatibility Tests** - LangChain/LangGraph compatibility
4. **End-to-End Tests** - Full workflow testing

### **Test Execution**
```bash
# Run specific test
python examples/test_gemini_simple.py

# Run with environment setup
echo "GOOGLE_API_KEY=your-key" > .env
python examples/test_gemini_simple.py

# Test LangChain compatibility
python examples/test_latest_compatibility.py

# Test LangGraph features
python examples/test_latest_langgraph.py
```

### **Test Requirements**
- All tests must handle missing API keys gracefully
- Use environment variables for sensitive data
- Include both positive and negative test cases
- Test async and sync operations
- Validate error handling paths

## 📊 **Performance Considerations**

### **Async Operations**
- Always prefer `async def` for I/O operations
- Use `asyncio.gather()` for concurrent operations
- Implement proper connection pooling
- Handle timeouts and cancellation

### **Memory Management**
- Use generators for streaming responses
- Implement proper cleanup in `__aexit__`
- Monitor memory usage in long-running operations
- Use weak references where appropriate

### **Error Recovery**
- Implement exponential backoff for retries
- Use circuit breaker pattern for failing services
- Log errors with sufficient context
- Provide meaningful error messages to users

## 🔐 **Security Guidelines**

### **API Key Management**
```python
# ✅ CORRECT - Use SecretStr for API keys
from pydantic import SecretStr

class Config(BaseModel):
    api_key: SecretStr = Field(..., description="API key")
    
    def get_api_key(self) -> str:
        return self.api_key.get_secret_value()
```

### **Environment Variables**
```bash
# ✅ CORRECT - Use .env files (never commit)
echo "GOOGLE_API_KEY=your-key" > .env
echo ".env" >> .gitignore

# ✅ CORRECT - Load with python-dotenv
from dotenv import load_dotenv
load_dotenv()
```

### **Input Validation**
- Always validate user inputs with Pydantic
- Sanitize file paths and URLs
- Implement rate limiting for API calls
- Use type hints and runtime validation

## 🚀 **Deployment Considerations**

### **Production Checklist**
- [ ] All API keys in environment variables
- [ ] Proper logging configuration
- [ ] Error monitoring and alerting
- [ ] Health check endpoints
- [ ] Resource limits and timeouts
- [ ] Graceful shutdown handling

### **Docker Considerations**
```dockerfile
# Use multi-stage builds
FROM python:3.11-slim as builder
# Install dependencies
FROM python:3.11-slim as runtime
# Copy only necessary files
```

### **Monitoring**
- Use structured logging for observability
- Implement health checks for all providers
- Monitor API usage and rate limits
- Track performance metrics
- Set up alerting for failures

## 📝 **Documentation Standards**

### **Code Documentation**
```python
class MyProvider(BaseLLMProvider):
    """
    Custom LLM provider implementation.
    
    This provider supports:
    - Feature 1
    - Feature 2
    
    Args:
        config: Provider configuration
        **kwargs: Additional parameters
        
    Example:
        >>> provider = MyProvider(config)
        >>> response = await provider.generate("Hello")
    """
```

### **README Updates**
- Keep status table current
- Update examples with working code
- Include installation instructions
- Document breaking changes
- Provide troubleshooting guide

### **API Documentation**
- Use docstrings for all public methods
- Include parameter types and descriptions
- Provide usage examples
- Document exceptions that can be raised
- Keep examples up to date

## 🎯 **Agent Instructions Summary**

When working on LLMBlocks:

1. **ALWAYS** use `src/llmblocks/` structure
2. **NEVER** create multiple Pydantic inheritance
3. **ALWAYS** use composition for BaseBlock functionality
4. **ALWAYS** handle LangGraph MRO conflicts gracefully
5. **ALWAYS** make metadata a mutable dict for LangChain
6. **ALWAYS** use async-first design patterns
7. **ALWAYS** implement comprehensive error handling
8. **ALWAYS** add tests for new functionality
9. **ALWAYS** update documentation
10. **ALWAYS** follow the established patterns

**Remember**: LLMBlocks is production-ready for LLM providers. Focus on maintaining this quality as we expand to memory, RAG, and agent systems in future phases.

---

*This knowledge base should be updated as the project evolves. Always refer to the latest version when making decisions.* 🤖
