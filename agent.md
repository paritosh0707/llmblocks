# 🤖 Agent Knowledge Base - LLMBlocks Project

This document contains comprehensive instructions, project knowledge, and important decisions for AI agents working on the LLMBlocks project.

## 📋 **Project Overview**

**LLMBlocks** is a modular, enterprise-grade AI application framework designed as "Lego blocks for AI applications." The project follows a phased development approach with a block-based architecture.

### **Current Status (August 2025)**
- **Phase 1 & 2**: ✅ **COMPLETE** - Foundation + LLM Provider System + User Experience Improvements
- **Phase 3**: 🚧 **NEXT** - Memory & State Management  
- **Phase 4**: 🚧 **PLANNED** - RAG & Agent Systems
- **Phase 5**: 🚧 **PLANNED** - Developer Experience & Playground

### **Latest Updates (August 2025)**
- ✅ **Simplified Input Format** - No more LLMMessage objects required
- ✅ **Environment-Controlled Logging** - Clean output by default
- ✅ **Deprecation Fixes** - Updated to latest LangChain/Pydantic patterns
- ✅ **LangChain-Style API** - String/dict input like LangChain

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

### **5. Simplified Input Format (August 2025)**

**MAJOR IMPROVEMENT**: Users no longer need to create LLMMessage objects manually.

#### **The Problem**
```python
# Old way - Complex and verbose
from llmblocks.blocks.llm_provider import get_provider, LLMMessage, LLMRole
provider = await get_provider("gemini", api_key=api_key)
response = await provider.generate([LLMMessage(role=LLMRole.USER, content="Hello!")])
```

#### **The Solution**
```python
# New way - Simple and intuitive
from llmblocks.blocks.llm_provider import get_provider
provider = await get_provider("gemini", api_key=api_key)
response = await provider.generate("Hello!")  # Just a string!

# Or dict format (LangChain-style)
response = await provider.generate({"role": "user", "content": "Hello!"})

# Or multiple messages
response = await provider.generate([
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
    {"role": "user", "content": "How are you?"}
])
```

**Implementation**: Added `_normalize_messages()` method in `BaseLLMProvider` that converts:
- `str` → `[LLMMessage(role=USER, content=str)]`
- `dict` → `[LLMMessage(role=dict["role"], content=dict["content"])]`
- `List[Union[str, dict, LLMMessage]]` → `List[LLMMessage]`

### **6. Environment-Controlled Logging (August 2025)**

**MAJOR IMPROVEMENT**: Clean output by default, logging only when needed.

#### **The Problem**
```bash
# Old way - Always verbose JSON logging
🚀 LLMBlocks Hello World - Just 3 lines!
2025-08-30 15:33:00,903 - TraceCollector - INFO - {"timestamp": "2025-08-30T10:03:00.903700Z", "logger": "TraceCollector"...
# 50+ lines of JSON spam
🤖 AI Response: Hello!
```

#### **The Solution**
```bash
# New way - Clean by default
🚀 LLMBlocks Hello World - Just 3 lines!
🤖 AI Response: Hello!
✅ That's it! Just 3 lines of code for AI power!

# Enable logging when debugging
LLMBLOCKS_ENABLE_LOGGING=true uv run python example.py
# Now shows readable logs when needed
```

**Environment Variables**:
- `LLMBLOCKS_ENABLE_LOGGING=false` (default) - No logging
- `LLMBLOCKS_ENABLE_LOGGING=true` - Enable logging
- `LLMBLOCKS_LOG_FORMAT=readable` (default) - Human-readable format
- `LLMBLOCKS_LOG_FORMAT=json` - JSON format for production

**Implementation**: 
- Updated `src/llmblocks/utils/logging.py` with environment-controlled configuration
- Added `NoOpLogger` class for disabled logging
- Modified `get_logger()` to respect environment settings

### **7. Deprecation Fixes (August 2025)**

**CRITICAL UPDATES**: Removed all deprecated code patterns.

#### **Fixed Deprecations**:
1. **LangChain Google GenAI**: Removed `convert_system_message_to_human=True` (deprecated)
2. **Pydantic v2**: Updated `@validator` → `@field_validator` with `@classmethod`

#### **Before (Deprecated)**:
```python
from pydantic import validator

class Config(BaseModel):
    @validator('model')
    def validate_model(cls, v):  # Missing @classmethod
        return v

# LangChain config
client_config = {
    "convert_system_message_to_human": True,  # Deprecated!
}
```

#### **After (Modern)**:
```python
from pydantic import field_validator

class Config(BaseModel):
    @field_validator('model')
    @classmethod  # Required in Pydantic v2
    def validate_model(cls, v):
        return v

# LangChain config - system messages handled natively
client_config = {
    # Note: convert_system_message_to_human is deprecated
    # System messages are now handled natively by the model
}
```

### **8. Metadata Compatibility**

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
# Setup checker
uv run python examples/minimal/00_check_setup.py

# Minimal examples (clean output)
uv run python examples/minimal/01_hello_world.py        # 3 lines
uv run python examples/minimal/02_smart_chatbot.py      # 5 lines  
uv run python examples/minimal/03_streaming_ai.py       # 4 lines

# With logging enabled
LLMBLOCKS_ENABLE_LOGGING=true uv run python examples/minimal/01_hello_world.py

# Integration tests
uv run python tests/integration/test_gemini_simple.py
uv run python tests/integration/test_latest_compatibility.py
```

### **Current Examples (August 2025)**

| Example | Lines | Input Format | Output |
|---------|-------|--------------|--------|
| `00_check_setup.py` | Setup | N/A | Dependency verification |
| `01_hello_world.py` | **3** | `"Hello!"` | Clean AI response |
| `02_smart_chatbot.py` | **5** | `{"role": "user", "content": "..."}` | Interactive chat |
| `03_streaming_ai.py` | **4** | `"Tell me a story"` | Real-time streaming |
| `04_multi_provider.py` | **6** | Multiple formats | Provider switching |
| `05_langchain_magic.py` | **3** | LangChain integration | Ecosystem compatibility |
| `06_ai_workflow.py` | **7** | LangGraph workflow | Complete workflow |
| `07_production_ready.py` | **10** | Enterprise features | Monitoring & observability |
| `08_zero_config.py` | **2** | Absolute minimum | Zero configuration |

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

### **🆕 New Patterns (August 2025):**

11. **ALWAYS** support simplified input formats (str/dict/list)
12. **ALWAYS** use environment-controlled logging (`LLMBLOCKS_ENABLE_LOGGING`)
13. **ALWAYS** use modern Pydantic patterns (`@field_validator` + `@classmethod`)
14. **NEVER** use deprecated LangChain parameters (`convert_system_message_to_human`)
15. **ALWAYS** provide clean output by default (no logging spam)
16. **ALWAYS** make APIs LangChain-compatible for user familiarity

**Remember**: LLMBlocks is production-ready for LLM providers. Focus on maintaining this quality as we expand to memory, RAG, and agent systems in future phases.

---

*This knowledge base should be updated as the project evolves. Always refer to the latest version when making decisions.* 🤖
