# Contributing to LLMBlocks

Thank you for your interest in contributing to LLMBlocks! We welcome contributions from everyone, whether you're fixing a bug, adding a feature, improving documentation, or sharing ideas.

## 🎯 **Vision & Values**

LLMBlocks aims to **democratize AI development** by making it as simple as writing a function call while maintaining enterprise-grade capabilities. Our core values:

- **Simplicity**: Complex AI should be simple to use
- **Reliability**: Production-ready from day one  
- **Compatibility**: Works with existing AI ecosystems
- **Performance**: Enterprise-grade speed and scale
- **Community**: Built by developers, for developers

## 🚀 **Quick Start**

### 1. Fork & Clone
```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/llmblocks.git
cd llmblocks
```

### 2. Development Setup
```bash
# Install uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync --extra dev

# Or with pip
pip install -e .[dev]
```

### 3. Verify Setup
```bash
# Run tests
uv run pytest

# Run code quality checks
uv run black --check src/ tests/
uv run isort --check src/ tests/
uv run flake8 src/ tests/
uv run mypy src/
```

## 🛠️ **Development Workflow**

### Branch Strategy
- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/your-feature` - Your feature branch
- `bugfix/issue-number` - Bug fix branches

### Making Changes
1. **Create a branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following our coding standards

3. **Add tests** for new functionality

4. **Run the test suite**:
   ```bash
   uv run pytest
   ```

5. **Check code quality**:
   ```bash
   uv run black src/ tests/
   uv run isort src/ tests/
   uv run flake8 src/ tests/
   uv run mypy src/
   ```

6. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

7. **Push and create PR**:
   ```bash
   git push origin feature/amazing-feature
   ```

## 📝 **Coding Standards**

### Code Style
- **Black** for code formatting (line length: 88)
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

### Code Quality
- **Type hints** for all public APIs
- **Docstrings** for all public functions/classes
- **Async/await** for all I/O operations
- **Error handling** with proper exception types
- **Logging** using structlog

### Example Code Style
```python
"""
Module docstring explaining the purpose.
"""

import asyncio
from typing import Dict, List, Optional, Any

from llmblocks.core.base_block import BaseBlock
from llmblocks.utils.logging import get_logger


class ExampleBlock(BaseBlock):
    """
    Example block demonstrating coding standards.
    
    This class shows how to write code that follows LLMBlocks
    conventions and best practices.
    
    Args:
        config: Configuration for the block
        
    Example:
        ```python
        block = ExampleBlock(config)
        await block.initialize()
        result = await block.process("input")
        ```
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = get_logger("ExampleBlock")
    
    async def _initialize_impl(self) -> None:
        """Initialize the example block."""
        self.logger.info("Initializing example block")
    
    async def process(self, input_data: str) -> str:
        """
        Process input data and return result.
        
        Args:
            input_data: The data to process
            
        Returns:
            Processed result
            
        Raises:
            ValueError: If input_data is empty
        """
        if not input_data:
            raise ValueError("Input data cannot be empty")
        
        self.logger.debug("Processing input", input_length=len(input_data))
        
        # Process the data
        result = f"Processed: {input_data}"
        
        self.logger.info("Processing complete", output_length=len(result))
        return result
```

## 🧪 **Testing Guidelines**

### Test Structure
```
tests/
├── unit/           # Unit tests (fast, isolated)
├── integration/    # Integration tests (components together)
├── e2e/           # End-to-end tests (full workflows)
└── conftest.py    # Shared test fixtures
```

### Writing Tests
```python
import pytest
from unittest.mock import AsyncMock, MagicMock

from llmblocks.blocks.example import ExampleBlock


class TestExampleBlock:
    """Test suite for ExampleBlock."""
    
    @pytest.mark.asyncio
    async def test_process_success(self):
        """Test successful processing."""
        config = {"setting": "value"}
        block = ExampleBlock(config)
        await block.initialize()
        
        result = await block.process("test input")
        
        assert result == "Processed: test input"
    
    @pytest.mark.asyncio
    async def test_process_empty_input(self):
        """Test processing with empty input."""
        config = {"setting": "value"}
        block = ExampleBlock(config)
        await block.initialize()
        
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            await block.process("")
    
    @pytest.fixture
    async def example_block(self):
        """Fixture providing an initialized ExampleBlock."""
        config = {"setting": "test_value"}
        block = ExampleBlock(config)
        await block.initialize()
        yield block
        await block.close()
```

### Test Categories
- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test component interactions
- **E2E tests**: Test complete user workflows
- **Performance tests**: Test speed and memory usage

### Running Tests
```bash
# All tests
uv run pytest

# Specific category
uv run pytest tests/unit/
uv run pytest -m integration
uv run pytest -m slow

# With coverage
uv run pytest --cov=llmblocks --cov-report=html

# Parallel execution
uv run pytest -n auto
```

## 📚 **Documentation**

### Documentation Types
- **README.md**: Project overview and quick start
- **API Documentation**: Generated from docstrings
- **Guides**: Step-by-step tutorials
- **Examples**: Real-world usage examples

### Writing Documentation
- Use **clear, concise language**
- Include **code examples** for all features
- Add **type hints** and **docstrings**
- Update **CHANGELOG.md** for changes

### Documentation Standards
```python
def example_function(param1: str, param2: Optional[int] = None) -> Dict[str, Any]:
    """
    Brief description of what the function does.
    
    Longer description with more details about the function's
    behavior, use cases, and any important considerations.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the optional second parameter
        
    Returns:
        Description of what the function returns
        
    Raises:
        ValueError: When param1 is empty
        RuntimeError: When operation fails
        
    Example:
        ```python
        result = example_function("hello", 42)
        print(result["status"])
        ```
        
    Note:
        Any important notes or warnings about usage.
    """
```

## 🎯 **Contribution Types**

### 🐛 **Bug Fixes**
1. **Check existing issues** first
2. **Create an issue** if none exists
3. **Reference the issue** in your PR
4. **Add regression tests**

### ✨ **New Features**
1. **Discuss in an issue** first
2. **Follow the architecture** patterns
3. **Add comprehensive tests**
4. **Update documentation**
5. **Add examples** if applicable

### 📖 **Documentation**
1. **Fix typos and errors**
2. **Improve clarity**
3. **Add missing examples**
4. **Update outdated information**

### 🧪 **Tests**
1. **Improve test coverage**
2. **Add edge case tests**
3. **Performance tests**
4. **Integration tests**

### 🔧 **Infrastructure**
1. **CI/CD improvements**
2. **Build process enhancements**
3. **Development tooling**
4. **Performance optimizations**

## 🏗️ **Architecture Guidelines**

### Design Principles
- **Modular**: Everything is a composable block
- **Async-first**: All I/O operations are async
- **Type-safe**: Full type hints throughout
- **Configurable**: Pydantic-based configuration
- **Observable**: Comprehensive logging and metrics

### Adding New Blocks
1. **Inherit from BaseBlock**
2. **Implement required abstract methods**
3. **Add proper configuration**
4. **Include comprehensive tests**
5. **Update factory registration**

### Example Block Structure
```python
from llmblocks.core.base_block import BaseBlock, BlockConfig

class NewBlockConfig(BlockConfig):
    """Configuration for NewBlock."""
    setting1: str
    setting2: int = 10

class NewBlock(BaseBlock):
    """New block implementation."""
    
    def __init__(self, config: NewBlockConfig):
        super().__init__(config)
        self.config = config
    
    async def _initialize_impl(self) -> None:
        """Initialize the block."""
        pass
    
    async def _close_impl(self) -> None:
        """Close the block."""
        pass
```

## 🔍 **Code Review Process**

### PR Requirements
- [ ] **Tests pass** (all categories)
- [ ] **Code quality** checks pass
- [ ] **Documentation** updated
- [ ] **CHANGELOG.md** updated
- [ ] **Examples** added (if applicable)

### Review Checklist
- **Functionality**: Does it work as intended?
- **Tests**: Are there sufficient tests?
- **Performance**: Any performance implications?
- **Security**: Any security concerns?
- **Documentation**: Is it well documented?
- **Compatibility**: Maintains backward compatibility?

### Review Process
1. **Automated checks** run first
2. **Maintainer review** for approval
3. **Community feedback** welcome
4. **Merge** after approval

## 🚀 **Release Process**

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Steps
1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md**
3. **Create release PR**
4. **Tag release** after merge
5. **Automated deployment** to PyPI

## 🤝 **Community**

### Communication Channels
- **[GitHub Issues](https://github.com/llmblocks/llmblocks/issues)**: Bug reports, feature requests
- **[GitHub Discussions](https://github.com/llmblocks/llmblocks/discussions)**: General discussion, Q&A
- **[Discord](https://discord.gg/llmblocks)**: Real-time chat (coming soon)

### Code of Conduct
We follow the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). Please read it before participating.

### Getting Help
- **Check documentation** first
- **Search existing issues**
- **Ask in discussions**
- **Create an issue** if needed

## 🎉 **Recognition**

Contributors are recognized in:
- **CONTRIBUTORS.md** file
- **Release notes**
- **GitHub contributors** page
- **Special mentions** for significant contributions

## 📋 **Checklist for Contributors**

Before submitting a PR:

- [ ] Code follows style guidelines
- [ ] Tests are added and passing
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Commit messages are clear
- [ ] PR description explains changes
- [ ] No breaking changes (or clearly marked)

## 🙏 **Thank You**

Every contribution, no matter how small, helps make LLMBlocks better for everyone. Thank you for being part of our community!

---

**Questions?** Feel free to ask in [GitHub Discussions](https://github.com/llmblocks/llmblocks/discussions) or create an [issue](https://github.com/llmblocks/llmblocks/issues).

**Ready to contribute?** Check out our [good first issues](https://github.com/llmblocks/llmblocks/labels/good%20first%20issue) to get started!
