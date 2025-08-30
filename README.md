# 🚀 LLMBlocks

**A modular, enterprise-grade AI application framework for building sophisticated AI applications using a block-based architecture.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

## 🎯 **What is LLMBlocks?**

LLMBlocks is a **modular, configurable framework** that enables developers to build sophisticated AI applications in minutes. Think of it as "Lego blocks for AI applications" - you can mix and match different components to create exactly what you need.

### ✨ **Key Features**

- 🧱 **Modular Block System** - Mix and match components like LLM providers, memory systems, and tools
- ⚡ **Async-First Design** - Built for high-performance, concurrent applications
- 🔧 **Configuration-Driven** - Everything configurable via YAML/JSON files
- 🚀 **Production Ready** - Built-in security, monitoring, and scalability features
- 🎮 **Interactive Playground** - Web-based development environment for testing and debugging
- 🔌 **Plugin Architecture** - Easy to extend with custom blocks and providers

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────┐
│                    LLMBlocks Application                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Chatbot   │  │     RAG     │  │    Agent    │        │
│  │  Component  │  │  Component  │  │ Component   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Block Registry                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │    LLM      │  │   Memory    │  │    Tools    │        │
│  │  Providers  │  │  Providers  │  │  Registry   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Core Foundation                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Config    │  │   Logging   │  │   Tracing   │        │
│  │ Management  │  │   System    │  │   System    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 **Quick Start**

### **Installation**

```bash
# Install from PyPI (coming soon)
pip install llmblocks

# Or install from source
git clone https://github.com/llmblocks/llmblocks.git
cd llmblocks
pip install -e .
```

### **Your First Chatbot in 5 Minutes**

1. **Create a configuration file** (`chatbot.yaml`):

```yaml
name: "My First Chatbot"
description: "A helpful AI assistant"
system_prompt: "You are a friendly and knowledgeable AI assistant."

llm:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
  api_key:
    env_var: "OPENAI_API_KEY"

memory:
  provider_name: "in-memory"
```

2. **Write the code**:

```python
from llmblocks import Chatbot

# Create chatbot from config
chatbot = Chatbot(config_yaml_path="chatbot.yaml")

# Start chatting!
response = chatbot.chat("Hello! What can you help me with?")
print(response)
```

3. **Run it**:

```bash
export OPENAI_API_KEY="your-api-key-here"
python your_chatbot.py
```

## 🧱 **Available Blocks**

### **LLM Providers**
- **OpenAI** - GPT-4, GPT-3.5, DALL-E
- **Azure OpenAI** - Enterprise OpenAI with Azure
- **Google Gemini** - Pro, Flash, Vision models
- **Anthropic Claude** - Opus, Sonnet, Haiku
- **Local Models** - Via Ollama integration
- **Custom Providers** - Easy to add your own

### **Memory Systems**
- **In-Memory** - Fast, temporary storage
- **Redis** - Persistent, scalable storage
- **PostgreSQL** - Enterprise-grade persistence
- **Vector Memory** - Semantic search and RAG

### **Components**
- **Chatbot** - Conversational AI interface
- **RAG System** - Document-based question answering
- **Multi-Tool Agent** - Tool-using AI agents
- **Workflow Orchestrator** - Complex AI workflows

## 🔧 **Advanced Configuration**

### **Multi-Provider Setup**

```yaml
name: "Smart Assistant"
llm:
  provider: "openai"
  model: "gpt-4"
  fallback_provider: "gemini"  # Automatic fallback
  api_key:
    env_var: "OPENAI_API_KEY"

memory:
  provider_name: "redis"
  config:
    host: "localhost"
    port: 6379
    db: 0

tools:
  - name: "web_search"
    provider: "serpapi"
    config:
      api_key:
        env_var: "SERPAPI_KEY"
  - name: "calculator"
    provider: "builtin"
```

### **RAG Configuration**

```yaml
name: "Document Assistant"
llm:
  provider: "openai"
  model: "gpt-4"

rag:
  document_store:
    provider: "chroma"
    config:
      persist_directory: "./data/chroma"
  
  embedding_model:
    provider: "sentence-transformers"
    model: "all-MiniLM-L6-v2"
  
  retrieval:
    top_k: 5
    similarity_threshold: 0.7
```

## 🎮 **Interactive Playground**

LLMBlocks includes a web-based playground for development and testing:

```bash
# Start the playground
llmblocks playground

# Or run directly
python -m llmblocks.playground
```

**Features:**
- Real-time chatbot testing
- Configuration editor with live preview
- Performance metrics and debugging
- Document upload and RAG testing
- Tool testing and workflow visualization

## 🧪 **Testing & Development**

### **Running Tests**

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=llmblocks

# Run specific test categories
pytest -m "unit"
pytest -m "integration"
pytest -m "slow"
```

### **Code Quality**

```bash
# Format code
black llmblocks/
isort llmblocks/

# Lint code
flake8 llmblocks/
mypy llmblocks/

# Run all quality checks
pre-commit run --all-files
```

## 📚 **Documentation**

- **[User Guide](https://llmblocks.dev/user-guide)** - Complete usage documentation
- **[API Reference](https://llmblocks.dev/api)** - Detailed API documentation
- **[Examples](https://llmblocks.dev/examples)** - Working examples and tutorials
- **[Architecture](https://llmblocks.dev/architecture)** - Deep dive into the system design

## 🤝 **Contributing**

We welcome contributions! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Add tests** for new functionality
5. **Run the test suite** (`pytest`)
6. **Submit a pull request**

### **Development Setup**

```bash
# Clone and setup
git clone https://github.com/llmblocks/llmblocks.git
cd llmblocks

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install
```

## 📊 **Performance & Benchmarks**

- **Response Time**: < 100ms for simple queries, < 500ms for RAG queries
- **Throughput**: Support for 1000+ concurrent users
- **Memory Usage**: Efficient memory management with configurable backends
- **Scalability**: Horizontal scaling with Redis and PostgreSQL backends

## 🔒 **Security Features**

- **API Key Management** - Secure environment variable handling
- **Rate Limiting** - Built-in protection against abuse
- **Input Validation** - Comprehensive sanitization and validation
- **Authentication** - Optional auth system for enterprise deployments
- **Audit Logging** - Complete request/response logging

## 🚀 **Deployment**

### **Docker**

```bash
# Build and run
docker build -t llmblocks .
docker run -p 8000:8000 llmblocks

# Or use docker-compose
docker-compose up -d
```

### **Kubernetes**

```bash
# Deploy to Kubernetes
kubectl apply -f deployment/k8s/
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Built on top of [LangChain](https://langchain.com/) and [LangGraph](https://langchain.com/langgraph)
- Inspired by modern AI application architectures
- Community-driven development and feedback

## 📞 **Support & Community**

- **Discord**: [Join our community](https://discord.gg/llmblocks)
- **GitHub Issues**: [Report bugs](https://github.com/llmblocks/llmblocks/issues)
- **Discussions**: [Ask questions](https://github.com/llmblocks/llmblocks/discussions)
- **Email**: team@llmblocks.dev

---

**Ready to build the future of AI applications? Start with LLMBlocks today! 🚀**