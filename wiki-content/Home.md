# 🚀 LLMBlocks Wiki - The Dream of 3-Line Stateful AI

Welcome to the **LLMBlocks Wiki** - your comprehensive guide to building powerful AI applications with minimal code!

[![CI/CD Pipeline](https://github.com/paritosh0707/llmblocks/workflows/CI/CD%20Pipeline/badge.svg)](https://github.com/paritosh0707/llmblocks/actions)
[![PyPI version](https://badge.fury.io/py/llmblocks.svg)](https://badge.fury.io/py/llmblocks)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

---

## 🎯 **The Vision**

Transform complex AI frameworks into pure simplicity. Create powerful, stateful AI assistants with just **3 lines of code**:

```python
# The dream realized: 3-line stateful AI!
ai = await get_stateful_ai("gemini", api_key="your-key")
response1 = await ai.chat("Hi! I'm Alice, a data scientist from NYC.")
response2 = await ai.chat("What do you know about me?")  # AI remembers: "Alice, data scientist, NYC"!
```

---

## 📖 **Wiki Navigation**

### 🚀 **Getting Started**
- **[[Quick Start]]** - Get up and running in 5 minutes
- **[[Installation]]** - Installation methods and requirements
- **[[Your First AI]]** - Build your first AI in 3 lines
- **[[Configuration]]** - Environment setup and API keys

### 🧠 **Core Concepts**
- **[[LLM Providers]]** - OpenAI, Gemini, Anthropic, and more
- **[[Memory System]]** - Conversation memory and state management
- **[[Streaming]]** - Real-time AI responses
- **[[LangChain Integration]]** - LangChain and LangGraph compatibility

### 📚 **Examples & Tutorials**
- **[[Minimal Examples]]** - 2-10 line code samples
- **[[Memory Examples]]** - Stateful AI conversations
- **[[Advanced Examples]]** - Complex AI applications
- **[[Use Cases]]** - Real-world applications

### 🔧 **API Reference**
- **[[LLM Provider API]]** - Complete provider reference
- **[[Memory API]]** - Memory system reference
- **[[Factory Functions]]** - Convenience functions
- **[[Configuration API]]** - Configuration options

### 🚀 **Advanced Topics**
- **[[Custom Providers]]** - Building your own providers
- **[[Production Deployment]]** - Docker, scaling, monitoring
- **[[Performance Optimization]]** - Speed and efficiency tips
- **[[Security Best Practices]]** - API keys, rate limiting, safety

### 🛠️ **Development**
- **[[Contributing]]** - How to contribute to LLMBlocks
- **[[Testing]]** - Running tests and adding new ones
- **[[Architecture]]** - Internal architecture and design
- **[[Troubleshooting]]** - Common issues and solutions

---

## ✨ **What Makes LLMBlocks Special**

### 🧠 **Intelligent Memory Management**
- **Automatic Context**: AI remembers your entire conversation
- **Smart Summarization**: Efficiently manages long conversations
- **Multiple Backends**: In-memory, file, Redis support
- **Session Management**: Multiple conversation threads

### 🔌 **Universal Compatibility**
- **Multiple Providers**: OpenAI, Gemini, Anthropic, local models
- **LangChain Integration**: Full LCEL and StateGraph support
- **Streaming Support**: Real-time responses
- **Async/Await**: High-performance async operations

### 🎯 **Developer Experience**
- **Minimal Code**: 2-10 lines for most use cases
- **Type Safety**: Full Pydantic v2 validation
- **Error Handling**: Comprehensive error management
- **Production Ready**: Docker, CI/CD, monitoring

---

## 🏆 **Quick Stats**

| Feature | Status |
|---------|--------|
| **Test Coverage** | 100% (59/59 tests passing) ✅ |
| **Deprecation Warnings** | 0 warnings ✅ |
| **LangChain Compatibility** | Full support ✅ |
| **Production Ready** | Docker + CI/CD ✅ |
| **Documentation** | Comprehensive ✅ |

---

## 🚀 **Quick Examples**

### **Hello World (2 lines)**
```python
ai = await get_provider("gemini", api_key="your-key")
print(await ai.generate("Hello, world!"))
```

### **Streaming AI (4 lines)**
```python
ai = await get_provider("gemini", stream=True)
async for chunk in ai.generate_stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### **Stateful AI (3 lines)**
```python
ai = await get_stateful_ai("gemini", api_key="your-key")
await ai.chat("I'm Alice, a data scientist from NYC.")
response = await ai.chat("What do you know about me?")  # Remembers Alice!
```

---

## 🤝 **Community & Support**

- **GitHub Issues**: [Report bugs or request features](https://github.com/paritosh0707/llmblocks/issues)
- **Discussions**: [Community discussions](https://github.com/paritosh0707/llmblocks/discussions)
- **Contributing**: See our [[Contributing]] guide
- **License**: MIT License

---

## 🎊 **Ready to Build Amazing AI?**

Start with our **[[Quick Start]]** guide and build your first AI application in minutes!

**Happy coding! 🚀✨**
