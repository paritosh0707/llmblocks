# 🚀 LLMBlocks Minimal Examples

**"Less Code, More Power"** - These examples showcase how LLMBlocks enables powerful AI functionality with minimal code.

## 🎯 **Philosophy**

LLMBlocks follows the principle of **"Lego blocks for AI"** - simple, composable pieces that create powerful applications with minimal effort.

## 📚 **Examples Overview**

| Example | Lines of Code | What It Does |
|---------|---------------|--------------|
| **01_hello_world.py** | **3 lines** | Basic AI interaction |
| **02_smart_chatbot.py** | **5 lines** | Conversational chatbot with memory |
| **03_streaming_ai.py** | **4 lines** | Real-time streaming responses |
| **04_multi_provider.py** | **6 lines** | Switch between AI providers |
| **05_langchain_magic.py** | **3 lines** | LangChain integration |
| **06_ai_workflow.py** | **7 lines** | Complete LangGraph workflow |
| **07_production_ready.py** | **10 lines** | Enterprise-grade AI service |
| **08_zero_config.py** | **2 lines** | Absolute minimum AI code |

## 🚀 **Quick Start**

### **Setup (One Time)**
```bash
# 1. Clone and install
git clone https://github.com/paritosh0707/llmblocks.git
cd llmblocks
uv sync

# 2. Get your free API key from Google AI Studio
# https://aistudio.google.com/app/apikey

# 3. Set your API key
export GOOGLE_API_KEY="your-api-key-here"
```

### **Run Examples**
```bash
# Hello World - Just 3 lines!
python examples/minimal/01_hello_world.py

# Smart Chatbot - Just 5 lines!
python examples/minimal/02_smart_chatbot.py

# Streaming AI - Just 4 lines!
python examples/minimal/03_streaming_ai.py

# Try them all!
for example in examples/minimal/*.py; do
    echo "Running $example..."
    python "$example"
    echo "---"
done
```

## 💡 **Key Features Demonstrated**

### **🔥 Simplicity**
```python
# Just 2 lines for AI!
from llmblocks.blocks.llm_provider import get_provider
response = await get_provider('gemini').generate('Hello!')
```

### **🌊 Streaming**
```python
# Real-time streaming in 4 lines
provider = get_provider("gemini", api_key=api_key)
async for chunk in provider.generate_stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### **🔄 Multi-Provider**
```python
# Switch providers seamlessly
gemini = get_provider("gemini", api_key=gemini_key)
openai = get_provider("openai", api_key=openai_key)
# Use either one with identical interface!
```

### **🔗 LangChain Integration**
```python
# Convert to LangChain LLM in 1 line
langchain_llm = provider.as_langchain()
# Now use with any LangChain chain or agent!
```

### **⚡ LangGraph Workflows**
```python
# Create workflow nodes in 2 lines
llm_node = provider.create_langgraph_node("assistant")
graph = provider.create_langgraph_graph("workflow")
```

### **🚀 Production Ready**
```python
# Enterprise features built-in
health = await provider.health_check()  # Health monitoring
logger.info("Request processed")        # Structured logging
with trace_span("ai_request"):         # Distributed tracing
    response = await provider.generate(prompt)
```

## 🎯 **Design Principles**

### **1. Minimal Code**
- **2-10 lines** for complete functionality
- **Zero boilerplate** - just the essential logic
- **Sensible defaults** - works out of the box

### **2. Maximum Power**
- **Enterprise features** included (logging, monitoring, error handling)
- **Production ready** from day one
- **Full ecosystem integration** (LangChain, LangGraph)

### **3. Developer Experience**
- **Clear error messages** with helpful suggestions
- **Comprehensive examples** for every use case
- **Copy-paste ready** code snippets

## 🔧 **Customization**

Each example can be customized with additional parameters:

```python
# Basic usage
provider = get_provider("gemini")

# With custom configuration
provider = get_provider(
    provider_name="gemini",
    api_key="your-key",
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    max_tokens=1000,
    timeout=30.0
)
```

## 🚨 **Troubleshooting**

### **API Key Issues**
```bash
# Check if API key is set
echo $GOOGLE_API_KEY

# Set API key for current session
export GOOGLE_API_KEY="your-api-key"

# Set permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export GOOGLE_API_KEY="your-api-key"' >> ~/.bashrc
```

### **Import Errors**
```bash
# Make sure you're in the project directory
cd llmblocks

# Install dependencies
uv sync
# or
pip install -e ".[dev]"
```

### **LangChain/LangGraph Issues**
```bash
# Install optional dependencies
pip install langchain langgraph
```

## 🎉 **What's Next?**

After trying these minimal examples:

1. **📚 Read the docs** - `docs/index.md` for comprehensive guide
2. **🔍 Explore advanced examples** - `examples/advanced/` for complex use cases
3. **🧪 Run tests** - `tests/integration/` to see full capabilities
4. **🚀 Build your app** - Use these patterns in your own projects

## 💬 **Community**

- **Issues**: [GitHub Issues](https://github.com/paritosh0707/llmblocks/issues)
- **Discussions**: [GitHub Discussions](https://github.com/paritosh0707/llmblocks/discussions)

---

**LLMBlocks: Building the future of AI applications, one block at a time.** 🚀

*Less code, more power!* ⚡
