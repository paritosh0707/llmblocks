# 🚨 Troubleshooting Guide

Complete troubleshooting guide for common LLMBlocks issues, errors, and solutions.

---

## 🎯 **Quick Diagnosis**

### **Common Error Patterns**

| Error Type | Typical Message | Quick Fix |
|------------|-----------------|-----------|
| **API Key** | `No API key found` | Set environment variable |
| **Import** | `ModuleNotFoundError` | Install package or activate venv |
| **Provider** | `Provider 'xyz' not available` | Check provider name spelling |
| **Memory** | `Backend connection failed` | Check backend configuration |
| **Rate Limit** | `Rate limit exceeded` | Implement retry logic |
| **Model** | `Model not found` | Use supported model name |

---

## 🔑 **API Key Issues**

### **Problem: No API Key Found**
```
Error: No API key found for provider 'gemini'
LLMProviderError: Authentication failed
```

#### **Solution 1: Environment Variables**
```bash
# Set API key
export GOOGLE_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Verify it's set
echo $GOOGLE_API_KEY
```

#### **Solution 2: .env File**
```bash
# Create .env file
cat > .env << EOF
GOOGLE_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
EOF

# Load in Python
from dotenv import load_dotenv
load_dotenv()
```

#### **Solution 3: Direct Configuration**
```python
# Pass API key directly
ai = await get_provider("gemini", api_key="your-api-key")
```

#### **Debugging API Keys**
```python
import os

# Check if API key is set
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ GOOGLE_API_KEY not set")
else:
    print(f"✅ API key found: {api_key[:10]}...")
```

---

## 📦 **Installation Issues**

### **Problem: Module Not Found**
```
ModuleNotFoundError: No module named 'llmblocks'
```

#### **Solution 1: Install Package**
```bash
# Install from PyPI
pip install llmblocks

# Or with uv
uv add llmblocks
```

#### **Solution 2: Virtual Environment**
```bash
# Check if virtual environment is activated
which python
pip list | grep llmblocks

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate  # Windows
```

#### **Solution 3: Development Installation**
```bash
# If installing from source
git clone https://github.com/paritosh0707/llmblocks.git
cd llmblocks
pip install -e .
```

### **Problem: Python Version**
```
ERROR: Python 3.10 is not supported. Please use Python 3.11+
```

#### **Solution: Upgrade Python**
```bash
# Check current version
python --version

# Install Python 3.11+ using pyenv
pyenv install 3.12.0
pyenv local 3.12.0

# Or use conda
conda install python=3.12
```

---

## 🤖 **Provider Issues**

### **Problem: Provider Not Available**
```
LLMProviderError: Provider 'gpt4' not available
```

#### **Solution: Use Correct Provider Names**
```python
# ❌ Wrong
ai = await get_provider("gpt4")
ai = await get_provider("chatgpt")

# ✅ Correct
ai = await get_provider("openai")  # For GPT models
ai = await get_provider("gemini")  # For Gemini models
ai = await get_provider("anthropic")  # For Claude models
```

#### **Check Available Providers**
```python
from llmblocks.blocks.llm_provider import get_factory

factory = get_factory()
providers = factory.list_providers()
print(f"Available providers: {providers}")
```

### **Problem: Model Not Found**
```
Error: Model 'gpt-5' not found
```

#### **Solution: Use Supported Models**
```python
# Gemini models
ai = await get_provider("gemini", model="gemini-2.0-flash")
ai = await get_provider("gemini", model="gemini-1.5-pro")

# OpenAI models
ai = await get_provider("openai", model="gpt-4o")
ai = await get_provider("openai", model="gpt-3.5-turbo")

# Anthropic models
ai = await get_provider("anthropic", model="claude-3-5-sonnet-20241022")
```

### **Problem: Rate Limit Exceeded**
```
RateLimitError: Rate limit exceeded. Retry after 60 seconds
```

#### **Solution: Implement Retry Logic**
```python
import asyncio
from llmblocks.blocks.llm_provider.base import RateLimitError

async def generate_with_retry(ai, prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await ai.generate(prompt)
        except RateLimitError as e:
            if attempt < max_retries - 1:
                print(f"Rate limited. Waiting {e.retry_after} seconds...")
                await asyncio.sleep(e.retry_after)
            else:
                raise
```

#### **Solution: Configure Rate Limits**
```python
# Set conservative rate limits
ai = await get_provider("openai",
    requests_per_minute=30,  # Lower than API limit
    tokens_per_minute=50000
)
```

---

## 🧠 **Memory Issues**

### **Problem: Memory Not Persisting**
```
# Messages are lost between sessions
ai = await get_stateful_ai("gemini")
# Memory is lost when script ends
```

#### **Solution: Use Persistent Backend**
```python
# ❌ In-memory (default) - not persistent
ai = await get_stateful_ai("gemini")

# ✅ File backend - persistent
from llmblocks.blocks.memory import MemoryConfig
config = MemoryConfig(backend="file", file_path="./memory.json")
ai = await get_stateful_ai("gemini", memory_config=config)

# ✅ Redis backend - persistent and scalable
config = MemoryConfig(backend="redis", redis_url="redis://localhost:6379")
ai = await get_stateful_ai("gemini", memory_config=config)
```

### **Problem: File Permission Denied**
```
PermissionError: [Errno 13] Permission denied: './conversations.json'
```

#### **Solution: Fix File Permissions**
```bash
# Create directory with proper permissions
mkdir -p ./conversations
chmod 755 ./conversations

# Or use user directory
import os
file_path = os.path.expanduser("~/llmblocks_conversations.json")
```

### **Problem: Redis Connection Failed**
```
ConnectionError: Error connecting to Redis
```

#### **Solution: Start Redis Server**
```bash
# Install Redis
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis

# Docker
docker run -d -p 6379:6379 redis:alpine

# Test connection
redis-cli ping  # Should return PONG
```

#### **Solution: Check Redis Configuration**
```python
# Test Redis connection
import redis

try:
    r = redis.Redis.from_url("redis://localhost:6379")
    r.ping()
    print("✅ Redis connection successful")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
```

---

## 🌊 **Streaming Issues**

### **Problem: Streaming Not Working**
```python
# No streaming output
ai = await get_provider("gemini")
async for chunk in ai.generate_stream("Hello"):
    print(chunk.content)  # Nothing prints
```

#### **Solution: Enable Streaming**
```python
# ✅ Enable streaming explicitly
ai = await get_provider("gemini", stream=True)

# Or check if streaming is supported
if ai.is_streaming_enabled:
    async for chunk in ai.generate_stream("Hello"):
        print(chunk.content, end="", flush=True)
else:
    response = await ai.generate("Hello")
    print(response.content)
```

### **Problem: Incomplete Streaming**
```python
# Streaming stops early
async for chunk in ai.generate_stream("Long story"):
    print(chunk.content, end="")
# Output is cut off
```

#### **Solution: Handle Stream Completion**
```python
try:
    async for chunk in ai.generate_stream("Long story"):
        if chunk.content:  # Check for content
            print(chunk.content, end="", flush=True)
    print()  # New line after completion
except Exception as e:
    print(f"\nStreaming error: {e}")
```

---

## 🔄 **Async/Await Issues**

### **Problem: Forgot to Await**
```python
# RuntimeWarning: coroutine was never awaited
ai = get_provider("gemini")  # Missing await
response = ai.generate("Hello")  # Missing await
```

#### **Solution: Use Async/Await Properly**
```python
import asyncio

async def main():
    # ✅ Await async functions
    ai = await get_provider("gemini")
    response = await ai.generate("Hello")
    print(response.content)

# ✅ Run with asyncio
asyncio.run(main())
```

### **Problem: Running in Jupyter/IPython**
```python
# SyntaxError: 'await' outside function
ai = await get_provider("gemini")  # In Jupyter cell
```

#### **Solution: Use Jupyter Async Support**
```python
# Jupyter automatically handles top-level await
ai = await get_provider("gemini")
response = await ai.generate("Hello")
print(response.content)

# Or use asyncio explicitly
import asyncio

async def main():
    ai = await get_provider("gemini")
    return await ai.generate("Hello")

response = await main()
```

---

## 🔧 **Configuration Issues**

### **Problem: Invalid Configuration**
```
ValidationError: 1 validation error for GeminiConfig
temperature: ensure this value is less than or equal to 1.0
```

#### **Solution: Use Valid Configuration Values**
```python
# ❌ Invalid values
ai = await get_provider("gemini", 
    temperature=2.0,  # Must be 0.0-1.0
    max_tokens=-100   # Must be positive
)

# ✅ Valid values
ai = await get_provider("gemini",
    temperature=0.7,  # 0.0-1.0
    max_tokens=1000,  # Positive integer
    top_k=40,         # Positive integer
    top_p=0.9         # 0.0-1.0
)
```

### **Problem: Conflicting Configuration**
```python
# Configuration conflicts
ai = await get_provider("gemini", 
    stream=True,
    config={"stream": False}  # Conflict!
)
```

#### **Solution: Consistent Configuration**
```python
# ✅ Use kwargs OR config, not both
ai = await get_provider("gemini", stream=True, temperature=0.7)

# OR
config = {"stream": True, "temperature": 0.7}
ai = await get_provider("gemini", config=config)
```

---

## 🐛 **Common Runtime Errors**

### **Problem: Context Length Exceeded**
```
Error: This model's maximum context length is 4096 tokens
```

#### **Solution: Manage Context Size**
```python
# Use memory with context management
from llmblocks.blocks.memory import MemoryConfig

config = MemoryConfig(
    context_strategy="summarized",  # Summarize old messages
    max_messages=50,               # Limit message count
    summary_threshold=30           # Summarize when > 30 messages
)

ai = await get_stateful_ai("gemini", memory_config=config)
```

### **Problem: Timeout Errors**
```
TimeoutError: Request timed out after 30 seconds
```

#### **Solution: Increase Timeout**
```python
# Increase timeout for long requests
ai = await get_provider("gemini", timeout=120)  # 2 minutes

# Or handle timeouts gracefully
import asyncio

try:
    response = await asyncio.wait_for(
        ai.generate("Long complex task"),
        timeout=60
    )
except asyncio.TimeoutError:
    print("Request timed out, trying shorter prompt")
```

### **Problem: JSON Decode Errors**
```
JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

#### **Solution: Handle API Response Errors**
```python
from llmblocks.blocks.llm_provider.base import LLMProviderError

try:
    response = await ai.generate("Hello")
except LLMProviderError as e:
    print(f"Provider error: {e}")
    # Implement fallback logic
```

---

## 🔍 **Debugging Techniques**

### **Enable Debug Logging**
```python
import os
import logging

# Enable LLMBlocks logging
os.environ["LLMBLOCKS_ENABLE_LOGGING"] = "true"
os.environ["LLMBLOCKS_LOG_FORMAT"] = "readable"

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
```

### **Check Provider Health**
```python
# Health check
health = await ai.health_check()
print(f"Provider: {health['provider']}")
print(f"Status: {health['status']}")
print(f"Last error: {health.get('last_error', 'None')}")
```

### **Inspect Provider Metadata**
```python
# Check provider metadata
metadata = ai.metadata
print(f"Total requests: {metadata.get('total_requests', 0)}")
print(f"Failed requests: {metadata.get('failed_requests', 0)}")
print(f"Average response time: {metadata.get('avg_response_time', 0)}")
```

### **Memory Diagnostics**
```python
# Memory statistics
if hasattr(ai, 'memory'):
    stats = await ai.memory.get_stats()
    print(f"Memory stats: {stats}")
```

---

## 🧪 **Testing & Validation**

### **Test API Connectivity**
```python
async def test_connectivity():
    providers = ["gemini", "openai", "anthropic"]
    
    for provider_name in providers:
        try:
            ai = await get_provider(provider_name)
            response = await ai.generate("Test")
            print(f"✅ {provider_name}: OK")
        except Exception as e:
            print(f"❌ {provider_name}: {e}")
```

### **Validate Configuration**
```python
from llmblocks.blocks.llm_provider import GeminiConfig

try:
    config = GeminiConfig(
        model="gemini-2.0-flash",
        temperature=0.7,
        max_tokens=1000
    )
    print("✅ Configuration valid")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

### **Test Memory Persistence**
```python
async def test_memory_persistence():
    # Create AI with file memory
    ai1 = await get_persistent_ai("gemini", session_id="test")
    await ai1.chat("Remember: I like coffee")
    
    # Create new instance with same session
    ai2 = await get_persistent_ai("gemini", session_id="test")
    response = await ai2.chat("What do I like?")
    
    if "coffee" in response.content.lower():
        print("✅ Memory persistence working")
    else:
        print("❌ Memory persistence failed")
```

---

## 📊 **Performance Issues**

### **Problem: Slow Response Times**
```python
# Responses are taking too long
start_time = time.time()
response = await ai.generate("Hello")
print(f"Response time: {time.time() - start_time:.2f}s")  # > 10 seconds
```

#### **Solution: Optimize Configuration**
```python
# Use faster models
ai = await get_provider("gemini", model="gemini-2.0-flash")  # Faster

# Reduce max_tokens
ai = await get_provider("gemini", max_tokens=500)  # Shorter responses

# Use streaming for perceived speed
ai = await get_provider("gemini", stream=True)
```

### **Problem: High Memory Usage**
```python
# Memory usage keeps growing
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

#### **Solution: Memory Management**
```python
# Limit conversation memory
config = MemoryConfig(
    max_messages=50,           # Limit messages
    context_strategy="recent"  # Keep only recent
)

# Clear memory periodically
if len(await ai.get_conversation_history()) > 100:
    await ai.clear_conversation()
```

---

## 🚨 **Emergency Fixes**

### **Quick Reset**
```python
# Reset everything and start fresh
import os
import shutil

# Clear environment
for key in list(os.environ.keys()):
    if key.startswith("LLMBLOCKS_"):
        del os.environ[key]

# Remove memory files
if os.path.exists("./conversations"):
    shutil.rmtree("./conversations")

# Reinstall package
# pip uninstall llmblocks -y
# pip install llmblocks
```

### **Fallback Provider**
```python
async def get_working_ai():
    """Get any working AI provider."""
    providers = ["gemini", "openai", "anthropic"]
    
    for provider in providers:
        try:
            ai = await get_provider(provider)
            await ai.generate("test")  # Test it works
            return ai
        except:
            continue
    
    raise Exception("No working providers available")

# Usage
ai = await get_working_ai()
```

---

## 📞 **Getting Help**

### **Before Asking for Help**
1. ✅ Check this troubleshooting guide
2. ✅ Verify API keys are set correctly
3. ✅ Ensure you're using supported Python version (3.11+)
4. ✅ Try the minimal examples first
5. ✅ Enable debug logging

### **Where to Get Help**
- **[GitHub Issues](https://github.com/paritosh0707/llmblocks/issues)** - Bug reports and feature requests
- **[GitHub Discussions](https://github.com/paritosh0707/llmblocks/discussions)** - Questions and community help
- **[[API Reference]]** - Complete API documentation
- **[[Minimal Examples]]** - Working code examples

### **What to Include in Bug Reports**
```python
# Include this information:
import sys
import llmblocks

print(f"Python version: {sys.version}")
print(f"LLMBlocks version: {llmblocks.__version__}")
print(f"Operating system: {sys.platform}")

# Include error traceback
# Include minimal code to reproduce the issue
# Include configuration (without API keys!)
```

---

## 🎯 **Prevention Tips**

### **Best Practices**
- Always use virtual environments
- Set API keys in environment variables, not code
- Use try/except blocks for API calls
- Implement retry logic for rate limits
- Monitor memory usage in long-running applications
- Use appropriate context strategies for memory
- Test with minimal examples first

### **Code Review Checklist**
- [ ] All async functions are awaited
- [ ] API keys are not hardcoded
- [ ] Error handling is implemented
- [ ] Memory limits are configured
- [ ] Timeouts are set appropriately
- [ ] Provider names are correct
- [ ] Configuration values are valid

---

**Most issues can be resolved quickly with the right approach. Don't hesitate to ask for help! 🚀✨**
