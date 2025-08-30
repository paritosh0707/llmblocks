# 🤖 LLM Providers Guide

Complete guide to all supported LLM providers in LLMBlocks, their features, configuration, and usage patterns.

---

## 🌟 **Supported Providers**

| Provider | Status | Free Tier | Streaming | Memory | Best For |
|----------|--------|-----------|-----------|---------|----------|
| **Gemini** | ✅ Full | ✅ Yes | ✅ Yes | ✅ Yes | General purpose, free development |
| **OpenAI** | ✅ Full | 💰 Paid | ✅ Yes | ✅ Yes | Production applications |
| **Anthropic** | ✅ Full | 💰 Paid | ✅ Yes | ✅ Yes | Complex reasoning, safety |
| **Azure OpenAI** | ✅ Full | 💰 Paid | ✅ Yes | ✅ Yes | Enterprise deployments |

---

## 🚀 **Quick Provider Usage**

### **Basic Usage Pattern**
```python
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    # Get any provider
    ai = await get_provider("provider_name", **config)
    
    # Generate response
    response = await ai.generate("Your prompt here")
    print(response.content)

asyncio.run(main())
```

---

## 🔷 **Gemini (Google)**

### **Overview**
- **Best For**: Development, prototyping, free tier usage
- **Models**: `gemini-2.0-flash`, `gemini-1.5-pro`, `gemini-1.5-flash`
- **Free Tier**: 15 requests/minute, 1M tokens/minute
- **Strengths**: Fast, multimodal, good reasoning

### **Setup**
```bash
# Get API key from: https://makersuite.google.com/app/apikey
export GOOGLE_API_KEY="your-gemini-api-key"
```

### **Basic Usage**
```python
# Simple usage
ai = await get_provider("gemini")
response = await ai.generate("Explain quantum computing")

# With configuration
ai = await get_provider("gemini", 
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=1000
)
```

### **Advanced Configuration**
```python
from llmblocks.blocks.llm_provider import GeminiProvider, GeminiConfig

config = GeminiConfig(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=1000,
    top_p=0.9,
    top_k=40,
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE"
    }
)

provider = GeminiProvider(config)
await provider.initialize()
```

### **Streaming Example**
```python
ai = await get_provider("gemini", stream=True)

print("AI: ", end="", flush=True)
async for chunk in ai.generate_stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
print()
```

---

## 🔶 **OpenAI**

### **Overview**
- **Best For**: Production applications, advanced reasoning
- **Models**: `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **Pricing**: Pay-per-token (starts ~$0.002/1K tokens)
- **Strengths**: Excellent reasoning, large context, reliable

### **Setup**
```bash
# Get API key from: https://platform.openai.com/api-keys
export OPENAI_API_KEY="your-openai-api-key"
```

### **Basic Usage**
```python
# Simple usage
ai = await get_provider("openai")
response = await ai.generate("Write a Python function to sort a list")

# With specific model
ai = await get_provider("openai", 
    model="gpt-4o",
    temperature=0.3,
    max_tokens=500
)
```

### **Advanced Configuration**
```python
from llmblocks.blocks.llm_provider import OpenAIProvider, OpenAIConfig

config = OpenAIConfig(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    presence_penalty=0.1,
    frequency_penalty=0.1,
    top_p=0.9,
    seed=42  # For reproducible outputs
)

provider = OpenAIProvider(config)
await provider.initialize()
```

### **Function Calling**
```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

ai = await get_provider("openai", tools=tools)
response = await ai.generate("What's the weather in San Francisco?")
```

---

## 🔸 **Anthropic (Claude)**

### **Overview**
- **Best For**: Complex reasoning, safety-critical applications
- **Models**: `claude-3-5-sonnet-20241022`, `claude-3-haiku-20240307`
- **Pricing**: Pay-per-token (competitive with OpenAI)
- **Strengths**: Excellent reasoning, safety, long context

### **Setup**
```bash
# Get API key from: https://console.anthropic.com/
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### **Basic Usage**
```python
# Simple usage
ai = await get_provider("anthropic")
response = await ai.generate("Analyze this business proposal...")

# With configuration
ai = await get_provider("anthropic",
    model="claude-3-5-sonnet-20241022",
    temperature=0.5,
    max_tokens=2000
)
```

### **Advanced Configuration**
```python
from llmblocks.blocks.llm_provider import AnthropicProvider, AnthropicConfig

config = AnthropicConfig(
    model="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_tokens=2000,
    top_p=0.9,
    top_k=250,
    stop_sequences=["Human:", "Assistant:"]
)

provider = AnthropicProvider(config)
await provider.initialize()
```

---

## 🔷 **Azure OpenAI**

### **Overview**
- **Best For**: Enterprise deployments, compliance requirements
- **Models**: Same as OpenAI (gpt-4o, gpt-3.5-turbo, etc.)
- **Pricing**: Azure pricing model
- **Strengths**: Enterprise features, compliance, SLA

### **Setup**
```bash
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

### **Basic Usage**
```python
ai = await get_provider("azure_openai",
    deployment_name="your-gpt4-deployment",
    api_version="2024-02-15-preview"
)

response = await ai.generate("Explain machine learning")
```

### **Advanced Configuration**
```python
from llmblocks.blocks.llm_provider import AzureOpenAIProvider, AzureOpenAIConfig

config = AzureOpenAIConfig(
    deployment_name="your-gpt4-deployment",
    api_version="2024-02-15-preview",
    temperature=0.7,
    max_tokens=1000,
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-azure-key"
)

provider = AzureOpenAIProvider(config)
await provider.initialize()
```

---

## ⚙️ **Provider Configuration**

### **Common Configuration Options**

| Parameter | Description | Default | All Providers |
|-----------|-------------|---------|---------------|
| `model` | Model name | Provider default | ✅ |
| `temperature` | Randomness (0-1) | 0.7 | ✅ |
| `max_tokens` | Max response length | 1000 | ✅ |
| `stream` | Enable streaming | False | ✅ |
| `timeout` | Request timeout | 30s | ✅ |

### **Provider-Specific Options**

#### **Gemini Only**
```python
config = {
    "top_k": 40,
    "top_p": 0.9,
    "safety_settings": {...}
}
```

#### **OpenAI Only**
```python
config = {
    "presence_penalty": 0.1,
    "frequency_penalty": 0.1,
    "seed": 42,
    "tools": [...]
}
```

#### **Anthropic Only**
```python
config = {
    "top_k": 250,
    "stop_sequences": ["Human:"]
}
```

---

## 🔄 **Provider Switching**

### **Runtime Provider Switching**
```python
# Switch providers dynamically
providers = ["gemini", "openai", "anthropic"]

for provider_name in providers:
    try:
        ai = await get_provider(provider_name)
        response = await ai.generate("Hello!")
        print(f"{provider_name}: {response.content}")
        break
    except Exception as e:
        print(f"{provider_name} failed: {e}")
        continue
```

### **Fallback Pattern**
```python
async def get_ai_with_fallback():
    """Get AI provider with automatic fallback."""
    fallback_order = ["gemini", "openai", "anthropic"]
    
    for provider in fallback_order:
        try:
            return await get_provider(provider)
        except Exception:
            continue
    
    raise Exception("No providers available")

# Usage
ai = await get_ai_with_fallback()
```

---

## 📊 **Provider Comparison**

### **Performance Comparison**

| Metric | Gemini | OpenAI | Anthropic | Azure OpenAI |
|--------|--------|--------|-----------|--------------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Cost** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Reasoning** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Safety** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

### **Use Case Recommendations**

#### **🎯 For Development & Prototyping**
```python
# Gemini - Free tier, fast development
ai = await get_provider("gemini")
```

#### **🏢 For Production Applications**
```python
# OpenAI - Reliable, well-tested
ai = await get_provider("openai", model="gpt-4o")
```

#### **🔒 For Safety-Critical Applications**
```python
# Anthropic - Enhanced safety features
ai = await get_provider("anthropic", model="claude-3-5-sonnet-20241022")
```

#### **🏢 For Enterprise Deployments**
```python
# Azure OpenAI - Enterprise features
ai = await get_provider("azure_openai", deployment_name="gpt4-prod")
```

---

## 🛠️ **Custom Provider Development**

### **Creating a Custom Provider**
```python
from llmblocks.blocks.llm_provider.base import BaseLLMProvider, LLMProviderConfig

class MyCustomConfig(LLMProviderConfig):
    provider_name: str = "mycustom"
    api_endpoint: str
    custom_param: str = "default"

class MyCustomProvider(BaseLLMProvider):
    def __init__(self, config: MyCustomConfig):
        super().__init__(config)
    
    async def _initialize_clients(self):
        # Initialize your custom client
        pass
    
    async def _generate_impl(self, messages, **kwargs):
        # Implement generation logic
        pass
    
    async def _generate_stream_impl(self, messages, **kwargs):
        # Implement streaming logic
        pass
```

### **Register Custom Provider**
```python
from llmblocks.blocks.llm_provider import get_factory

factory = get_factory()
factory.register_provider("mycustom", MyCustomProvider)

# Use your custom provider
ai = await get_provider("mycustom", api_endpoint="https://api.example.com")
```

---

## 🔧 **Advanced Features**

### **Rate Limiting**
```python
ai = await get_provider("openai",
    requests_per_minute=60,
    tokens_per_minute=90000
)
```

### **Retry Configuration**
```python
ai = await get_provider("gemini",
    max_retries=3,
    retry_delay=1.0
)
```

### **Custom Headers**
```python
ai = await get_provider("openai",
    headers={"Custom-Header": "value"}
)
```

### **Proxy Support**
```python
ai = await get_provider("openai",
    proxy="http://proxy.example.com:8080"
)
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **API Key Not Found**
```python
# Check if API key is set
import os
if not os.getenv("GOOGLE_API_KEY"):
    print("Please set GOOGLE_API_KEY environment variable")
```

#### **Rate Limit Exceeded**
```python
# Handle rate limits gracefully
try:
    response = await ai.generate("Hello")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after} seconds")
    await asyncio.sleep(e.retry_after)
```

#### **Model Not Available**
```python
# Check available models
factory = get_factory()
available_models = factory.get_available_models("openai")
print(f"Available models: {available_models}")
```

---

## 📚 **Next Steps**

- **[[Memory System]]** - Add memory to your providers
- **[[Streaming]]** - Implement real-time responses
- **[[Configuration]]** - Advanced configuration options
- **[[API Reference]]** - Complete API documentation

**Ready to build amazing AI applications! 🚀✨**
