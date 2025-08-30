# ⚙️ Configuration Guide

Complete guide to configuring LLMBlocks for different environments, providers, and use cases.

---

## 🎯 **Configuration Overview**

LLMBlocks uses a hierarchical configuration system:
1. **Environment Variables** (highest priority)
2. **Configuration Files** (.env, config.yaml)
3. **Direct Parameters** (in code)
4. **Default Values** (lowest priority)

---

## 🔑 **API Keys Configuration**

### **Environment Variables (Recommended)**
```bash
# Core providers
export GOOGLE_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

### **.env File Configuration**
```bash
# Create .env file in your project root
cat > .env << EOF
# LLM Provider API Keys
GOOGLE_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# LLMBlocks Settings
LLMBLOCKS_ENABLE_LOGGING=true
LLMBLOCKS_LOG_FORMAT=readable
LLMBLOCKS_DEFAULT_PROVIDER=gemini
EOF
```

### **Loading .env Files**
```python
# Automatic loading (recommended)
from dotenv import load_dotenv
load_dotenv()

# Manual loading with path
load_dotenv(".env.production")

# Override existing variables
load_dotenv(override=True)
```

---

## 🤖 **Provider Configuration**

### **Basic Provider Configuration**
```python
# Simple configuration
ai = await get_provider("gemini", 
    temperature=0.7,
    max_tokens=1000
)

# Configuration dictionary
config = {
    "model": "gemini-2.0-flash",
    "temperature": 0.7,
    "max_tokens": 1000,
    "stream": True
}
ai = await get_provider("gemini", config=config)
```

### **Advanced Provider Configuration**
```python
from llmblocks.blocks.llm_provider import GeminiConfig, GeminiProvider

# Using configuration classes
config = GeminiConfig(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=1000,
    top_k=40,
    top_p=0.9,
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE"
    },
    requests_per_minute=60,
    tokens_per_minute=90000,
    timeout=30
)

provider = GeminiProvider(config)
await provider.initialize()
```

---

## 🔧 **Provider-Specific Configuration**

### **Gemini Configuration**
```python
gemini_config = {
    # Model selection
    "model": "gemini-2.0-flash",  # or "gemini-1.5-pro", "gemini-1.5-flash"
    
    # Generation parameters
    "temperature": 0.7,           # 0.0-1.0, creativity level
    "max_tokens": 1000,           # Maximum response length
    "top_k": 40,                  # Top-k sampling
    "top_p": 0.9,                 # Top-p sampling
    
    # Safety settings
    "safety_settings": {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE"
    },
    
    # Rate limiting
    "requests_per_minute": 60,
    "tokens_per_minute": 90000,
    
    # Network settings
    "timeout": 30,
    "max_retries": 3,
    "retry_delay": 1.0
}

ai = await get_provider("gemini", **gemini_config)
```

### **OpenAI Configuration**
```python
openai_config = {
    # Model selection
    "model": "gpt-4o",            # or "gpt-3.5-turbo", "gpt-4o-mini"
    
    # Generation parameters
    "temperature": 0.7,           # 0.0-2.0, creativity level
    "max_tokens": 1000,           # Maximum response length
    "top_p": 0.9,                 # Top-p sampling
    "presence_penalty": 0.0,      # -2.0 to 2.0, penalize new topics
    "frequency_penalty": 0.0,     # -2.0 to 2.0, penalize repetition
    
    # Reproducibility
    "seed": 42,                   # For consistent outputs
    
    # Function calling
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    
    # Rate limiting
    "requests_per_minute": 500,
    "tokens_per_minute": 150000,
    
    # Network settings
    "timeout": 30,
    "max_retries": 3
}

ai = await get_provider("openai", **openai_config)
```

### **Anthropic Configuration**
```python
anthropic_config = {
    # Model selection
    "model": "claude-3-5-sonnet-20241022",  # or "claude-3-haiku-20240307"
    
    # Generation parameters
    "temperature": 0.7,           # 0.0-1.0, creativity level
    "max_tokens": 2000,           # Maximum response length
    "top_p": 0.9,                 # Top-p sampling
    "top_k": 250,                 # Top-k sampling
    
    # Stop sequences
    "stop_sequences": ["Human:", "Assistant:"],
    
    # Rate limiting
    "requests_per_minute": 50,
    "tokens_per_minute": 100000,
    
    # Network settings
    "timeout": 60,
    "max_retries": 3
}

ai = await get_provider("anthropic", **anthropic_config)
```

### **Azure OpenAI Configuration**
```python
azure_config = {
    # Azure-specific settings
    "deployment_name": "gpt-4-deployment",
    "api_version": "2024-02-15-preview",
    "azure_endpoint": "https://your-resource.openai.azure.com/",
    
    # Generation parameters (same as OpenAI)
    "temperature": 0.7,
    "max_tokens": 1000,
    
    # Network settings
    "timeout": 30,
    "max_retries": 3
}

ai = await get_provider("azure_openai", **azure_config)
```

---

## 🧠 **Memory Configuration**

### **Memory Backend Configuration**
```python
from llmblocks.blocks.memory import MemoryConfig

# In-memory backend (default)
memory_config = MemoryConfig(
    backend="in_memory",
    max_messages=100
)

# File backend
memory_config = MemoryConfig(
    backend="file",
    file_path="./conversations/user_123.json",
    max_messages=200,
    auto_save=True,
    backup_count=5,
    compression=True
)

# Redis backend
memory_config = MemoryConfig(
    backend="redis",
    redis_url="redis://localhost:6379",
    redis_key_prefix="llmblocks:",
    redis_db=0,
    ttl=86400,  # 24 hours
    max_messages=500
)
```

### **Context Management Configuration**
```python
# Recent messages strategy
memory_config = MemoryConfig(
    context_strategy="recent",
    max_messages=50  # Keep last 50 messages
)

# Summarized strategy
memory_config = MemoryConfig(
    context_strategy="summarized",
    max_messages=100,
    summary_threshold=60,    # Summarize when > 60 messages
    preserve_recent=20,      # Keep last 20 messages as-is
    summary_model="gemini"   # Model for summarization
)

# All messages strategy
memory_config = MemoryConfig(
    context_strategy="all",
    max_messages=None  # No limit (use carefully)
)
```

---

## 🌍 **Environment-Specific Configuration**

### **Development Configuration**
```python
# development.env
LLMBLOCKS_ENABLE_LOGGING=true
LLMBLOCKS_LOG_FORMAT=readable
LLMBLOCKS_DEFAULT_PROVIDER=gemini
GOOGLE_API_KEY=your-dev-api-key

# Relaxed rate limits for development
LLMBLOCKS_DEFAULT_REQUESTS_PER_MINUTE=100
LLMBLOCKS_DEFAULT_TIMEOUT=60
```

### **Production Configuration**
```python
# production.env
LLMBLOCKS_ENABLE_LOGGING=true
LLMBLOCKS_LOG_FORMAT=json
LLMBLOCKS_DEFAULT_PROVIDER=openai
OPENAI_API_KEY=your-prod-api-key

# Strict rate limits for production
LLMBLOCKS_DEFAULT_REQUESTS_PER_MINUTE=500
LLMBLOCKS_DEFAULT_TIMEOUT=30
LLMBLOCKS_ENABLE_METRICS=true
```

### **Testing Configuration**
```python
# testing.env
LLMBLOCKS_ENABLE_LOGGING=false
LLMBLOCKS_USE_MOCK_PROVIDERS=true
LLMBLOCKS_DEFAULT_PROVIDER=mock

# Mock responses for testing
LLMBLOCKS_MOCK_RESPONSE="This is a mock response"
```

---

## 📊 **Logging Configuration**

### **Basic Logging**
```python
import os

# Enable logging
os.environ["LLMBLOCKS_ENABLE_LOGGING"] = "true"

# Set log format
os.environ["LLMBLOCKS_LOG_FORMAT"] = "readable"  # or "json"

# Set log level
os.environ["LLMBLOCKS_LOG_LEVEL"] = "INFO"  # DEBUG, INFO, WARNING, ERROR
```

### **Advanced Logging**
```python
import logging
from llmblocks.core.logger import get_logger

# Configure Python logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llmblocks.log'),
        logging.StreamHandler()
    ]
)

# Get LLMBlocks logger
logger = get_logger("my_app")
logger.info("Application started")
```

### **Structured Logging**
```python
import os
import json

# Enable JSON logging
os.environ["LLMBLOCKS_LOG_FORMAT"] = "json"

# Custom log processor
def log_processor(logger, method_name, event_dict):
    event_dict["app"] = "my_llm_app"
    event_dict["version"] = "1.0.0"
    return event_dict

# Configure structured logging
import structlog
structlog.configure(
    processors=[
        log_processor,
        structlog.processors.JSONRenderer()
    ]
)
```

---

## 🔄 **Configuration Files**

### **YAML Configuration**
```yaml
# config.yaml
llmblocks:
  default_provider: gemini
  enable_logging: true
  log_format: readable
  
providers:
  gemini:
    model: gemini-2.0-flash
    temperature: 0.7
    max_tokens: 1000
    requests_per_minute: 60
    
  openai:
    model: gpt-4o
    temperature: 0.5
    max_tokens: 500
    requests_per_minute: 500
    
memory:
  default_backend: file
  file_path: ./conversations
  max_messages: 100
  context_strategy: summarized
```

### **Loading YAML Configuration**
```python
import yaml
from llmblocks.blocks.llm_provider import get_provider

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Use configuration
provider_config = config["providers"]["gemini"]
ai = await get_provider("gemini", **provider_config)
```

### **JSON Configuration**
```json
{
  "llmblocks": {
    "default_provider": "gemini",
    "enable_logging": true
  },
  "providers": {
    "gemini": {
      "model": "gemini-2.0-flash",
      "temperature": 0.7,
      "max_tokens": 1000
    }
  },
  "memory": {
    "backend": "redis",
    "redis_url": "redis://localhost:6379",
    "max_messages": 200
  }
}
```

---

## 🚀 **Dynamic Configuration**

### **Runtime Configuration Changes**
```python
# Change configuration at runtime
ai = await get_provider("gemini", temperature=0.7)

# Update configuration
ai.provider_config.temperature = 0.9
ai.provider_config.max_tokens = 500

# Or create new instance with new config
ai_creative = await get_provider("gemini", temperature=0.9)
```

### **Configuration Validation**
```python
from llmblocks.blocks.llm_provider import GeminiConfig
from pydantic import ValidationError

try:
    config = GeminiConfig(
        model="gemini-2.0-flash",
        temperature=0.7,
        max_tokens=1000
    )
    print("✅ Configuration valid")
except ValidationError as e:
    print(f"❌ Configuration error: {e}")
```

### **Configuration Inheritance**
```python
# Base configuration
base_config = {
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30
}

# Specific configurations inheriting from base
creative_config = {**base_config, "temperature": 0.9}
conservative_config = {**base_config, "temperature": 0.1}
long_form_config = {**base_config, "max_tokens": 2000}
```

---

## 🔒 **Security Configuration**

### **API Key Security**
```python
# ❌ Never hardcode API keys
ai = await get_provider("gemini", api_key="hardcoded-key")

# ✅ Use environment variables
ai = await get_provider("gemini")  # Reads from GOOGLE_API_KEY

# ✅ Use secure key management
import keyring
api_key = keyring.get_password("llmblocks", "gemini")
ai = await get_provider("gemini", api_key=api_key)
```

### **Rate Limiting for Security**
```python
# Conservative rate limits
security_config = {
    "requests_per_minute": 30,    # Lower than API limits
    "tokens_per_minute": 50000,   # Conservative token usage
    "timeout": 15,                # Shorter timeout
    "max_retries": 2              # Fewer retries
}

ai = await get_provider("gemini", **security_config)
```

### **Content Filtering**
```python
# Strict safety settings for Gemini
safety_config = {
    "safety_settings": {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_LOW_AND_ABOVE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_LOW_AND_ABOVE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_LOW_AND_ABOVE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_LOW_AND_ABOVE"
    }
}

ai = await get_provider("gemini", **safety_config)
```

---

## 📈 **Performance Configuration**

### **High-Performance Configuration**
```python
# Optimized for speed
performance_config = {
    "model": "gemini-2.0-flash",  # Fastest model
    "max_tokens": 500,            # Shorter responses
    "temperature": 0.3,           # More deterministic
    "stream": True,               # Perceived speed
    "timeout": 10,                # Quick timeout
    "requests_per_minute": 100    # Higher rate limit
}

ai = await get_provider("gemini", **performance_config)
```

### **Batch Processing Configuration**
```python
import asyncio

# Configure for batch processing
batch_config = {
    "requests_per_minute": 1000,  # High rate limit
    "timeout": 60,                # Longer timeout
    "max_retries": 5              # More retries
}

ai = await get_provider("gemini", **batch_config)

# Process multiple requests concurrently
prompts = ["Hello", "Goodbye", "Thank you"]
tasks = [ai.generate(prompt) for prompt in prompts]
responses = await asyncio.gather(*tasks)
```

---

## 🧪 **Testing Configuration**

### **Mock Configuration for Testing**
```python
# Mock provider for testing
class MockProvider:
    async def generate(self, prompt):
        return MockResponse("Mock response")

# Use in tests
import pytest

@pytest.fixture
async def mock_ai():
    return MockProvider()

async def test_my_function(mock_ai):
    response = await mock_ai.generate("test")
    assert response.content == "Mock response"
```

### **Test Environment Configuration**
```python
# test.env
LLMBLOCKS_ENABLE_LOGGING=false
LLMBLOCKS_USE_MOCK_RESPONSES=true
LLMBLOCKS_MOCK_DELAY=0.1
GOOGLE_API_KEY=test-key-not-real
```

---

## 📋 **Configuration Best Practices**

### **1. Environment Separation**
```bash
# Use different .env files for different environments
.env.development
.env.staging
.env.production
.env.testing
```

### **2. Configuration Validation**
```python
# Always validate configuration
from pydantic import ValidationError

try:
    config = GeminiConfig(**user_config)
except ValidationError as e:
    logger.error(f"Invalid configuration: {e}")
    raise
```

### **3. Secure Defaults**
```python
# Use secure defaults
default_config = {
    "temperature": 0.7,           # Balanced creativity
    "max_tokens": 1000,           # Reasonable limit
    "timeout": 30,                # Reasonable timeout
    "requests_per_minute": 60,    # Conservative rate limit
    "stream": False               # Explicit streaming
}
```

### **4. Configuration Documentation**
```python
# Document your configuration
PROVIDER_CONFIG = {
    "model": "gemini-2.0-flash",     # Fast, cost-effective model
    "temperature": 0.7,              # Balanced creativity (0.0-1.0)
    "max_tokens": 1000,              # Reasonable response length
    "requests_per_minute": 60,       # Within free tier limits
    "timeout": 30                    # 30 second timeout
}
```

---

## 🎯 **Configuration Examples**

### **Chatbot Configuration**
```python
chatbot_config = {
    "model": "gemini-2.0-flash",
    "temperature": 0.8,           # Creative responses
    "max_tokens": 500,            # Conversational length
    "stream": True,               # Real-time feel
    "memory_config": MemoryConfig(
        backend="redis",
        context_strategy="recent",
        max_messages=50
    )
}

ai = await get_stateful_ai("gemini", **chatbot_config)
```

### **Content Generation Configuration**
```python
content_config = {
    "model": "gpt-4o",
    "temperature": 0.9,           # High creativity
    "max_tokens": 2000,           # Long-form content
    "presence_penalty": 0.1,      # Encourage new topics
    "frequency_penalty": 0.1      # Reduce repetition
}

ai = await get_provider("openai", **content_config)
```

### **Code Assistant Configuration**
```python
code_config = {
    "model": "claude-3-5-sonnet-20241022",
    "temperature": 0.2,           # Precise, deterministic
    "max_tokens": 1500,           # Code + explanation
    "stop_sequences": ["```"]     # Stop at code blocks
}

ai = await get_provider("anthropic", **code_config)
```

---

**Master LLMBlocks configuration for any use case! ⚙️✨**
