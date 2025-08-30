# 🛠️ Installation Guide

Complete installation guide for LLMBlocks across different environments and use cases.

---

## 📋 **System Requirements**

### **Minimum Requirements**
- **Python**: 3.11 or higher
- **Memory**: 512MB RAM minimum
- **Storage**: 100MB free space
- **Network**: Internet connection for API calls

### **Recommended Requirements**
- **Python**: 3.12+ (latest stable)
- **Memory**: 2GB+ RAM for optimal performance
- **Storage**: 1GB+ for development with examples
- **OS**: Linux, macOS, or Windows

---

## 🚀 **Installation Methods**

### **Method 1: PyPI (Recommended)**

```bash
# Install latest stable version
pip install llmblocks

# Install with all optional dependencies
pip install llmblocks[all]

# Install specific extras
pip install llmblocks[redis,dev]
```

### **Method 2: UV (Fastest)**

```bash
# Install with uv (fastest Python package manager)
uv add llmblocks

# Install with extras
uv add llmblocks[all]
```

### **Method 3: From Source (Development)**

```bash
# Clone repository
git clone https://github.com/paritosh0707/llmblocks.git
cd llmblocks

# Install in development mode
pip install -e .

# Or with uv
uv sync
```

### **Method 4: Docker (Production)**

```bash
# Pull official image
docker pull llmblocks/llmblocks:latest

# Or build from source
git clone https://github.com/paritosh0707/llmblocks.git
cd llmblocks
docker build -t llmblocks .
```

---

## 📦 **Optional Dependencies**

LLMBlocks supports optional dependencies for enhanced functionality:

### **Available Extras**

| Extra | Description | Install Command |
|-------|-------------|-----------------|
| `redis` | Redis memory backend | `pip install llmblocks[redis]` |
| `dev` | Development tools | `pip install llmblocks[dev]` |
| `test` | Testing dependencies | `pip install llmblocks[test]` |
| `docs` | Documentation tools | `pip install llmblocks[docs]` |
| `all` | All optional dependencies | `pip install llmblocks[all]` |

### **Individual Packages**

```bash
# Redis support
pip install redis

# Development tools
pip install pytest black ruff mypy

# Documentation
pip install mkdocs mkdocs-material
```

---

## 🔑 **API Keys Setup**

### **Supported Providers**

| Provider | API Key Variable | Free Tier | Documentation |
|----------|------------------|-----------|---------------|
| **Gemini** | `GOOGLE_API_KEY` | ✅ Yes | [Get API Key](https://makersuite.google.com/app/apikey) |
| **OpenAI** | `OPENAI_API_KEY` | 💰 Paid | [Get API Key](https://platform.openai.com/api-keys) |
| **Anthropic** | `ANTHROPIC_API_KEY` | 💰 Paid | [Get API Key](https://console.anthropic.com/) |

### **Environment Variables**

#### **Option 1: Export (Temporary)**
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

#### **Option 2: .env File (Recommended)**
```bash
# Create .env file in your project root
cat > .env << EOF
GOOGLE_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
EOF
```

#### **Option 3: Shell Profile (Permanent)**
```bash
# Add to ~/.bashrc, ~/.zshrc, or ~/.profile
echo 'export GOOGLE_API_KEY="your-gemini-api-key"' >> ~/.bashrc
source ~/.bashrc
```

---

## 🐳 **Docker Installation**

### **Quick Start with Docker**

```bash
# Run with environment variables
docker run -e GOOGLE_API_KEY="your-key" llmblocks/llmblocks:latest

# Run with .env file
docker run --env-file .env llmblocks/llmblocks:latest

# Interactive development
docker run -it -v $(pwd):/app llmblocks/llmblocks:latest bash
```

### **Docker Compose (Development)**

```yaml
# docker-compose.yml
version: '3.8'
services:
  llmblocks:
    image: llmblocks/llmblocks:latest
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
    volumes:
      - .:/app
    working_dir: /app
    
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

```bash
# Start services
docker-compose up -d
```

---

## 🌍 **Virtual Environment Setup**

### **Using venv (Built-in)**

```bash
# Create virtual environment
python -m venv llmblocks-env

# Activate (Linux/macOS)
source llmblocks-env/bin/activate

# Activate (Windows)
llmblocks-env\Scripts\activate

# Install LLMBlocks
pip install llmblocks
```

### **Using conda**

```bash
# Create conda environment
conda create -n llmblocks python=3.12
conda activate llmblocks

# Install LLMBlocks
pip install llmblocks
```

### **Using pyenv**

```bash
# Install Python version
pyenv install 3.12.0
pyenv local 3.12.0

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install LLMBlocks
pip install llmblocks
```

---

## ✅ **Verify Installation**

### **Quick Test**

```python
# test_installation.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def test():
    try:
        # Test basic import
        print("✅ LLMBlocks imported successfully")
        
        # Test provider creation (requires API key)
        if os.getenv("GOOGLE_API_KEY"):
            ai = await get_provider("gemini")
            print("✅ Gemini provider created successfully")
        else:
            print("⚠️  No API key found, skipping provider test")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import os
    asyncio.run(test())
```

```bash
python test_installation.py
```

### **Run Example**

```bash
# Download and run example
curl -O https://raw.githubusercontent.com/paritosh0707/llmblocks/main/examples/minimal/01_hello_world.py
python 01_hello_world.py
```

---

## 🔧 **Development Setup**

### **Full Development Environment**

```bash
# Clone repository
git clone https://github.com/paritosh0707/llmblocks.git
cd llmblocks

# Install with development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
ruff check src/
black src/
mypy src/
```

### **Using UV (Recommended for Development)**

```bash
# Clone and setup with uv
git clone https://github.com/paritosh0707/llmblocks.git
cd llmblocks

# Install everything
uv sync

# Activate virtual environment
source .venv/bin/activate

# Run tests
uv run pytest
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Python Version Error**
```
ERROR: Python 3.10 is not supported. Please use Python 3.11+
```
**Solution**: Upgrade Python or use pyenv/conda to install Python 3.11+

#### **API Key Not Found**
```
Error: No API key found for provider 'gemini'
```
**Solution**: Set environment variable or create .env file

#### **Import Error**
```
ModuleNotFoundError: No module named 'llmblocks'
```
**Solution**: Ensure virtual environment is activated and package is installed

#### **Permission Error (macOS/Linux)**
```
Permission denied: '/usr/local/lib/python3.x/site-packages/'
```
**Solution**: Use virtual environment or `pip install --user`

### **Getting Help**

- **[[Troubleshooting]]** - Detailed troubleshooting guide
- **[GitHub Issues](https://github.com/paritosh0707/llmblocks/issues)** - Report installation issues
- **[Discussions](https://github.com/paritosh0707/llmblocks/discussions)** - Community help

---

## 🎯 **Next Steps**

After successful installation:

1. **[[Quick Start]]** - Build your first AI in 5 minutes
2. **[[Configuration]]** - Configure providers and settings
3. **[[Your First AI]]** - Detailed first application guide
4. **[[Minimal Examples]]** - Explore example applications

---

## 📊 **Installation Verification Checklist**

- [ ] Python 3.11+ installed
- [ ] LLMBlocks package installed
- [ ] API keys configured
- [ ] Virtual environment activated
- [ ] Test script runs successfully
- [ ] Ready to build AI applications!

**Happy coding! 🚀✨**
