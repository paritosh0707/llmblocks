# 🚀 Quick Start - Build Your First AI in 5 Minutes

Get up and running with LLMBlocks in just 5 minutes! This guide will take you from zero to your first AI application.

---

## 📋 **Prerequisites**

- **Python 3.11+** (Check: `python --version`)
- **API Key** for your chosen provider (Gemini, OpenAI, etc.)
- **5 minutes** of your time! ⏰

---

## 🛠️ **Step 1: Installation**

### **Option A: Using pip (Recommended)**
```bash
pip install llmblocks
```

### **Option B: Using uv (Fastest)**
```bash
uv add llmblocks
```

### **Option C: From Source**
```bash
git clone https://github.com/paritosh0707/llmblocks.git
cd llmblocks
pip install -e .
```

---

## 🔑 **Step 2: Set Up API Key**

### **For Gemini (Recommended - Free tier available)**
```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

### **For OpenAI**
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

### **Or use a .env file**
```bash
# Create .env file
echo "GOOGLE_API_KEY=your-gemini-api-key" > .env
```

---

## 🎯 **Step 3: Your First AI (2 Lines!)**

Create `hello_ai.py`:

```python
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    # 🚀 Just 2 lines for AI!
    ai = await get_provider("gemini")
    response = await ai.generate("Hello! Tell me a fun fact about space.")
    
    print(f"🤖 AI: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Run it:**
```bash
python hello_ai.py
```

**Expected Output:**
```
🤖 AI: Did you know that a day on Venus is longer than its year? Venus takes 243 Earth days to rotate once, but only 225 Earth days to orbit the Sun!
```

---

## 🧠 **Step 4: Add Memory (3 Lines!)**

Create `smart_ai.py`:

```python
import asyncio
from llmblocks.blocks.llm_provider import get_stateful_ai

async def main():
    # 🧠 3 lines for stateful AI with memory!
    ai = await get_stateful_ai("gemini")
    
    # First conversation
    response1 = await ai.chat("Hi! I'm Alice, a data scientist from NYC who loves pizza.")
    print(f"🤖 AI: {response1.content}")
    
    # AI remembers the context!
    response2 = await ai.chat("What do you know about me?")
    print(f"🤖 AI: {response2.content}")
    
    # AI still remembers!
    response3 = await ai.chat("What's my favorite food?")
    print(f"🤖 AI: {response3.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Run it:**
```bash
python smart_ai.py
```

**Expected Output:**
```
🤖 AI: Hello Alice! Nice to meet you! It's great to connect with a fellow data scientist. NYC is such a vibrant city, and pizza is definitely one of its best features!

🤖 AI: I know that you're Alice, a data scientist who lives in NYC and loves pizza! You seem like someone who appreciates both analytical work and good food.

🤖 AI: Your favorite food is pizza! You mentioned that when you introduced yourself.
```

---

## 🌊 **Step 5: Real-Time Streaming (4 Lines!)**

Create `streaming_ai.py`:

```python
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    # 🌊 4 lines for streaming AI!
    ai = await get_provider("gemini", stream=True)
    
    print("🤖 AI: ", end="", flush=True)
    async for chunk in ai.generate_stream("Tell me a short story about a robot learning to paint"):
        print(chunk.content, end="", flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
```

**Run it:**
```bash
python streaming_ai.py
```

**Expected Output:**
```
🤖 AI: Once upon a time, there was a little robot named Pixel who discovered an old paintbrush in a dusty attic. At first, Pixel's circuits couldn't understand why humans made colorful marks on canvas. But as Pixel experimented, mixing blues and yellows, something magical happened—the robot began to see beauty in imperfection, and soon created the most wonderful abstract art the world had ever seen. 🎨
```

---

## 🎊 **Congratulations! You're Now an AI Developer!**

In just 5 minutes, you've:
- ✅ Built a basic AI application (2 lines)
- ✅ Created a stateful AI with memory (3 lines)
- ✅ Implemented real-time streaming (4 lines)

---

## 🚀 **What's Next?**

### **Explore More Examples**
- **[[Minimal Examples]]** - More 2-10 line examples
- **[[Memory Examples]]** - Advanced memory usage
- **[[Advanced Examples]]** - Complex applications

### **Learn Core Concepts**
- **[[LLM Providers]]** - Different AI providers
- **[[Memory System]]** - How memory works
- **[[Configuration]]** - Advanced configuration

### **Build Something Amazing**
- **[[Use Cases]]** - Real-world applications
- **[[Production Deployment]]** - Deploy to production
- **[[Custom Providers]]** - Build your own providers

---

## 🆘 **Need Help?**

- **[[Troubleshooting]]** - Common issues and solutions
- **[[API Reference]]** - Complete API documentation
- **[GitHub Issues](https://github.com/paritosh0707/llmblocks/issues)** - Report bugs or ask questions

---

## 🎯 **Pro Tips**

### **💡 Environment Variables**
```bash
# Set multiple providers
export GOOGLE_API_KEY="your-gemini-key"
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-claude-key"
```

### **💡 Error Handling**
```python
try:
    ai = await get_provider("gemini")
    response = await ai.generate("Hello!")
except Exception as e:
    print(f"Error: {e}")
```

### **💡 Multiple Providers**
```python
# Use different providers
gemini_ai = await get_provider("gemini")
openai_ai = await get_provider("openai")
claude_ai = await get_provider("anthropic")
```

---

**Ready to dive deeper? Check out our [[Minimal Examples]] for more inspiration!** 🚀✨
