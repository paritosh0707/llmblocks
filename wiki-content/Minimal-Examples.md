# ✨ Minimal Examples - AI in 2-10 Lines

Collection of minimal, production-ready examples demonstrating LLMBlocks' power with incredibly concise code.

---

## 🎯 **Philosophy: Less Code, More AI**

LLMBlocks transforms complex AI development into simple, readable code. Each example is designed to be:
- **Minimal**: 2-10 lines of actual code
- **Complete**: Ready to run as-is
- **Practical**: Solves real problems
- **Educational**: Shows best practices

---

## 🚀 **Quick Reference**

| Example | Lines | Description | Use Case |
|---------|-------|-------------|----------|
| **Hello World** | 2 | Basic AI response | Getting started |
| **Smart Chatbot** | 4 | Interactive chat | Simple chatbots |
| **Streaming AI** | 4 | Real-time responses | Live interactions |
| **Multi-Provider** | 6 | Provider switching | Reliability |
| **Stateful AI** | 3 | Memory-enabled AI | Conversations |
| **Persistent AI** | 3 | Cross-session memory | User sessions |
| **Zero Config** | 2 | Auto-configuration | Rapid prototyping |
| **RAG System** | 8 | Knowledge retrieval | Q&A systems |
| **Function Calling** | 6 | Tool integration | AI agents |

---

## 🌟 **Core Examples**

### **1. Hello World (2 lines)**
```python
# examples/minimal/01_hello_world.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    ai = await get_provider("gemini")  # Line 1: Get AI
    print(await ai.generate("Hello!"))  # Line 2: Generate response

asyncio.run(main())
```

**Output:**
```
Hello! It's great to meet you. How can I help you today?
```

---

### **2. Smart Chatbot (4 lines)**
```python
# examples/minimal/02_smart_chatbot.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    ai = await get_provider("gemini")  # Line 1: Get AI
    
    while True:  # Line 2: Chat loop
        user_input = input("You: ")  # Line 3: Get user input
        response = await ai.generate(user_input)  # Line 4: AI response
        print(f"AI: {response.content}")

asyncio.run(main())
```

**Usage:**
```
You: What's the capital of France?
AI: The capital of France is Paris.
You: Tell me a fun fact about it.
AI: Paris has more dogs than children! There are about 300,000 dogs compared to 260,000 children under 20.
```

---

### **3. Streaming AI (4 lines)**
```python
# examples/minimal/03_streaming_ai.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    ai = await get_provider("gemini", stream=True)  # Line 1: Get streaming AI
    
    print("AI: ", end="", flush=True)  # Line 2: Start output
    async for chunk in ai.generate_stream("Tell me a short story"):  # Line 3: Stream
        print(chunk.content, end="", flush=True)  # Line 4: Print chunks
    print()

asyncio.run(main())
```

**Output:**
```
AI: Once upon a time, in a small village nestled between rolling hills, there lived a young baker named Maya who discovered that her bread could grant wishes to those who truly needed help...
```

---

### **4. Multi-Provider Fallback (6 lines)**
```python
# examples/minimal/04_multi_provider.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    providers = ["gemini", "openai", "anthropic"]  # Line 1: Provider list
    
    for provider in providers:  # Line 2: Try each provider
        try:  # Line 3: Error handling
            ai = await get_provider(provider)  # Line 4: Get provider
            response = await ai.generate("Hello!")  # Line 5: Generate
            print(f"{provider}: {response.content}")  # Line 6: Success!
            break
        except Exception as e:
            print(f"{provider} failed: {e}")

asyncio.run(main())
```

**Output:**
```
gemini: Hello! I'm happy to help you today. What would you like to know or discuss?
```

---

### **5. Stateful AI with Memory (3 lines)**
```python
# examples/minimal/05_stateful_ai.py
import asyncio
from llmblocks.blocks.llm_provider import get_stateful_ai

async def main():
    ai = await get_stateful_ai("gemini")  # Line 1: Get stateful AI
    await ai.chat("I'm Alice, a data scientist from NYC.")  # Line 2: First message
    response = await ai.chat("What do you know about me?")  # Line 3: AI remembers!
    print(f"AI: {response.content}")

asyncio.run(main())
```

**Output:**
```
AI: I know that you're Alice, and you work as a data scientist in New York City! It's nice to meet you, Alice.
```

---

### **6. Persistent AI Across Sessions (3 lines)**
```python
# examples/minimal/06_persistent_ai.py
import asyncio
from llmblocks.blocks.llm_provider import get_persistent_ai

async def main():
    ai = await get_persistent_ai("gemini", session_id="user_123")  # Line 1: Persistent AI
    await ai.chat("My favorite color is purple.")  # Line 2: Store preference
    # Later, in a different session...
    response = await ai.chat("What's my favorite color?")  # Line 3: Remembers!
    print(f"AI: {response.content}")

asyncio.run(main())
```

**Output:**
```
AI: Your favorite color is purple!
```

---

### **7. Zero Configuration AI (2 lines)**
```python
# examples/minimal/07_zero_config.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    ai = await get_provider("gemini")  # Line 1: Auto-configured AI
    print(await ai.generate("Explain quantum computing in one sentence"))  # Line 2: Done!

asyncio.run(main())
```

**Output:**
```
Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that could solve certain problems exponentially faster than classical computers.
```

---

## 🧠 **Advanced Examples**

### **8. RAG System (8 lines)**
```python
# examples/minimal/08_rag_system.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    knowledge = "LLMBlocks is a Python framework for building AI applications with minimal code."  # Line 1: Knowledge
    
    ai = await get_provider("gemini")  # Line 2: Get AI
    
    query = "What is LLMBlocks?"  # Line 3: User query
    context_prompt = f"Context: {knowledge}\n\nQuestion: {query}\nAnswer:"  # Line 4: RAG prompt
    
    response = await ai.generate(context_prompt)  # Line 5: Generate with context
    print(f"AI: {response.content}")  # Line 6: Output

asyncio.run(main())
```

**Output:**
```
AI: LLMBlocks is a Python framework designed for building AI applications with minimal code, making it easier for developers to create powerful AI solutions without complex boilerplate.
```

---

### **9. Function Calling AI (6 lines)**
```python
# examples/minimal/09_function_calling.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

def get_weather(location: str) -> str:  # Line 1: Define function
    return f"The weather in {location} is sunny and 72°F"

async def main():
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}]  # Line 2: Tool definition
    
    ai = await get_provider("openai", tools=tools)  # Line 3: AI with tools
    response = await ai.generate("What's the weather in San Francisco?")  # Line 4: Query
    
    if response.tool_calls:  # Line 5: Handle tool calls
        result = get_weather(response.tool_calls[0].function.arguments["location"])  # Line 6: Execute
        print(f"AI: {result}")

asyncio.run(main())
```

---

### **10. Conversation Summary (5 lines)**
```python
# examples/minimal/10_conversation_summary.py
import asyncio
from llmblocks.blocks.llm_provider import get_stateful_ai

async def main():
    ai = await get_stateful_ai("gemini", memory_config={"context_strategy": "summarized", "max_messages": 10})  # Line 1: AI with summarization
    
    # Simulate long conversation
    for i in range(15):  # Line 2: Long conversation
        await ai.chat(f"Message {i}: Tell me about topic {i}")  # Line 3: Add messages
    
    response = await ai.chat("Summarize our conversation")  # Line 4: Request summary
    print(f"AI: {response.content}")  # Line 5: Show summary

asyncio.run(main())
```

---

## 🎨 **Creative Examples**

### **11. AI Story Generator (3 lines)**
```python
# examples/creative/story_generator.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    ai = await get_provider("gemini", temperature=0.9)  # Line 1: Creative AI
    story = await ai.generate("Write a 50-word sci-fi story about a robot discovering emotions")  # Line 2: Generate
    print(f"📖 Story: {story.content}")  # Line 3: Display

asyncio.run(main())
```

---

### **12. Code Reviewer (4 lines)**
```python
# examples/tools/code_reviewer.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    code = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"  # Line 1: Code to review
    ai = await get_provider("gemini")  # Line 2: Get AI
    prompt = f"Review this Python code and suggest improvements:\n{code}"  # Line 3: Review prompt
    review = await ai.generate(prompt)  # Line 4: Get review
    print(f"🔍 Review: {review.content}")

asyncio.run(main())
```

---

### **13. Language Translator (3 lines)**
```python
# examples/tools/translator.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def main():
    ai = await get_provider("gemini")  # Line 1: Get AI
    translation = await ai.generate("Translate 'Hello, how are you?' to Spanish, French, and German")  # Line 2: Translate
    print(f"🌍 Translations: {translation.content}")  # Line 3: Show results

asyncio.run(main())
```

---

## 🏢 **Production Examples**

### **14. Customer Support Bot (7 lines)**
```python
# examples/production/customer_support.py
import asyncio
from llmblocks.blocks.llm_provider import get_persistent_ai

async def handle_customer(customer_id: str, message: str):
    ai = await get_persistent_ai("gemini", session_id=f"customer_{customer_id}")  # Line 1: Customer AI
    
    system_prompt = "You are a helpful customer support agent. Be polite and professional."  # Line 2: System prompt
    full_prompt = f"{system_prompt}\n\nCustomer: {message}\nAgent:"  # Line 3: Format prompt
    
    response = await ai.generate(full_prompt)  # Line 4: Generate response
    return response.content  # Line 5: Return response

# Usage
async def main():
    response = await handle_customer("123", "I need help with my order")  # Line 6: Handle request
    print(f"Agent: {response}")  # Line 7: Display response

asyncio.run(main())
```

---

### **15. Content Moderator (5 lines)**
```python
# examples/production/content_moderator.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def moderate_content(text: str) -> bool:
    ai = await get_provider("gemini", temperature=0.1)  # Line 1: Conservative AI
    prompt = f"Is this text appropriate for a family-friendly platform? Answer only 'YES' or 'NO': {text}"  # Line 2: Moderation prompt
    response = await ai.generate(prompt)  # Line 3: Check content
    return "YES" in response.content.upper()  # Line 4: Parse result

async def main():
    is_safe = await moderate_content("Hello, this is a nice message!")  # Line 5: Test
    print(f"Content is safe: {is_safe}")

asyncio.run(main())
```

---

## 🧪 **Experimental Examples**

### **16. AI Pair Programmer (6 lines)**
```python
# examples/experimental/pair_programmer.py
import asyncio
from llmblocks.blocks.llm_provider import get_stateful_ai

async def main():
    ai = await get_stateful_ai("gemini")  # Line 1: Stateful AI
    await ai.chat("I'm working on a Python web API using FastAPI")  # Line 2: Context
    
    while True:  # Line 3: Programming loop
        task = input("What should we code? ")  # Line 4: Get task
        code = await ai.chat(f"Help me code: {task}")  # Line 5: Get help
        print(f"💻 AI: {code.content}")  # Line 6: Show code

asyncio.run(main())
```

---

### **17. Sentiment Analyzer (4 lines)**
```python
# examples/tools/sentiment_analyzer.py
import asyncio
from llmblocks.blocks.llm_provider import get_provider

async def analyze_sentiment(text: str):
    ai = await get_provider("gemini", temperature=0.2)  # Line 1: Analytical AI
    prompt = f"Analyze the sentiment of this text (positive/negative/neutral): '{text}'"  # Line 2: Analysis prompt
    result = await ai.generate(prompt)  # Line 3: Analyze
    return result.content  # Line 4: Return result

# Usage
async def main():
    sentiment = await analyze_sentiment("I love this new feature!")
    print(f"Sentiment: {sentiment}")

asyncio.run(main())
```

---

## 🎯 **Usage Patterns**

### **Error Handling Pattern**
```python
async def robust_ai():
    try:
        ai = await get_provider("gemini")
        return await ai.generate("Hello")
    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, AI is unavailable"
```

### **Configuration Pattern**
```python
async def configured_ai():
    config = {
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": True
    }
    return await get_provider("gemini", **config)
```

### **Batch Processing Pattern**
```python
async def batch_process(prompts):
    ai = await get_provider("gemini")
    tasks = [ai.generate(prompt) for prompt in prompts]
    return await asyncio.gather(*tasks)
```

---

## 📊 **Performance Tips**

### **Optimize for Speed**
```python
# Use streaming for long responses
ai = await get_provider("gemini", stream=True)

# Batch multiple requests
tasks = [ai.generate(prompt) for prompt in prompts]
results = await asyncio.gather(*tasks)
```

### **Optimize for Cost**
```python
# Use shorter max_tokens
ai = await get_provider("gemini", max_tokens=100)

# Use lower temperature for consistent results
ai = await get_provider("gemini", temperature=0.1)
```

### **Optimize for Memory**
```python
# Use recent context strategy
config = MemoryConfig(context_strategy="recent", max_messages=20)
ai = await get_stateful_ai("gemini", memory_config=config)
```

---

## 🚀 **Running the Examples**

### **Prerequisites**
```bash
# Install LLMBlocks
pip install llmblocks

# Set API key
export GOOGLE_API_KEY="your-api-key"
```

### **Run Any Example**
```bash
# Download example
curl -O https://raw.githubusercontent.com/paritosh0707/llmblocks/main/examples/minimal/01_hello_world.py

# Run it
python 01_hello_world.py
```

### **Modify for Your Needs**
```python
# Change provider
ai = await get_provider("openai")  # Instead of "gemini"

# Change model
ai = await get_provider("gemini", model="gemini-1.5-pro")

# Add configuration
ai = await get_provider("gemini", temperature=0.9, max_tokens=200)
```

---

## 🎊 **Your Turn!**

These examples show the power of LLMBlocks - complex AI functionality in just a few lines of code. 

**Pick an example, modify it, and build something amazing!**

### **Next Steps**
- **[[API Reference]]** - Complete API documentation
- **[[Advanced Examples]]** - More complex applications
- **[[Production Deployment]]** - Deploy your AI applications
- **[[Custom Providers]]** - Build your own providers

**Happy coding! 🚀✨**
