#!/usr/bin/env python3
"""
🚀 Production Ready - Just 10 lines!

A production-ready AI service with error handling, logging, and monitoring.
"""
import asyncio
import sys
import os
sys.path.insert(0, 'src')

async def main():
    # Setup
    os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-api-key-here')
    
    print("🚀 Production Ready AI - Enterprise features in 10 lines!")
    print("Includes: Error handling, logging, health checks, and monitoring\n")
    
    # ✨ THE MAGIC - Only 10 lines for production-ready AI!
    from llmblocks.blocks.llm_provider import get_provider
    from llmblocks.utils.logging import get_logger
    from llmblocks.core.tracing import trace_span
    
    logger = get_logger("ProductionAI")
    provider = get_provider("gemini", api_key=os.environ['GOOGLE_API_KEY'])
    
    # Health check
    health = await provider.health_check()
    logger.info("AI service health check", status=health['status'])
    
    # Process request with monitoring
    with trace_span("ai_request", metadata={"user": "demo"}):
        try:
            response = await provider.generate("What are the benefits of using LLMBlocks?")
            logger.info("AI request successful", response_length=len(response.content))
            print(f"🤖 AI Response: {response.content}")
        except Exception as e:
            logger.error("AI request failed", error=str(e))
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Production-ready AI service with monitoring!")
    print(f"📊 Health Status: {health['status']}")
    print(f"🔍 Check logs for detailed monitoring data")

if __name__ == "__main__":
    asyncio.run(main())
