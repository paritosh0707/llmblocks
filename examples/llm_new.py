#!/usr/bin/env python3
"""
LLM Provider Factory Examples

This module demonstrates how to use the LLMFactory for creating and managing
LLM provider instances with comprehensive validation and error handling.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from llmblocks.blocks.llm_provider.factory import LLMFactory, AVAILABLE_PROVIDERS
from llmblocks.blocks.llm_provider.gemini_provider import GeminiProvider


def run_examples(kwargs):
    """
    Run all LLMFactory usage examples.
    
    Args:
        kwargs (dict): Configuration parameters including api_key for provider creation
    """
    
    print("🚀 LLMFactory Usage Examples")
    print("=" * 50)
    
    # Extract credentials from kwargs
    api_key = kwargs.get('api_key')
    if not api_key:
        raise ValueError("API key is required to run examples")
    
    # Example 1: Check pre-populated providers
    print("\n📋 Example 1: Pre-populated Providers")
    print(f"Available providers: {LLMFactory.list_available_providers()}")
    print(f"Direct access to AVAILABLE_PROVIDERS: {list(AVAILABLE_PROVIDERS.keys())}")
    
    # Example 2: Check if provider is available
    print(f"\n🔍 Example 2: Provider Availability Check")
    print(f"Is 'gemini' available? {LLMFactory.is_provider_available('gemini')}")
    print(f"Is 'GEMINI' available? {LLMFactory.is_provider_available('GEMINI')}")  # Case-insensitive
    print(f"Is 'invalid' available? {LLMFactory.is_provider_available('invalid')}")
    
    # Example 3: Creating provider with pre-registered provider
    print(f"\n🔧 Example 3: Create Provider (Pre-registered)")
    try:
        provider = LLMFactory.create_provider(
            provider_name="gemini",
            api_key=api_key,
            model_name="gemini-2.0-flash"
        )
        print(f"✅ Created {provider.PROVIDER_NAME} provider")
        # print(f"   Model: {provider.credentials.model_name}")
        # print(f"   Temperature: {provider.credentials.temperature}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: Register additional providers using the API
    print(f"\n➕ Example 4: Register Additional Providers")
    try:
        # Register aliases for Gemini using the registration API
        LLMFactory.register_provider('gemini_flash', GeminiProvider)
        LLMFactory.register_provider('custom_gemini', GeminiProvider)
        
        print(f"✅ Registered providers. Now available: {list(AVAILABLE_PROVIDERS.keys())}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 5: Using newly registered provider alias
    print(f"\n🔧 Example 5: Create Provider (Using Registered Alias)")
    try:
        provider = LLMFactory.create_provider(
            provider_name="gemini",
            api_key=api_key,
            model_name="gemini-2.0-flash"
        )
        print(f"✅ Created provider using alias: {provider.PROVIDER_NAME}")
        # print(f"   Model: {provider.credentials.model_name}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 6: Invalid provider
    print(f"\n❌ Example 6: Invalid Provider")
    try:
        provider = LLMFactory.create_provider(
            provider_name="invalid-provider",
            api_key=api_key
        )
        print(f"✅ Created provider: {provider.PROVIDER_NAME}")
    except Exception as e:
        print(f"❌ Expected error (invalid provider): {e}")
    
    # Example 7: Missing provider name
    print(f"\n❌ Example 7: Missing Provider Name")
    try:
        provider = LLMFactory.create_provider(
            api_key=api_key,
            model_name="gemini-2.0-flash"
        )
        print(f"✅ Created provider: {provider.PROVIDER_NAME}")
    except Exception as e:
        print(f"❌ Expected error (missing provider): {e}")
    
    # Example 8: Get provider class directly
    print(f"\n🔍 Example 8: Get Provider Class")
    try:
        provider_class = LLMFactory.get_provider('gemini')
        print(f"✅ Got provider class: {provider_class}")
        print(f"   Class name: {provider_class.__name__}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 9: Provider validation
    print(f"\n✅ Example 9: Provider Validation")
    print(f"Total providers registered: {len(AVAILABLE_PROVIDERS)}")
    print(f"Provider names: {', '.join(AVAILABLE_PROVIDERS.keys())}")
    
    # Example 10: Using provider_name vs provider key
    print(f"\n🔑 Example 10: Different Provider Keys")
    try:
        # Using provider_name key
        provider1 = LLMFactory.create_provider(
            provider_name="gemini",
            api_key=api_key,
            model_name="gemini-2.0-flash"
        )
        print(f"✅ Created with 'provider_name': {provider1.PROVIDER_NAME}")
        
        # Using provider key (alternative)
        provider2 = LLMFactory.create_provider(
            provider="gemini",
            api_key=api_key,
            model_name="gemini-2.0-flash-lite"
        )
        print(f"✅ Created with 'provider': {provider2.PROVIDER_NAME}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 11: Case-insensitive provider access
    print(f"\n🔤 Example 11: Case-Insensitive Provider Access")
    try:
        provider = LLMFactory.create_provider(
            provider_name="GEMINI",  # Uppercase
            api_key=api_key,
            model_name="gemini-2.0-flash"
        )
        print(f"✅ Created with uppercase provider name: {provider.PROVIDER_NAME}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 12: Validation error handling
    print(f"\n⚠️ Example 12: Validation Error Handling")
    try:
        provider = LLMFactory.create_provider(
            provider_name="gemini",
            api_key="",  # Invalid empty API key
            model_name="gemini-2.0-flash"
        )
        print(f"✅ Created provider: {provider.PROVIDER_NAME}")
    except Exception as e:
        print(f"❌ Expected validation error: {e}")
        print(f"   Error type: {type(e).__name__}")

    # Example 13: Test LLM invocation
    print(f"\n🤖 Example 13: Test LLM Invocation")
    try:
        provider = LLMFactory.create_provider(**kwargs)
        print(f"✅ Created provider: {provider.PROVIDER_NAME}")
        
        llm = provider.get_llm()
        if llm is None:
            raise ValueError("LLM instance is None")
            
        response = llm.invoke("Hello, how are you?")
        print(f"Response from LLM: {response}")
    except Exception as e:
        print(f"❌ Error during LLM invocation: {e}")
        print(f"   Error type: {type(e).__name__}")
    
    print(f"\n🎉 All examples completed!")
    print(f"Final provider count: {len(AVAILABLE_PROVIDERS)}")
    print(f"Final providers: {list(AVAILABLE_PROVIDERS.keys())}")


if __name__ == "__main__":
    print("Starting LLMFactory examples...")
    try:
        # Get API key from environment or user input
        api_key = input("Please enter your Google API key: ").strip() or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("API key is required")
            
        kwargs = {
            "provider_name": "gemini",
            "api_key": api_key,
            "model_name": "gemini-2.0-flash"
        }
        run_examples(kwargs)
    except Exception as e:
        print(f"❌ Error: {e}")
    print("LLMFactory examples completed!")