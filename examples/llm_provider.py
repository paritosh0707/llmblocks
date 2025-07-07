#!/usr/bin/env python3
"""
LLM Provider Factory Examples

This module demonstrates how to use the LLMFactory for creating and managing
LLM provider instances with comprehensive validation and error handling.
"""

from .factory import LLMFactory, AVAILABLE_PROVIDERS
from .openai_provider import OpenAIProvider


def run_examples():
    """Run all LLMFactory usage examples."""
    
    print("🚀 LLMFactory Usage Examples")
    print("=" * 50)
    
    # Example 1: Check pre-populated providers
    print("\n📋 Example 1: Pre-populated Providers")
    print(f"Available providers: {LLMFactory.list_available_providers()}")
    print(f"Direct access to AVAILABLE_PROVIDERS: {list(AVAILABLE_PROVIDERS.keys())}")
    
    # Example 2: Check if provider is available
    print(f"\n🔍 Example 2: Provider Availability Check")
    print(f"Is 'openai' available? {LLMFactory.is_provider_available('openai')}")
    print(f"Is 'OPENAI' available? {LLMFactory.is_provider_available('OPENAI')}")  # Case-insensitive
    print(f"Is 'invalid' available? {LLMFactory.is_provider_available('invalid')}")
    
    # Example 3: Creating provider with pre-registered provider
    print(f"\n🔧 Example 3: Create Provider (Pre-registered)")
    try:
        provider = LLMFactory.create_provider(
            provider_name="openai",
            api_key="sk-your-api-key-here",
            model_name="gpt-4o"
        )
        print(f"✅ Created {provider.PROVIDER_NAME} provider")
        print(f"   Model: {provider.credentials.model_name}")
        print(f"   Temperature: {provider.credentials.temperature}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 4: Register additional providers using the API
    print(f"\n➕ Example 4: Register Additional Providers")
    try:
        # Register aliases for OpenAI using the registration API
        LLMFactory.register_provider('gpt4', OpenAIProvider)
        LLMFactory.register_provider('custom_openai', OpenAIProvider)
        
        print(f"✅ Registered providers. Now available: {list(AVAILABLE_PROVIDERS.keys())}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 5: Using newly registered provider alias
    print(f"\n🔧 Example 5: Create Provider (Using Registered Alias)")
    try:
        provider = LLMFactory.create_provider(
            provider_name="gpt4",
            api_key="sk-your-api-key-here",
            model_name="gpt-4"
        )
        print(f"✅ Created provider using alias: {provider.PROVIDER_NAME}")
        print(f"   Model: {provider.credentials.model_name}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 6: Invalid provider
    print(f"\n❌ Example 6: Invalid Provider")
    try:
        provider = LLMFactory.create_provider(
            provider_name="invalid-provider",
            api_key="sk-your-api-key-here"
        )
        print(f"✅ Created provider: {provider.PROVIDER_NAME}")
    except Exception as e:
        print(f"❌ Expected error (invalid provider): {e}")
    
    # Example 7: Missing provider name
    print(f"\n❌ Example 7: Missing Provider Name")
    try:
        provider = LLMFactory.create_provider(
            api_key="sk-your-api-key-here",
            model_name="gpt-4"
        )
        print(f"✅ Created provider: {provider.PROVIDER_NAME}")
    except Exception as e:
        print(f"❌ Expected error (missing provider): {e}")
    
    # Example 8: Get provider class directly
    print(f"\n🔍 Example 8: Get Provider Class")
    try:
        provider_class = LLMFactory.get_provider('openai')
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
            provider_name="openai",
            api_key="sk-your-api-key-here",
            model_name="gpt-4o"
        )
        print(f"✅ Created with 'provider_name': {provider1.PROVIDER_NAME}")
        
        # Using provider key (alternative)
        provider2 = LLMFactory.create_provider(
            provider="openai",
            api_key="sk-your-api-key-here",
            model_name="gpt-4o-mini"
        )
        print(f"✅ Created with 'provider': {provider2.PROVIDER_NAME}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 11: Case-insensitive provider access
    print(f"\n🔤 Example 11: Case-Insensitive Provider Access")
    try:
        provider = LLMFactory.create_provider(
            provider_name="OPENAI",  # Uppercase
            api_key="sk-your-api-key-here",
            model_name="gpt-4o"
        )
        print(f"✅ Created with uppercase provider name: {provider.PROVIDER_NAME}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 12: Validation error handling
    print(f"\n⚠️ Example 12: Validation Error Handling")
    try:
        provider = LLMFactory.create_provider(
            provider_name="openai",
            api_key="",  # Invalid empty API key
            model_name="gpt-4o"
        )
        print(f"✅ Created provider: {provider.PROVIDER_NAME}")
    except Exception as e:
        print(f"❌ Expected validation error: {e}")
        print(f"   Error type: {type(e).__name__}")
    
    print(f"\n🎉 All examples completed!")
    print(f"Final provider count: {len(AVAILABLE_PROVIDERS)}")
    print(f"Final providers: {list(AVAILABLE_PROVIDERS.keys())}")


if __name__ == "__main__":
    run_examples() 