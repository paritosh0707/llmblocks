#!/usr/bin/env python3
"""
Test script for LLMBlocks RAG Playground.

This script tests the basic functionality of the playground components.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")

    try:
        # Add project root to path
        project_root = Path(__file__).parent.absolute()
        sys.path.insert(0, str(project_root))

        # Test core imports
        print("  📦 Testing core imports...")
        from llmblocks.core.base_component import BaseComponent, ComponentConfig
        print("    ✅ BaseComponent imported successfully")

        from llmblocks.core.config_loader import ConfigLoader
        print("    ✅ ConfigLoader imported successfully")

        # Test RAG imports
        print("  🧩 Testing RAG imports...")
        from llmblocks.blocks.rag import BasicRAG, StreamingRAG, MemoryRAG, RAGConfig, create_rag
        print("    ✅ RAG components imported successfully")

        # Test playground imports
        print("  🎮 Testing playground imports...")
        import streamlit as st
        print("    ✅ Streamlit imported successfully")

        import pandas as pd
        print("    ✅ Pandas imported successfully")

        # Test LangChain imports
        print("  🔗 Testing LangChain imports...")
        from langchain_core.documents import Document
        print("    ✅ LangChain Document imported successfully")

        from langchain_text_splitters import RecursiveCharacterTextSplitter
        print("    ✅ LangChain TextSplitter imported successfully")

        print("✅ All imports successful!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_rag_creation():
    """Test RAG pipeline creation."""
    print("\n🧪 Testing RAG pipeline creation...")

    try:
        from llmblocks.blocks.rag import create_rag, RAGConfig

        # Test config creation
        config = RAGConfig(name="test_rag", chunk_size=500, top_k=3)
        print("    ✅ RAGConfig created successfully")

        # Test RAG creation
        rag = create_rag("basic", config)
        print("    ✅ BasicRAG created successfully")

        print("✅ RAG pipeline creation successful!")
        return True

    except Exception as e:
        print(f"❌ RAG creation error: {e}")
        return False

def test_config_loader():
    """Test configuration loader."""
    print("\n🧪 Testing configuration loader...")

    try:
        from llmblocks.core.config_loader import ConfigLoader

        # Test default config path
        config_path = ConfigLoader.get_default_config_path()
        print(f"    ✅ Default config path: {config_path}")

        # Test listing config files
        config_files = ConfigLoader.list_config_files()
        print(f"    ✅ Found {len(config_files)} config files")

        print("✅ Configuration loader test successful!")
        return True

    except Exception as e:
        print(f"❌ Config loader error: {e}")
        return False

def test_document_processing():
    """Test document processing functionality."""
    print("\n🧪 Testing document processing...")

    try:
        from langchain_core.documents import Document
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # Create test document
        test_text = "This is a test document. " * 50  # Create longer text
        doc = Document(page_content=test_text, metadata={"source": "test"})
        print("    ✅ Test document created")

        # Test text splitting
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_text(test_text)
        print(f"    ✅ Text split into {len(chunks)} chunks")

        print("✅ Document processing test successful!")
        return True

    except Exception as e:
        print(f"❌ Document processing error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧩 LLMBlocks RAG Playground Test Suite")
    print("=" * 50)

    tests = [
        test_imports,
        test_rag_creation,
        test_config_loader,
        test_document_processing
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The playground should work correctly.")
        print("\n🚀 To launch the playground, run:")
        print("   python run_playground.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main()) 