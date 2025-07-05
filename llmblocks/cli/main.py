"""
CLI interface for LLMBlocks.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from ..blocks.rag import create_rag, RAGConfig
from langchain.schema import Document


def create_parser() -> argparse.ArgumentParser:
    """Create the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="LLMBlocks - Modular LLM-powered systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic RAG query
  llmblocks rag query --question "What is LLMBlocks?" --documents docs/

  # Streaming RAG
  llmblocks rag stream --question "Explain RAG" --documents docs/

  # Memory RAG with conversation
  llmblocks rag chat --documents docs/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # RAG command
    rag_parser = subparsers.add_parser('rag', help='RAG pipeline operations')
    rag_subparsers = rag_parser.add_subparsers(dest='rag_command', help='RAG subcommands')
    
    # RAG query command
    query_parser = rag_subparsers.add_parser('query', help='Query RAG pipeline')
    query_parser.add_argument('--question', '-q', required=True, help='Question to ask')
    query_parser.add_argument('--documents', '-d', required=True, help='Path to documents directory or file')
    query_parser.add_argument('--rag-type', default='basic', choices=['basic', 'streaming', 'memory'], 
                             help='Type of RAG pipeline')
    query_parser.add_argument('--config', help='Path to configuration file')
    query_parser.add_argument('--chunk-size', type=int, default=1000, help='Document chunk size')
    query_parser.add_argument('--top-k', type=int, default=4, help='Number of documents to retrieve')
    query_parser.add_argument('--model', default='gpt-3.5-turbo', help='LLM model to use')
    query_parser.add_argument('--temperature', type=float, default=0.0, help='LLM temperature')
    
    # RAG stream command
    stream_parser = rag_subparsers.add_parser('stream', help='Stream RAG response')
    stream_parser.add_argument('--question', '-q', required=True, help='Question to ask')
    stream_parser.add_argument('--documents', '-d', required=True, help='Path to documents directory or file')
    stream_parser.add_argument('--config', help='Path to configuration file')
    stream_parser.add_argument('--chunk-size', type=int, default=1000, help='Document chunk size')
    stream_parser.add_argument('--top-k', type=int, default=4, help='Number of documents to retrieve')
    stream_parser.add_argument('--model', default='gpt-3.5-turbo', help='LLM model to use')
    stream_parser.add_argument('--temperature', type=float, default=0.0, help='LLM temperature')
    
    # RAG chat command
    chat_parser = rag_subparsers.add_parser('chat', help='Interactive RAG chat')
    chat_parser.add_argument('--documents', '-d', required=True, help='Path to documents directory or file')
    chat_parser.add_argument('--config', help='Path to configuration file')
    chat_parser.add_argument('--chunk-size', type=int, default=1000, help='Document chunk size')
    chat_parser.add_argument('--top-k', type=int, default=4, help='Number of documents to retrieve')
    chat_parser.add_argument('--model', default='gpt-3.5-turbo', help='LLM model to use')
    chat_parser.add_argument('--temperature', type=float, default=0.0, help='LLM temperature')
    
    return parser


def load_documents(path: str) -> list[Document]:
    """Load documents from a file or directory."""
    path_obj = Path(path)
    documents = []
    
    if path_obj.is_file():
        # Load single file
        try:
            with open(path_obj, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(
                    page_content=content,
                    metadata={"source": str(path_obj)}
                ))
        except Exception as e:
            print(f"Error reading file {path}: {e}")
            return []
    
    elif path_obj.is_dir():
        # Load all text files in directory
        for file_path in path_obj.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ['.txt', '.md', '.py', '.js', '.html']:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": str(file_path)}
                        ))
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")
    
    else:
        print(f"Path {path} does not exist")
        return []
    
    print(f"Loaded {len(documents)} documents")
    return documents


def create_rag_config(args) -> RAGConfig:
    """Create RAG configuration from command line arguments."""
    config = RAGConfig(
        name="cli_rag",
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        llm_model=args.model,
        temperature=args.temperature
    )
    
    # Load from config file if provided
    if args.config and os.path.exists(args.config):
        # TODO: Implement config file loading
        print(f"Config file loading not yet implemented: {args.config}")
    
    return config


def run_rag_query(args) -> None:
    """Run a single RAG query."""
    print(f"Loading documents from: {args.documents}")
    documents = load_documents(args.documents)
    
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    print(f"Creating {args.rag_type} RAG pipeline...")
    config = create_rag_config(args)
    
    try:
        rag = create_rag(args.rag_type, config)
        
        with rag:
            rag.add_documents(documents)
            
            print(f"\nQuestion: {args.question}")
            print("Answer: ", end="", flush=True)
            
            answer = rag.query(args.question)
            print(answer)
            
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print("Note: This requires proper embedding setup. See documentation for details.")
    except Exception as e:
        print(f"\nError: {e}")


def run_rag_stream(args) -> None:
    """Run a streaming RAG query."""
    print(f"Loading documents from: {args.documents}")
    documents = load_documents(args.documents)
    
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    print("Creating streaming RAG pipeline...")
    config = create_rag_config(args)
    
    try:
        rag = create_rag("streaming", config)
        
        with rag:
            rag.add_documents(documents)
            
            print(f"\nQuestion: {args.question}")
            print("Answer: ", end="", flush=True)
            
            for chunk in rag.query_stream(args.question):
                print(chunk, end="", flush=True)
            print()  # New line after streaming
            
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print("Note: This requires proper embedding setup. See documentation for details.")
    except Exception as e:
        print(f"\nError: {e}")


def run_rag_chat(args) -> None:
    """Run interactive RAG chat."""
    print(f"Loading documents from: {args.documents}")
    documents = load_documents(args.documents)
    
    if not documents:
        print("No documents loaded. Exiting.")
        return
    
    print("Creating memory RAG pipeline...")
    config = create_rag_config(args)
    
    try:
        rag = create_rag("memory", config)
        
        with rag:
            rag.add_documents(documents)
            
            print("\n=== LLMBlocks RAG Chat ===")
            print("Type 'quit' or 'exit' to end the conversation")
            print("Type 'clear' to clear conversation memory")
            print("=" * 30)
            
            while True:
                try:
                    question = input("\nYou: ").strip()
                    
                    if question.lower() in ['quit', 'exit']:
                        print("Goodbye!")
                        break
                    
                    if question.lower() == 'clear':
                        if hasattr(rag, 'clear_memory'):
                            rag.clear_memory()
                            print("Conversation memory cleared.")
                        continue
                    
                    if not question:
                        continue
                    
                    print("Assistant: ", end="", flush=True)
                    answer = rag.query(question)
                    print(answer)
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                except Exception as e:
                    print(f"Error: {e}")
            
    except NotImplementedError as e:
        print(f"\nError: {e}")
        print("Note: This requires proper embedding setup. See documentation for details.")
    except Exception as e:
        print(f"\nError: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'rag':
        if not args.rag_command:
            parser.print_help()
            return
        
        if args.rag_command == 'query':
            run_rag_query(args)
        elif args.rag_command == 'stream':
            run_rag_stream(args)
        elif args.rag_command == 'chat':
            run_rag_chat(args)
        else:
            print(f"Unknown RAG command: {args.rag_command}")
            parser.print_help()
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == '__main__':
    main()