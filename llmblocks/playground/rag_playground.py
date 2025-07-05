"""
RAG Playground - Interactive Streamlit UI for testing RAG pipelines.

This playground provides a user-friendly interface for:
- Uploading and processing documents
- Configuring RAG pipeline settings
- Testing different RAG types (Basic, Streaming, Memory)
- Viewing results and conversation history
- Demonstrating LangChain Runnable interface (.invoke, .batch, .stream)
"""

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import os

# Import LLMBlocks components
from llmblocks.blocks.rag import BasicRAG, StreamingRAG, MemoryRAG, RAGConfig, create_rag
from llmblocks.core.config_loader import ConfigLoader
from llmblocks.core.llm_providers import LLMProviderFactory
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Page configuration
st.set_page_config(
    page_title="LLMBlocks RAG Playground",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .runnable-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'documents' not in st.session_state:
        st.session_state.documents = []
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'pipeline_initialized' not in st.session_state:
        st.session_state.pipeline_initialized = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'batch_questions' not in st.session_state:
        st.session_state.batch_questions = []

def create_documents_from_text(text: str, filename: str = "uploaded_text") -> List[Document]:
    """Create documents from uploaded text."""
    return [Document(
        page_content=text,
        metadata={"source": filename, "type": "text_upload"}
    )]

def create_documents_from_file(uploaded_file) -> List[Document]:
    """Create documents from uploaded file."""
    try:
        content = uploaded_file.read().decode('utf-8')
        return [Document(
            page_content=content,
            metadata={"source": uploaded_file.name, "type": "file_upload"}
        )]
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return []

def process_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Process documents with text splitting."""
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    processed_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            processed_docs.append(Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            ))

    return processed_docs

def create_rag_config_from_ui() -> RAGConfig:
    """Create RAG configuration from UI inputs."""
    return RAGConfig(
        name="playground_rag",
        chunk_size=st.session_state.get('chunk_size', 1000),
        chunk_overlap=st.session_state.get('chunk_overlap', 200),
        llm_provider=st.session_state.get('llm_provider', 'openai'),
        llm_model=st.session_state.get('llm_model', 'gpt-3.5-turbo'),
        llm_api_key=st.session_state.get('llm_api_key'),
        llm_base_url=st.session_state.get('llm_base_url'),
        llm_task=st.session_state.get('llm_task'),
        temperature=st.session_state.get('temperature', 0.0),
        max_tokens=st.session_state.get('max_tokens', 1000),
        top_k=st.session_state.get('top_k', 4),
        memory_enabled=st.session_state.get('memory_enabled', False),
        streaming_enabled=st.session_state.get('streaming_enabled', False)
    )

def display_metrics(documents: List[Document], processed_docs: List[Document]):
    """Display document processing metrics."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Original Documents", len(documents))

    with col2:
        st.metric("Processed Chunks", len(processed_docs))

    with col3:
        avg_chunk_size = sum(len(doc.page_content) for doc in processed_docs) / len(processed_docs) if processed_docs else 0
        st.metric("Avg Chunk Size", f"{avg_chunk_size:.0f} chars")

    with col4:
        total_chars = sum(len(doc.page_content) for doc in processed_docs)
        st.metric("Total Characters", f"{total_chars:,}")

def display_document_preview(documents: List[Document]):
    """Display a preview of uploaded documents."""
    if not documents:
        return

    st.subheader("📄 Document Preview")

    # Create a DataFrame for better display
    doc_data = []
    for i, doc in enumerate(documents[:5]):  # Show first 5 documents
        doc_data.append({
            "Document": i + 1,
            "Source": doc.metadata.get("source", "Unknown"),
            "Type": doc.metadata.get("type", "Unknown"),
            "Content Preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "Length": len(doc.page_content)
        })

    df = pd.DataFrame(doc_data)
    st.dataframe(df, use_container_width=True)

    if len(documents) > 5:
        st.info(f"Showing first 5 of {len(documents)} documents. Upload more to see additional previews.")

def display_runnable_info():
    """Display information about the new Runnable interface."""
    st.markdown("""
    <div class="runnable-info">
        <h4>🔄 LangChain Runnable Interface</h4>
        <p>This RAG playground now supports the new LangChain Runnable interface:</p>
        <ul>
            <li><strong>.invoke()</strong> - Standard single input execution</li>
            <li><strong>.batch()</strong> - Process multiple questions at once</li>
            <li><strong>.stream()</strong> - Real-time streaming responses</li>
            <li><strong>| operator</strong> - Chain with other components</li>
        </ul>
        <p><em>Note: .query() and .query_stream() are still available for backward compatibility.</em></p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main playground application."""
    initialize_session_state()

    # Header
    st.markdown('<h1 class="main-header">🧩 LLMBlocks RAG Playground</h1>', unsafe_allow_html=True)

    # Display Runnable interface info
    display_runnable_info()

    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # RAG Type Selection
        st.subheader("RAG Type")
        rag_type = st.selectbox(
            "Choose RAG Type",
            ["Basic", "Streaming", "Memory"],
            help="Basic: Simple Q&A, Streaming: Real-time responses, Memory: Conversation history"
        )

        # Document Processing Settings
        st.subheader("📄 Document Processing")
        st.session_state.chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=1000,
            step=100,
            help="Size of text chunks for processing"
        )

        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=200,
            step=50,
            help="Overlap between consecutive chunks"
        )

        # LLM Settings
        st.subheader("🤖 LLM Settings")

        # Provider selection
        available_providers = LLMProviderFactory.list_providers()
        st.session_state.llm_provider = st.selectbox(
            "Provider",
            available_providers,
            help="Select the LLM provider"
        )

        # Provider-specific model selection
        provider_models = {
            'openai': ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o"],
            'google': ["gemini-pro", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"],
            'huggingface': ["meta-llama/Llama-2-7b-chat-hf", "microsoft/DialoGPT-medium", "gpt2"],
            'groq': ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
            'anthropic': ["claude-3-sonnet-20240229", "claude-3-haiku-20240307", "claude-3-opus-20240229"],
            'ollama': ["llama2", "llama2:13b", "llama2:70b", "mistral", "codellama"]
        }

        models = provider_models.get(st.session_state.llm_provider, ["gpt-3.5-turbo"])
        st.session_state.llm_model = st.selectbox(
            "Model",
            models,
            help=f"Select {st.session_state.llm_provider} model"
        )

        # Provider-specific settings
        if st.session_state.llm_provider == 'huggingface':
            st.session_state.llm_task = st.selectbox(
                "Task",
                ["text-generation", "conversational"],
                help="Hugging Face model task"
            )

        if st.session_state.llm_provider == 'ollama':
            st.session_state.llm_base_url = st.text_input(
                "Ollama Base URL",
                value="http://localhost:11434",
                help="Ollama server URL"
            )

        # API Key input (optional, can use environment variables)
        st.session_state.llm_api_key = st.text_input(
            "API Key (optional)",
            type="password",
            help="Leave empty to use environment variable"
        )

        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Creativity level (0 = focused, 1 = creative)"
        )

        st.session_state.max_tokens = st.slider(
            "Max Tokens",
            min_value=100,
            max_value=4000,
            value=1000,
            step=100,
            help="Maximum response length"
        )

        st.session_state.top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of documents to retrieve"
        )

        # Feature toggles
        st.subheader("🔧 Features")
        st.session_state.memory_enabled = st.checkbox(
            "Enable Memory",
            value=False,
            help="Remember conversation history"
        )

        st.session_state.streaming_enabled = st.checkbox(
            "Enable Streaming",
            value=False,
            help="Stream responses in real-time"
        )

        # Pipeline status
        st.subheader("📊 Status")
        if st.session_state.pipeline_initialized:
            st.success("✅ Pipeline Ready")
        else:
            st.warning("⚠️ Pipeline Not Initialized")

        # Clear button
        if st.button("🗑️ Clear All", type="secondary"):
            st.session_state.documents = []
            st.session_state.rag_pipeline = None
            st.session_state.conversation_history = []
            st.session_state.pipeline_initialized = False
            st.session_state.uploaded_files = []
            st.session_state.batch_questions = []
            st.rerun()

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Document Upload")

        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['txt', 'md', 'py', 'js', 'html', 'json'],
            accept_multiple_files=True,
            help="Upload text files to create your knowledge base"
        )

        # Text input
        text_input = st.text_area(
            "Or paste text directly",
            height=150,
            placeholder="Paste your text here...",
            help="Enter text content directly"
        )

        # Process uploads
        if uploaded_files or text_input:
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.uploaded_files:
                        docs = create_documents_from_file(uploaded_file)
                        st.session_state.documents.extend(docs)
                        st.session_state.uploaded_files.append(uploaded_file.name)

            if text_input:
                docs = create_documents_from_text(text_input)
                st.session_state.documents.extend(docs)

        # Display document info
        if st.session_state.documents:
            st.success(f"✅ Loaded {len(st.session_state.documents)} documents")
            display_document_preview(st.session_state.documents)

            # Process documents
            processed_docs = process_documents(
                st.session_state.documents,
                st.session_state.chunk_size,
                st.session_state.chunk_overlap
            )

            display_metrics(st.session_state.documents, processed_docs)

            # Initialize pipeline button
            if st.button("🚀 Initialize RAG Pipeline", type="primary"):
                with st.spinner("Initializing RAG pipeline..."):
                    try:
                        config = create_rag_config_from_ui()

                        # Map UI selection to RAG type
                        rag_type_map = {
                            "Basic": "basic",
                            "Streaming": "streaming", 
                            "Memory": "memory"
                        }

                        st.session_state.rag_pipeline = create_rag(
                            rag_type_map[rag_type],
                            config
                        )

                        # Initialize and add documents
                        with st.session_state.rag_pipeline:
                            st.session_state.rag_pipeline.add_documents(processed_docs)

                        st.session_state.pipeline_initialized = True
                        st.success("✅ RAG pipeline initialized successfully!")

                    except NotImplementedError as e:
                        st.error(f"❌ Embedding setup required: {e}")
                        st.info("💡 This playground requires proper embedding setup. See documentation for details.")
                    except Exception as e:
                        st.error(f"❌ Error initializing pipeline: {e}")

    with col2:
        st.header("📋 Quick Actions")

        # Sample questions
        st.subheader("💡 Sample Questions")
        sample_questions = [
            "What is the main topic?",
            "Can you summarize the content?",
            "What are the key points?",
            "Explain the main concepts",
            "What are the important details?"
        ]

        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                st.session_state.current_question = question

        # Batch processing section
        if st.session_state.pipeline_initialized:
            st.subheader("🔄 Batch Processing")
            batch_input = st.text_area(
                "Enter multiple questions (one per line)",
                height=100,
                placeholder="Question 1\nQuestion 2\nQuestion 3",
                help="Process multiple questions at once using .batch()"
            )

            if batch_input and st.button("🚀 Process Batch"):
                questions = [q.strip() for q in batch_input.split('\n') if q.strip()]
                if questions:
                    with st.spinner(f"Processing {len(questions)} questions..."):
                        try:
                            responses = st.session_state.rag_pipeline.batch(questions)

                            st.success(f"✅ Processed {len(questions)} questions")

                            # Display batch results
                            for i, (question, response) in enumerate(zip(questions, responses)):
                                with st.expander(f"Q{i+1}: {question[:50]}...", expanded=False):
                                    st.markdown(f"**Question:** {question}")
                                    st.markdown(f"**Response:** {response}")
                        except Exception as e:
                            st.error(f"❌ Error in batch processing: {e}")

    # Chat interface
    if st.session_state.pipeline_initialized:
        st.header("💬 Chat Interface")

        # Question input
        question = st.text_input(
            "Ask a question",
            value=st.session_state.get('current_question', ''),
            placeholder="Type your question here...",
            key="question_input"
        )

        # Clear current question after use
        if 'current_question' in st.session_state:
            del st.session_state.current_question

        if question:
            col1, col2 = st.columns([3, 1])

            with col1:
                # Display question
                st.markdown(f"**You:** {question}")

                # Get response using new invoke() method
                with st.spinner("Generating response..."):
                    try:
                        if st.session_state.streaming_enabled and hasattr(st.session_state.rag_pipeline, 'stream'):
                            # Streaming response using new stream() method
                            response_placeholder = st.empty()
                            full_response = ""

                            for chunk in st.session_state.rag_pipeline.stream(question):
                                full_response += chunk
                                response_placeholder.markdown(f"**Assistant:** {full_response}")
                                time.sleep(0.01)  # Small delay for better UX
                        else:
                            # Regular response using new invoke() method
                            response = st.session_state.rag_pipeline.invoke(question)
                            st.markdown(f"**Assistant:** {response}")
                            full_response = response

                        # Add to conversation history
                        st.session_state.conversation_history.append({
                            "question": question,
                            "answer": full_response,
                            "timestamp": time.time()
                        })

                    except Exception as e:
                        st.error(f"❌ Error generating response: {e}")

            with col2:
                # Response metrics
                st.metric("Response Length", len(full_response))
                st.metric("History Entries", len(st.session_state.conversation_history))

        # Conversation history
        if st.session_state.conversation_history:
            st.header("📜 Conversation History")

            # Export button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("📥 Export History"):
                    history_data = {
                        "conversation": st.session_state.conversation_history,
                        "config": {
                            "rag_type": rag_type,
                            "chunk_size": st.session_state.chunk_size,
                            "chunk_overlap": st.session_state.chunk_overlap,
                            "llm_model": st.session_state.llm_model,
                            "temperature": st.session_state.temperature
                        }
                    }

                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(history_data, indent=2),
                        file_name="rag_conversation.json",
                        mime="application/json"
                    )

            with col2:
                if st.button("🗑️ Clear History"):
                    st.session_state.conversation_history = []
                    st.rerun()

            # Display history
            for i, entry in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Q{i+1}: {entry['question'][:50]}...", expanded=False):
                    st.markdown(f"**Question:** {entry['question']}")
                    st.markdown(f"**Answer:** {entry['answer']}")
                    st.caption(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry['timestamp']))}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🧩 LLMBlocks RAG Playground | Built with Streamlit</p>
            <p>For more information, visit the <a href='https://github.com/llmblocks/llmblocks'>documentation</a></p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()