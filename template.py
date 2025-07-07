import os

# Define folder structure
project_structure = {
    "llmblocks": [
        "blocks",
        "core",
        "prebuilt",
        "cli/commands",
        "playground",
        "config"
    ],
    "examples": [],
    "tests": [
        "blocks",
        "core"
    ],
    "scripts": [],
    "docs": [
        "usage",
        "api"
    ]
}

# Top-level files
top_level_files = [
    "README.md",
    ".gitignore",
    "pyproject.toml",
    "LICENSE",
    "CHANGELOG.md"
]

# Files to create inside certain folders
template_files = {
    "llmblocks/__init__.py": "",
    "llmblocks/blocks/__init__.py": "",
    "llmblocks/blocks/rag.py": "# RAG Pipelines go here",
    "llmblocks/blocks/agents.py": "# LangGraph agent flows",
    "llmblocks/blocks/memory.py": "# Memory interfaces",
    "llmblocks/blocks/prompts.py": "# Prompt factory",
    "llmblocks/blocks/tools.py": "# Tool wrappers",
    "llmblocks/blocks/eval.py": "# Eval modules",
    "llmblocks/blocks/ui.py": "# Optional Streamlit dashboards",

    "llmblocks/core/__init__.py": "",
    "llmblocks/core/base_component.py": "# Base class for Blocks",
    "llmblocks/core/config_loader.py": "# YAML/Dict config loading",
    "llmblocks/core/tracing.py": "# Tracing utilities",
    "llmblocks/core/logger.py": "# Logger setup",
    "llmblocks/core/registry.py": "# Block registry if needed",
    "llmblocks/core/utils.py": "# load_docs, load_llm, etc.",

    "llmblocks/cli/__init__.py": "",
    "llmblocks/cli/main.py": "# Entry point for CLI commands",
    "llmblocks/cli/commands/__init__.py": "",

    "llmblocks/playground/__init__.py": "",
    "llmblocks/playground/rag_playground.py": "# UI for testing RAG",
    "llmblocks/playground/agent_playground.py": "# UI for testing agent",
    "llmblocks/playground/viewer.py": "# Logs and trace viewer",

    "llmblocks/config/prompts.yaml": "# prompt templates",
    "llmblocks/config/tools.yaml": "# tool configurations",
    "llmblocks/config/defaults.yaml": "# default params",

    "examples/basic_rag.py": "# Example: RAG with documents",
    "examples/multi_tool_agent.py": "# Example: LangGraph agent",

    "tests/conftest.py": "",
    "scripts/export_traces.py": "# Export trace data",
    "docs/index.md": "# Project documentation",
}

# Create folders
for root, subfolders in project_structure.items():
    os.makedirs(root, exist_ok=True)
    for folder in subfolders:
        os.makedirs(os.path.join(root, folder), exist_ok=True)

# Create files with placeholder content
for path, content in template_files.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

# Create top-level files
for filename in top_level_files:
    with open(filename, "w") as f:
        f.write(f"# {filename.replace('.md', '').upper()}")

print("✅ Project structure for LLMBlocks created successfully!")
