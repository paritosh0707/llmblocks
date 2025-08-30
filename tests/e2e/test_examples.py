"""
End-to-end tests for LLMBlocks examples.

These tests verify that all examples work correctly.
"""

import pytest
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict


class TestMinimalExamples:
    """Test all minimal examples."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    @pytest.fixture
    def examples_dir(self, project_root):
        """Get examples directory."""
        return project_root / "examples" / "minimal"
    
    @pytest.fixture
    def skip_if_no_api_key(self):
        """Skip test if API key is not available."""
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set - skipping example tests")
    
    def run_example(self, example_path: Path, timeout: int = 30) -> Dict[str, any]:
        """Run an example and return result."""
        try:
            result = subprocess.run(
                [sys.executable, str(example_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=example_path.parent.parent.parent  # Project root
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Test timed out",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def test_check_setup(self, examples_dir):
        """Test the setup checker."""
        example_path = examples_dir / "00_check_setup.py"
        assert example_path.exists(), "Setup checker example not found"
        
        result = self.run_example(example_path)
        
        # Should always succeed (even without API key)
        assert result["success"], f"Setup check failed: {result['stderr']}"
        assert "LLMBlocks Environment Check" in result["stdout"]
    
    def test_hello_world(self, examples_dir, skip_if_no_api_key):
        """Test the hello world example."""
        example_path = examples_dir / "01_hello_world.py"
        assert example_path.exists(), "Hello world example not found"
        
        result = self.run_example(example_path)
        
        assert result["success"], f"Hello world failed: {result['stderr']}"
        assert "🚀 LLMBlocks Hello World" in result["stdout"]
        assert "🤖 AI Response:" in result["stdout"]
        assert "✅ That's it!" in result["stdout"]
    
    def test_streaming_ai(self, examples_dir, skip_if_no_api_key):
        """Test the streaming AI example."""
        example_path = examples_dir / "03_streaming_ai.py"
        assert example_path.exists(), "Streaming AI example not found"
        
        result = self.run_example(example_path)
        
        assert result["success"], f"Streaming AI failed: {result['stderr']}"
        assert "🚀 LLMBlocks Streaming AI" in result["stdout"]
        assert "🤖 AI:" in result["stdout"]
    
    @pytest.mark.slow
    def test_multi_provider(self, examples_dir):
        """Test the multi-provider example."""
        example_path = examples_dir / "04_multi_provider.py"
        if not example_path.exists():
            pytest.skip("Multi-provider example not implemented yet")
        
        # Only test if we have multiple API keys
        api_keys = [
            os.getenv("GOOGLE_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY")
        ]
        
        if sum(1 for key in api_keys if key) < 2:
            pytest.skip("Need at least 2 API keys for multi-provider test")
        
        result = self.run_example(example_path, timeout=60)
        assert result["success"], f"Multi-provider failed: {result['stderr']}"
    
    def test_zero_config(self, examples_dir, skip_if_no_api_key):
        """Test the zero config example."""
        example_path = examples_dir / "08_zero_config.py"
        if not example_path.exists():
            pytest.skip("Zero config example not implemented yet")
        
        result = self.run_example(example_path)
        assert result["success"], f"Zero config failed: {result['stderr']}"


class TestAdvancedExamples:
    """Test advanced examples."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    @pytest.fixture
    def examples_dir(self, project_root):
        """Get advanced examples directory."""
        return project_root / "examples" / "advanced"
    
    @pytest.mark.slow
    def test_rag_example(self, examples_dir):
        """Test RAG example if it exists."""
        example_path = examples_dir / "rag_example.py"
        if not example_path.exists():
            pytest.skip("RAG example not implemented yet")
        
        # Test would go here when RAG is implemented
    
    @pytest.mark.slow
    def test_agent_example(self, examples_dir):
        """Test agent example if it exists."""
        example_path = examples_dir / "agent_example.py"
        if not example_path.exists():
            pytest.skip("Agent example not implemented yet")
        
        # Test would go here when agents are implemented


class TestIntegrationExamples:
    """Test integration with external systems."""
    
    @pytest.fixture
    def project_root(self):
        """Get project root directory."""
        return Path(__file__).parent.parent.parent
    
    def test_langchain_compatibility(self, project_root):
        """Test LangChain compatibility examples."""
        example_path = project_root / "examples" / "minimal" / "05_langchain_magic.py"
        
        if not example_path.exists():
            pytest.skip("LangChain magic example not implemented yet")
        
        if not os.getenv("GOOGLE_API_KEY"):
            pytest.skip("GOOGLE_API_KEY not set")
        
        try:
            result = subprocess.run(
                [sys.executable, str(example_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root
            )
            
            assert result.returncode == 0, f"LangChain compatibility failed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("LangChain compatibility test timed out")


class TestExampleDocumentation:
    """Test that examples are properly documented."""
    
    @pytest.fixture
    def examples_dir(self):
        """Get examples directory."""
        return Path(__file__).parent.parent.parent / "examples"
    
    def test_readme_exists(self, examples_dir):
        """Test that README files exist."""
        main_readme = examples_dir / "README.md"
        minimal_readme = examples_dir / "minimal" / "README.md"
        advanced_readme = examples_dir / "advanced" / "README.md"
        
        assert main_readme.exists(), "Main examples README missing"
        assert minimal_readme.exists(), "Minimal examples README missing"
        assert advanced_readme.exists(), "Advanced examples README missing"
    
    def test_examples_have_docstrings(self, examples_dir):
        """Test that examples have proper docstrings."""
        minimal_dir = examples_dir / "minimal"
        
        for example_file in minimal_dir.glob("*.py"):
            if example_file.name.startswith("__"):
                continue
            
            content = example_file.read_text()
            
            # Should have a docstring or comment at the top
            lines = content.strip().split('\n')
            has_documentation = False
            
            for line in lines[:10]:  # Check first 10 lines
                if line.strip().startswith('"""') or line.strip().startswith('#'):
                    has_documentation = True
                    break
            
            assert has_documentation, f"Example {example_file.name} lacks documentation"
    
    def test_examples_are_executable(self, examples_dir):
        """Test that all Python examples are syntactically correct."""
        for example_file in examples_dir.rglob("*.py"):
            if example_file.name.startswith("__"):
                continue
            
            try:
                with open(example_file, 'r') as f:
                    compile(f.read(), str(example_file), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {example_file}: {e}")


# Utility function for running examples in CI/CD
def run_all_examples():
    """Utility function to run all examples (for CI/CD)."""
    examples_dir = Path(__file__).parent.parent.parent / "examples" / "minimal"
    
    results = {}
    
    for example_file in examples_dir.glob("*.py"):
        if example_file.name.startswith("__"):
            continue
        
        print(f"Running {example_file.name}...")
        
        try:
            result = subprocess.run(
                [sys.executable, str(example_file)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=example_file.parent.parent.parent
            )
            
            results[example_file.name] = {
                "success": result.returncode == 0,
                "output": result.stdout if result.returncode == 0 else result.stderr
            }
            
            status = "✅" if result.returncode == 0 else "❌"
            print(f"  {status} {example_file.name}")
            
        except Exception as e:
            results[example_file.name] = {
                "success": False,
                "output": str(e)
            }
            print(f"  ❌ {example_file.name} - {e}")
    
    return results


if __name__ == "__main__":
    # Allow running this file directly to test all examples
    results = run_all_examples()
    
    successful = sum(1 for r in results.values() if r["success"])
    total = len(results)
    
    print(f"\n📊 Results: {successful}/{total} examples passed")
    
    if successful < total:
        print("\n❌ Failed examples:")
        for name, result in results.items():
            if not result["success"]:
                print(f"  - {name}: {result['output']}")
        
        sys.exit(1)
    else:
        print("\n🎉 All examples passed!")
        sys.exit(0)
