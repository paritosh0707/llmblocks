#!/usr/bin/env python3
"""
Rigorous Memory System Testing

This script performs comprehensive testing of the LLMBlocks memory system
to ensure reliability, performance, and correctness.

Test Categories:
- Basic functionality
- Edge cases and error handling
- Performance under load
- Persistence and recovery
- LangChain compatibility
- Multi-session management
"""

import asyncio
import os
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(project_root))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class MemoryTestSuite:
    """Comprehensive test suite for memory systems."""
    
    def __init__(self):
        self.test_results: Dict[str, Dict[str, Any]] = {}
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
    
    def log_test_result(self, test_name: str, passed: bool, details: str = "", duration: float = 0):
        """Log a test result."""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            self.failed_tests += 1
            status = "❌ FAIL"
        
        self.test_results[test_name] = {
            "passed": passed,
            "details": details,
            "duration": duration,
            "status": status
        }
        
        print(f"{status} {test_name} ({duration:.3f}s)")
        if details:
            print(f"    {details}")
    
    async def test_basic_functionality(self):
        """Test basic memory operations."""
        print("\n🔧 Testing Basic Functionality")
        print("-" * 40)
        
        try:
            from llmblocks.blocks.memory import get_conversation_memory
            
            # Test 1: Memory creation
            start_time = time.time()
            memory = await get_conversation_memory(
                backend_type="in_memory",
                session_id="test_basic"
            )
            duration = time.time() - start_time
            self.log_test_result("memory_creation", True, "In-memory backend", duration)
            
            # Test 2: Add messages
            start_time = time.time()
            msg1 = await memory.add_user_message("Hello, world!")
            msg2 = await memory.add_assistant_message("Hi there!")
            duration = time.time() - start_time
            
            assert msg1.content == "Hello, world!"
            assert msg2.content == "Hi there!"
            self.log_test_result("add_messages", True, "2 messages added", duration)
            
            # Test 3: Retrieve messages
            start_time = time.time()
            history = await memory.get_conversation_history()
            duration = time.time() - start_time
            
            assert len(history) == 2
            assert history[0].content == "Hello, world!"
            assert history[1].content == "Hi there!"
            self.log_test_result("retrieve_messages", True, f"{len(history)} messages retrieved", duration)
            
            # Test 4: Context window
            start_time = time.time()
            context = await memory.get_context_window()
            duration = time.time() - start_time
            
            assert len(context) == 2
            self.log_test_result("context_window", True, f"{len(context)} messages in context", duration)
            
            # Test 5: Search functionality
            start_time = time.time()
            results = await memory.search_messages("Hello")
            duration = time.time() - start_time
            
            assert len(results) == 1
            assert "Hello" in results[0].content
            self.log_test_result("search_messages", True, f"Found {len(results)} results", duration)
            
            # Test 6: Statistics
            start_time = time.time()
            stats = await memory.get_conversation_stats()
            duration = time.time() - start_time
            
            assert stats["total_messages"] == 2
            assert stats["exists"] == True
            self.log_test_result("get_statistics", True, f"Stats: {stats['total_messages']} messages", duration)
            
            # Test 7: Clear memory
            start_time = time.time()
            cleared_count = await memory.clear_memory()
            duration = time.time() - start_time
            
            assert cleared_count == 2
            history_after_clear = await memory.get_conversation_history()
            assert len(history_after_clear) == 0
            self.log_test_result("clear_memory", True, f"Cleared {cleared_count} messages", duration)
            
            await memory.close()
            
        except Exception as e:
            self.log_test_result("basic_functionality", False, f"Error: {e}")
    
    async def test_edge_cases(self):
        """Test edge cases and error handling."""
        print("\n⚠️  Testing Edge Cases")
        print("-" * 40)
        
        try:
            from llmblocks.blocks.memory import get_conversation_memory
            
            # Test 1: Empty messages
            memory = await get_conversation_memory(backend_type="in_memory")
            
            start_time = time.time()
            try:
                await memory.add_user_message("")  # Empty message
                empty_history = await memory.get_conversation_history()
                duration = time.time() - start_time
                self.log_test_result("empty_message", True, "Empty message handled", duration)
            except Exception as e:
                duration = time.time() - start_time
                self.log_test_result("empty_message", False, f"Failed: {e}", duration)
            
            # Test 2: Very long messages
            start_time = time.time()
            long_message = "A" * 10000  # 10KB message
            await memory.add_user_message(long_message)
            duration = time.time() - start_time
            self.log_test_result("long_message", True, f"10KB message stored", duration)
            
            # Test 3: Special characters
            start_time = time.time()
            special_msg = "Hello! 🚀 Testing émojis and spëcial chars: @#$%^&*()"
            await memory.add_user_message(special_msg)
            history = await memory.get_conversation_history()
            found_special = any(special_msg in msg.content for msg in history)
            duration = time.time() - start_time
            self.log_test_result("special_characters", found_special, "Unicode and special chars", duration)
            
            # Test 4: Rapid message addition
            start_time = time.time()
            tasks = []
            for i in range(100):
                tasks.append(memory.add_user_message(f"Rapid message {i}"))
            
            await asyncio.gather(*tasks)
            rapid_history = await memory.get_conversation_history()
            duration = time.time() - start_time
            
            rapid_count = len([m for m in rapid_history if "Rapid message" in m.content])
            self.log_test_result("rapid_messages", rapid_count == 100, f"{rapid_count}/100 messages", duration)
            
            await memory.close()
            
        except Exception as e:
            self.log_test_result("edge_cases", False, f"Error: {e}")
    
    async def test_persistence(self):
        """Test persistence across sessions."""
        print("\n💾 Testing Persistence")
        print("-" * 40)
        
        try:
            from llmblocks.blocks.memory import get_conversation_memory
            
            with tempfile.TemporaryDirectory() as temp_dir:
                session_id = "persistence_test"
                
                # Test 1: Save data
                start_time = time.time()
                memory1 = await get_conversation_memory(
                    backend_type="file",
                    session_id=session_id,
                    backend_config={"storage_dir": temp_dir}
                )
                
                await memory1.add_user_message("Persistent message 1")
                await memory1.add_assistant_message("Persistent response 1")
                await memory1.add_user_message("Persistent message 2")
                
                await memory1.close()
                duration = time.time() - start_time
                self.log_test_result("save_persistent_data", True, "3 messages saved", duration)
                
                # Test 2: Load data
                start_time = time.time()
                memory2 = await get_conversation_memory(
                    backend_type="file",
                    session_id=session_id,
                    backend_config={"storage_dir": temp_dir}
                )
                
                loaded_history = await memory2.get_conversation_history()
                duration = time.time() - start_time
                
                assert len(loaded_history) == 3
                assert "Persistent message 1" in loaded_history[0].content
                assert "Persistent response 1" in loaded_history[1].content
                assert "Persistent message 2" in loaded_history[2].content
                
                self.log_test_result("load_persistent_data", True, f"Loaded {len(loaded_history)} messages", duration)
                
                # Test 3: Append to existing data
                start_time = time.time()
                await memory2.add_user_message("Additional message")
                updated_history = await memory2.get_conversation_history()
                duration = time.time() - start_time
                
                assert len(updated_history) == 4
                self.log_test_result("append_persistent_data", True, f"Total: {len(updated_history)} messages", duration)
                
                await memory2.close()
                
        except Exception as e:
            self.log_test_result("persistence", False, f"Error: {e}")
    
    async def test_context_strategies(self):
        """Test different context management strategies."""
        print("\n🪟 Testing Context Strategies")
        print("-" * 40)
        
        try:
            from llmblocks.blocks.memory import get_conversation_memory
            
            strategies = ["sliding_window", "summarized", "priority_based"]
            
            for strategy in strategies:
                start_time = time.time()
                
                memory = await get_conversation_memory(
                    backend_type="in_memory",
                    session_id=f"strategy_{strategy}",
                    context_strategy=strategy,
                    max_context_messages=5
                )
                
                # Add more messages than the context window
                for i in range(10):
                    await memory.add_user_message(f"Message {i}")
                    await memory.add_assistant_message(f"Response {i}")
                
                context = await memory.get_context_window()
                duration = time.time() - start_time
                
                # Context should be limited by max_context_messages
                success = len(context) <= 5
                self.log_test_result(
                    f"context_strategy_{strategy}", 
                    success, 
                    f"Context: {len(context)}/20 messages", 
                    duration
                )
                
                await memory.close()
                
        except Exception as e:
            self.log_test_result("context_strategies", False, f"Error: {e}")
    
    async def test_multi_session(self):
        """Test multi-session management."""
        print("\n👥 Testing Multi-Session Management")
        print("-" * 40)
        
        try:
            from llmblocks.blocks.memory import get_conversation_memory
            
            # Test 1: Multiple independent sessions
            start_time = time.time()
            
            sessions = {}
            for i in range(5):
                session_id = f"session_{i}"
                memory = await get_conversation_memory(
                    backend_type="in_memory",
                    session_id=session_id
                )
                
                await memory.add_user_message(f"Message from session {i}")
                sessions[session_id] = memory
            
            duration = time.time() - start_time
            self.log_test_result("create_multiple_sessions", True, f"Created {len(sessions)} sessions", duration)
            
            # Test 2: Verify session isolation
            start_time = time.time()
            isolation_success = True
            
            for session_id, memory in sessions.items():
                history = await memory.get_conversation_history()
                if len(history) != 1:
                    isolation_success = False
                    break
                
                expected_content = f"Message from session {session_id.split('_')[1]}"
                if expected_content not in history[0].content:
                    isolation_success = False
                    break
            
            duration = time.time() - start_time
            self.log_test_result("session_isolation", isolation_success, "Sessions are isolated", duration)
            
            # Cleanup
            for memory in sessions.values():
                await memory.close()
                
        except Exception as e:
            self.log_test_result("multi_session", False, f"Error: {e}")
    
    async def test_performance(self):
        """Test performance under load."""
        print("\n⚡ Testing Performance")
        print("-" * 40)
        
        try:
            from llmblocks.blocks.memory import get_conversation_memory
            
            memory = await get_conversation_memory(backend_type="in_memory")
            
            # Test 1: Bulk message insertion
            start_time = time.time()
            message_count = 1000
            
            for i in range(message_count):
                await memory.add_user_message(f"Performance test message {i}")
            
            duration = time.time() - start_time
            messages_per_second = message_count / duration
            
            self.log_test_result(
                "bulk_insertion", 
                True, 
                f"{messages_per_second:.0f} messages/sec", 
                duration
            )
            
            # Test 2: Bulk retrieval
            start_time = time.time()
            history = await memory.get_conversation_history()
            duration = time.time() - start_time
            
            retrieval_rate = len(history) / duration
            self.log_test_result(
                "bulk_retrieval", 
                len(history) == message_count, 
                f"{retrieval_rate:.0f} messages/sec", 
                duration
            )
            
            # Test 3: Search performance
            start_time = time.time()
            search_results = await memory.search_messages("test", limit=100)
            duration = time.time() - start_time
            
            self.log_test_result(
                "search_performance", 
                len(search_results) > 0, 
                f"Found {len(search_results)} results", 
                duration
            )
            
            await memory.close()
            
        except Exception as e:
            self.log_test_result("performance", False, f"Error: {e}")
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("🧪 MEMORY SYSTEM TEST SUMMARY")
        print("="*60)
        
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests)*100:.1f}%")
        
        if self.failed_tests > 0:
            print("\n❌ Failed Tests:")
            for test_name, result in self.test_results.items():
                if not result["passed"]:
                    print(f"  - {test_name}: {result['details']}")
        
        print(f"\n⏱️  Performance Summary:")
        perf_tests = {k: v for k, v in self.test_results.items() 
                     if "performance" in k or "bulk" in k or "rapid" in k}
        
        for test_name, result in perf_tests.items():
            print(f"  - {test_name}: {result['details']} ({result['duration']:.3f}s)")
        
        overall_status = "🎉 ALL TESTS PASSED!" if self.failed_tests == 0 else "⚠️  SOME TESTS FAILED"
        print(f"\n{overall_status}")


async def run_rigorous_tests():
    """Run all rigorous memory tests."""
    print("🚀 LLMBlocks Memory System - Rigorous Testing")
    print("="*60)
    
    test_suite = MemoryTestSuite()
    
    # Run all test categories
    await test_suite.test_basic_functionality()
    await test_suite.test_edge_cases()
    await test_suite.test_persistence()
    await test_suite.test_context_strategies()
    await test_suite.test_multi_session()
    await test_suite.test_performance()
    
    # Print final summary
    test_suite.print_summary()
    
    return test_suite.failed_tests == 0


if __name__ == "__main__":
    success = asyncio.run(run_rigorous_tests())
    sys.exit(0 if success else 1)
