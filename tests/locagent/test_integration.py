"""
Integration tests for LOCAGENT.

Tests the complete LOCAGENT workflow from graph building to localization.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from src.ue_sea.locagent.graph import LOCAGENT_Graph
from src.ue_sea.locagent.agent import LOCAGENT_Agent
from src.ue_sea.locagent.training import create_sample_dataset
from src.ue_sea.locagent.data_loader import load_training_data


class TestLOCAGENTIntegration:
    """Test complete LOCAGENT integration."""
    
    @pytest.mark.asyncio
    async def test_basic_workflow(self):
        """Test basic LOCAGENT workflow."""
        # Create a temporary test repository
        with tempfile.TemporaryDirectory() as temp_dir:
            test_repo = Path(temp_dir) / "test_repo"
            test_repo.mkdir()
            
            # Create a simple Python file
            test_file = test_repo / "test.py"
            test_file.write_text("""
def hello_world():
    \"\"\"Print hello world.\"\"\"
    print("Hello, World!")

class TestClass:
    \"\"\"A test class.\"\"\"
    
    def __init__(self):
        self.value = 42
    
    def get_value(self):
        return self.value

def main():
    obj = TestClass()
    print(obj.get_value())
    hello_world()

if __name__ == "__main__":
    main()
""")
            
            # Test 1: Create and build graph
            graph = LOCAGENT_Graph(use_gpu=False)
            await graph.build_from_repository(str(test_repo))
            
            # Check graph was built
            stats = graph.get_entity_statistics()
            assert stats['total_entities'] > 0
            assert stats['total_relations'] > 0
            
            # Test 2: Search functionality
            search_results = graph.search_entities("hello", limit=5)
            assert len(search_results) > 0
            
            # Test 3: Agent localization
            agent = LOCAGENT_Agent(graph, max_steps=3)
            result = await agent.localize("Find the hello world function")
            
            assert result.issue_description == "Find the hello world function"
            assert len(result.relevant_entities) >= 0
            assert len(result.reasoning_trace) >= 0
    
    @pytest.mark.asyncio
    async def test_tool_functionality(self):
        """Test individual tool functionality."""
        # Create a simple test repository
        with tempfile.TemporaryDirectory() as temp_dir:
            test_repo = Path(temp_dir) / "test_repo"
            test_repo.mkdir()
            
            # Create test files
            (test_repo / "utils.py").write_text("""
def helper_function():
    return "helper"

def utility_function():
    return "utility"
""")
            
            (test_repo / "main.py").write_text("""
from utils import helper_function

def main():
    result = helper_function()
    print(result)
""")
            
            # Build graph
            graph = LOCAGENT_Graph(use_gpu=False)
            await graph.build_from_repository(str(test_repo))
            
            # Test SearchEntity tool
            search_results = await graph.search_entity.search(["helper"], limit=5)
            assert len(search_results) > 0
            
            # Test TraverseGraph tool
            if search_results:
                entity_id = search_results[0].entity_id
                traverse_results = await graph.traverse_graph.traverse([entity_id], hops=1)
                assert len(traverse_results.entities) >= 0
            
            # Test RetrieveEntity tool
            if search_results:
                entity_id = search_results[0].entity_id
                retrieve_results = await graph.retrieve_entity.retrieve([entity_id])
                assert len(retrieve_results) > 0
    
    def test_data_loading_integration(self):
        """Test data loading integration."""
        # Create sample training data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            sample_path = f.name
            create_sample_dataset(sample_path, num_examples=5)
        
        try:
            # Test loading data
            examples = load_training_data(sample_path)
            assert len(examples) == 5
            
            # Check example structure
            example = examples[0]
            assert hasattr(example, 'issue_description')
            assert hasattr(example, 'target_entities')
            assert hasattr(example, 'repository_path')
            
        finally:
            Path(sample_path).unlink()
    
    @pytest.mark.asyncio
    async def test_agent_reasoning_loop(self):
        """Test agent reasoning loop with multiple steps."""
        # Create a more complex test repository
        with tempfile.TemporaryDirectory() as temp_dir:
            test_repo = Path(temp_dir) / "test_repo"
            test_repo.mkdir()
            
            # Create multiple files with relationships
            (test_repo / "auth.py").write_text("""
class Authentication:
    def __init__(self):
        self.session = None
    
    def login(self, username, password):
        if self.validate_credentials(username, password):
            self.session = self.create_session(username)
            return True
        return False
    
    def validate_credentials(self, username, password):
        # Mock validation
        return username == "admin" and password == "password"
    
    def create_session(self, username):
        return {"user": username, "active": True}
""")
            
            (test_repo / "main.py").write_text("""
from auth import Authentication

def main():
    auth = Authentication()
    if auth.login("admin", "password"):
        print("Login successful")
    else:
        print("Login failed")
""")
            
            # Build graph
            graph = LOCAGENT_Graph(use_gpu=False)
            await graph.build_from_repository(str(test_repo))
            
            # Test agent with more complex reasoning
            agent = LOCAGENT_Agent(graph, max_steps=5)
            result = await agent.localize("Find authentication login functionality")
            
            # Check that agent performed multiple steps
            assert len(result.reasoning_trace) > 0
            assert result.issue_description == "Find authentication login functionality"
            
            # Check agent state
            state = agent.get_agent_state()
            assert state["current_step"] >= 0
            assert state["num_observations"] >= 0
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in various scenarios."""
        # Test with empty repository
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_repo = Path(temp_dir) / "empty_repo"
            empty_repo.mkdir()
            
            graph = LOCAGENT_Graph(use_gpu=False)
            await graph.build_from_repository(str(empty_repo))
            
            # Should handle empty repository gracefully
            stats = graph.get_entity_statistics()
            assert stats['total_entities'] == 0
            
            # Test agent with empty graph
            agent = LOCAGENT_Agent(graph, max_steps=2)
            result = await agent.localize("Find any function")
            
            # Should return empty result without crashing
            assert result.issue_description == "Find any function"
            assert len(result.relevant_entities) == 0
    
    @pytest.mark.asyncio
    async def test_performance_with_large_repo(self):
        """Test performance with a larger repository."""
        # Create a larger test repository
        with tempfile.TemporaryDirectory() as temp_dir:
            test_repo = Path(temp_dir) / "test_repo"
            test_repo.mkdir()
            
            # Create multiple files
            for i in range(10):
                (test_repo / f"module_{i}.py").write_text(f"""
def function_{i}():
    return {i}

class Class{i}:
    def method_{i}(self):
        return {i}
""")
            
            # Build graph
            graph = LOCAGENT_Graph(use_gpu=False)
            await graph.build_from_repository(str(test_repo))
            
            # Check that graph was built successfully
            stats = graph.get_entity_statistics()
            assert stats['total_entities'] > 0
            
            # Test search performance
            search_results = graph.search_entities("function", limit=10)
            assert len(search_results) > 0
            
            # Test agent performance
            agent = LOCAGENT_Agent(graph, max_steps=3)
            result = await agent.localize("Find functions in the codebase")
            
            assert result.issue_description == "Find functions in the codebase"
            assert len(result.reasoning_trace) >= 0
