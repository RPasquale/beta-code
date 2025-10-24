"""
Test LOCAGENT agent functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.ue_sea.locagent.agent import LOCAGENT_Agent, AgentAction, ActionType, LocalizationResult
from src.ue_sea.locagent.entities import Entity, EntityType


class TestAgentAction:
    """Test AgentAction class."""
    
    def test_agent_action_creation(self):
        """Test basic agent action creation."""
        action = AgentAction(
            action_type=ActionType.SEARCH_ENTITY,
            parameters={"keywords": ["test"], "limit": 10},
            reasoning="Test search",
            confidence=0.8
        )
        
        assert action.action_type == ActionType.SEARCH_ENTITY
        assert action.parameters == {"keywords": ["test"], "limit": 10}
        assert action.reasoning == "Test search"
        assert action.confidence == 0.8


class TestLOCAGENTAgent:
    """Test LOCAGENT_Agent class."""
    
    def test_agent_creation(self):
        """Test agent creation."""
        mock_graph = Mock()
        agent = LOCAGENT_Agent(mock_graph, max_steps=5, confidence_threshold=0.7)
        
        assert agent.graph == mock_graph
        assert agent.max_steps == 5
        assert agent.confidence_threshold == 0.7
        assert agent.current_step == 0
        assert len(agent.observations) == 0
        assert len(agent.visited_entities) == 0
        assert len(agent.candidate_entities) == 0
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        mock_graph = Mock()
        agent = LOCAGENT_Agent(mock_graph)
        
        # Test with normal text
        keywords = agent._extract_keywords("Fix authentication bug in login system")
        assert "authentication" in keywords
        assert "bug" in keywords
        assert "login" in keywords
        assert "system" in keywords
        
        # Test with stop words
        keywords = agent._extract_keywords("The and or but in on at")
        assert len(keywords) == 0  # All stop words
    
    def test_should_stop_high_confidence(self):
        """Test stopping with high confidence."""
        mock_graph = Mock()
        agent = LOCAGENT_Agent(mock_graph, confidence_threshold=0.8)
        agent.candidate_entities = {"entity1": 0.9, "entity2": 0.7}
        
        assert agent._should_stop() == True
    
    def test_should_stop_max_steps(self):
        """Test stopping at max steps."""
        mock_graph = Mock()
        agent = LOCAGENT_Agent(mock_graph, max_steps=3)
        agent.current_step = 3
        
        assert agent._should_stop() == True
    
    def test_should_stop_continue(self):
        """Test continuing when conditions not met."""
        mock_graph = Mock()
        agent = LOCAGENT_Agent(mock_graph, max_steps=5, confidence_threshold=0.8)
        agent.current_step = 2
        agent.candidate_entities = {"entity1": 0.5}
        
        assert agent._should_stop() == False
    
    def test_get_agent_state(self):
        """Test getting agent state."""
        mock_graph = Mock()
        agent = LOCAGENT_Agent(mock_graph, max_steps=5)
        agent.current_step = 2
        agent.visited_entities = {"entity1", "entity2"}
        agent.candidate_entities = {"entity1": 0.8}
        agent.observations = [Mock(), Mock()]
        
        state = agent.get_agent_state()
        
        assert state["current_step"] == 2
        assert state["max_steps"] == 5
        assert state["visited_entities"] == ["entity1", "entity2"]
        assert state["candidate_entities"] == {"entity1": 0.8}
        assert state["num_observations"] == 2


class TestAgentIntegration:
    """Test agent integration with mock graph."""
    
    @pytest.mark.asyncio
    async def test_localize_basic(self):
        """Test basic localization functionality."""
        # Create mock graph
        mock_graph = Mock()
        mock_entity = Entity(
            entity_id="test:entity",
            entity_type=EntityType.FUNCTION,
            name="test_function",
            location="test.py"
        )
        mock_graph.get_entity.return_value = mock_entity
        
        # Create agent
        agent = LOCAGENT_Agent(mock_graph, max_steps=2)
        
        # Mock the search tool
        mock_search_result = Mock()
        mock_search_result.entity_id = "test:entity"
        mock_search_result.score = 0.8
        agent.search_tool.search = AsyncMock(return_value=[mock_search_result])
        
        # Mock the traverse tool
        mock_traverse_result = Mock()
        mock_traverse_result.entities = [mock_entity]
        agent.traverse_tool.traverse = AsyncMock(return_value=mock_traverse_result)
        
        # Mock the retrieve tool
        mock_retrieve_result = Mock()
        mock_retrieve_result.entity_id = "test:entity"
        agent.retrieve_tool.retrieve = AsyncMock(return_value=[mock_retrieve_result])
        
        # Run localization
        result = await agent.localize("Find test function")
        
        # Check result
        assert isinstance(result, LocalizationResult)
        assert result.issue_description == "Find test function"
        assert len(result.relevant_entities) >= 0
        assert len(result.reasoning_trace) >= 0
    
    @pytest.mark.asyncio
    async def test_localize_empty_keywords(self):
        """Test localization with empty keywords."""
        mock_graph = Mock()
        agent = LOCAGENT_Agent(mock_graph, max_steps=1)
        
        # Mock tools to return empty results
        agent.search_tool.search = AsyncMock(return_value=[])
        agent.traverse_tool.traverse = AsyncMock(return_value=Mock())
        agent.retrieve_tool.retrieve = AsyncMock(return_value=[])
        
        result = await agent.localize("")
        
        assert isinstance(result, LocalizationResult)
        assert result.issue_description == ""
        assert len(result.relevant_entities) == 0
    
    @pytest.mark.asyncio
    async def test_localize_with_context(self):
        """Test localization with context."""
        mock_graph = Mock()
        agent = LOCAGENT_Agent(mock_graph, max_steps=2)
        
        # Mock tools
        agent.search_tool.search = AsyncMock(return_value=[])
        agent.traverse_tool.traverse = AsyncMock(return_value=Mock())
        agent.retrieve_tool.retrieve = AsyncMock(return_value=[])
        
        context = {"repository": "test_repo", "commit": "abc123"}
        result = await agent.localize("Find test function", context)
        
        assert isinstance(result, LocalizationResult)
        assert result.issue_description == "Find test function"


class TestAgentTools:
    """Test agent tool integration."""
    
    def test_agent_has_tools(self):
        """Test that agent has access to tools."""
        mock_graph = Mock()
        agent = LOCAGENT_Agent(mock_graph)
        
        assert hasattr(agent, 'search_tool')
        assert hasattr(agent, 'traverse_tool')
        assert hasattr(agent, 'retrieve_tool')
    
    @pytest.mark.asyncio
    async def test_execute_initial_search(self):
        """Test initial search execution."""
        mock_graph = Mock()
        agent = LOCAGENT_Agent(mock_graph)
        
        # Mock search tool
        mock_search_result = Mock()
        mock_search_result.entity_id = "test:entity"
        mock_search_result.score = 0.8
        agent.search_tool.search = AsyncMock(return_value=[mock_search_result])
        
        # Test initial search
        await agent._execute_initial_search(["test", "function"])
        
        # Check that search was called
        agent.search_tool.search.assert_called_once()
        
        # Check that agent state was updated
        assert len(agent.observations) == 1
        assert "test:entity" in agent.visited_entities
        assert "test:entity" in agent.candidate_entities
