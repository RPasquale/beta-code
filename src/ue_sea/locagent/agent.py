"""
LOCAGENT Agent: LLM-based code localization agent.

Implements the core agent loop with tool usage, reasoning, and training capabilities.
"""

import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

from .entities import Entity, EntityType, RelationType
from .tools import SearchEntity, TraverseGraph, RetrieveEntity, SearchResult, TraverseResult, RetrieveResult


class ActionType(Enum):
    """Types of actions the agent can take."""
    SEARCH_ENTITY = "search_entity"
    TRAVERSE_GRAPH = "traverse_graph"
    RETRIEVE_ENTITY = "retrieve_entity"
    STOP = "stop"


@dataclass
class AgentAction:
    """Represents an action taken by the agent."""
    action_type: ActionType
    parameters: Dict[str, Any]
    reasoning: str = ""
    confidence: float = 0.0


@dataclass
class AgentObservation:
    """Represents an observation from the environment."""
    action: AgentAction
    result: Union[SearchResult, TraverseResult, RetrieveResult, List[Any]]
    success: bool = True
    error_message: str = ""


@dataclass
class LocalizationResult:
    """Final result of the localization process."""
    issue_description: str
    relevant_entities: List[Entity]
    confidence_scores: Dict[str, float]
    reasoning_trace: List[AgentObservation]
    final_ranking: List[Tuple[str, float]]  # (entity_id, score)


class LOCAGENT_Agent:
    """
    LLM-based agent for code localization.
    
    Implements the core agent loop with tool usage, reasoning, and learning capabilities.
    """
    
    def __init__(self, graph, max_steps: int = 10, confidence_threshold: float = 0.8):
        self.graph = graph
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        
        # Initialize tools
        self.search_tool = SearchEntity(graph)
        self.traverse_tool = TraverseGraph(graph)
        self.retrieve_tool = RetrieveEntity(graph)
        
        # Agent state
        self.current_step = 0
        self.observations: List[AgentObservation] = []
        self.visited_entities: set = set()
        self.candidate_entities: Dict[str, float] = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    async def localize(self, issue_description: str, 
                      context: Optional[Dict[str, Any]] = None) -> LocalizationResult:
        """
        Main localization method.
        
        Args:
            issue_description: Natural language description of the issue
            context: Optional context information
            
        Returns:
            LocalizationResult with relevant entities and reasoning trace
        """
        self.logger.info(f"Starting localization for: {issue_description}")
        
        # Reset agent state
        self.current_step = 0
        self.observations = []
        self.visited_entities = set()
        self.candidate_entities = {}
        
        # Initial search based on issue keywords
        initial_keywords = self._extract_keywords(issue_description)
        await self._execute_initial_search(initial_keywords)
        
        # Main agent loop
        while self.current_step < self.max_steps:
            # Decide next action
            action = await self._decide_action(issue_description, context)
            
            if action.action_type == ActionType.STOP:
                break
            
            # Execute action
            observation = await self._execute_action(action)
            self.observations.append(observation)
            
            # Update agent state
            self._update_state(observation)
            
            # Check termination conditions
            if self._should_stop():
                break
            
            self.current_step += 1
        
        # Generate final result
        return await self._generate_final_result(issue_description)
    
    def _extract_keywords(self, issue_description: str) -> List[str]:
        """Extract keywords from issue description."""
        # Simple keyword extraction - can be enhanced with NLP
        import re
        
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        words = re.findall(r'\b\w+\b', issue_description.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    async def _execute_initial_search(self, keywords: List[str]) -> None:
        """Execute initial search based on keywords."""
        if not keywords:
            return
        
        action = AgentAction(
            action_type=ActionType.SEARCH_ENTITY,
            parameters={
                "keywords": keywords,
                "entity_types": [EntityType.FUNCTION, EntityType.CLASS, EntityType.FILE],
                "limit": 20
            },
            reasoning="Initial search based on issue keywords",
            confidence=0.9
        )
        
        observation = await self._execute_action(action)
        self.observations.append(observation)
        self._update_state(observation)
    
    async def _decide_action(self, issue_description: str, 
                           context: Optional[Dict[str, Any]] = None) -> AgentAction:
        """
        Enhanced decision making with multi-step reasoning and hypothesis validation.
        
        Uses advanced reasoning patterns:
        1. Hypothesis generation based on issue analysis
        2. Multi-step reasoning chains
        3. Evidence gathering and validation
        4. Adaptive search strategies
        """
        # Generate hypotheses about the issue
        hypotheses = self._generate_hypotheses(issue_description)
        
        # Multi-step reasoning: analyze current state and plan next steps
        reasoning_chain = self._build_reasoning_chain(issue_description, hypotheses)
        
        # Validate hypotheses based on current evidence
        validated_hypotheses = self._validate_hypotheses(hypotheses)
        
        # Decide action based on reasoning chain and validated hypotheses
        if self.current_step == 0:
            # Initial search with hypothesis-driven keywords
            keywords = self._extract_hypothesis_keywords(validated_hypotheses)
            return AgentAction(
                action_type=ActionType.SEARCH_ENTITY,
                parameters={
                    "keywords": keywords,
                    "entity_types": [EntityType.FUNCTION, EntityType.CLASS, EntityType.FILE],
                    "limit": 10
                },
                reasoning=f"Initial search based on hypotheses: {', '.join(validated_hypotheses[:3])}",
                confidence=0.9
            )
        
        elif len(self.candidate_entities) > 0 and self.current_step < self.max_steps - 3:
            # Multi-hop traversal with reasoning
            top_candidates = sorted(self.candidate_entities.items(), key=lambda x: x[1], reverse=True)[:3]
            start_ids = [entity_id for entity_id, _ in top_candidates]
            
            return AgentAction(
                action_type=ActionType.TRAVERSE_GRAPH,
                parameters={
                    "start_ids": start_ids,
                    "direction": "both",
                    "hops": 2,
                    "entity_types": [EntityType.FUNCTION, EntityType.CLASS],
                    "relation_types": ["invoke", "import", "inherit"]
                },
                reasoning=f"Multi-hop traversal from top candidates to gather evidence for hypotheses",
                confidence=0.8
            )
        
        elif len(self.visited_entities) > 0 and self.current_step < self.max_steps - 2:
            # Retrieve detailed information for hypothesis validation
            entity_ids = list(self.visited_entities)[:5]
            
            return AgentAction(
                action_type=ActionType.RETRIEVE_ENTITY,
                parameters={"entity_ids": entity_ids},
                reasoning="Retrieving detailed entity information for hypothesis validation",
                confidence=0.9
            )
        
        elif self.current_step < self.max_steps - 1:
            # Additional search based on reasoning chain
            additional_keywords = self._extract_additional_keywords(reasoning_chain)
            if additional_keywords:
                return AgentAction(
                    action_type=ActionType.SEARCH_ENTITY,
                    parameters={
                        "keywords": additional_keywords,
                        "entity_types": [EntityType.FUNCTION, EntityType.CLASS],
                        "limit": 5
                    },
                    reasoning="Additional search based on reasoning chain insights",
                    confidence=0.7
                )
        
        # Stop with confidence assessment
        final_confidence = self._assess_final_confidence(validated_hypotheses)
        return AgentAction(
            action_type=ActionType.STOP,
            parameters={},
            reasoning=f"Reasoning complete. Final confidence: {final_confidence:.2f}",
            confidence=final_confidence
        )
    
    async def _execute_action(self, action: AgentAction) -> AgentObservation:
        """Execute an action and return the observation."""
        try:
            if action.action_type == ActionType.SEARCH_ENTITY:
                result = await self.search_tool.search(
                    keywords=action.parameters["keywords"],
                    entity_types=action.parameters.get("entity_types"),
                    limit=action.parameters.get("limit", 10)
                )
                return AgentObservation(action=action, result=result, success=True)
            
            elif action.action_type == ActionType.TRAVERSE_GRAPH:
                result = await self.traverse_tool.traverse(
                    start_ids=action.parameters["start_ids"],
                    direction=action.parameters.get("direction", "both"),
                    hops=action.parameters.get("hops", 2),
                    entity_types=action.parameters.get("entity_types"),
                    relation_types=action.parameters.get("relation_types")
                )
                return AgentObservation(action=action, result=result, success=True)
            
            elif action.action_type == ActionType.RETRIEVE_ENTITY:
                result = await self.retrieve_tool.retrieve(
                    entity_ids=action.parameters["entity_ids"]
                )
                return AgentObservation(action=action, result=result, success=True)
            
            else:
                return AgentObservation(
                    action=action, 
                    result=[], 
                    success=False, 
                    error_message="Unknown action type"
                )
        
        except Exception as e:
            self.logger.error(f"Error executing action {action.action_type}: {e}")
            return AgentObservation(
                action=action,
                result=[],
                success=False,
                error_message=str(e)
            )
    
    def _generate_hypotheses(self, issue_description: str) -> List[str]:
        """Generate hypotheses about what the issue might involve."""
        # Simple hypothesis generation based on common patterns
        hypotheses = []
        
        # Extract key terms
        words = issue_description.lower().split()
        
        # Common issue patterns
        if any(word in words for word in ['error', 'exception', 'fail', 'crash']):
            hypotheses.extend(['error_handling', 'exception_management', 'logging'])
        
        if any(word in words for word in ['slow', 'performance', 'timeout', 'lag']):
            hypotheses.extend(['performance_optimization', 'caching', 'database_query'])
        
        if any(word in words for word in ['auth', 'login', 'security', 'permission']):
            hypotheses.extend(['authentication', 'authorization', 'security_middleware'])
        
        if any(word in words for word in ['api', 'endpoint', 'request', 'response']):
            hypotheses.extend(['api_handling', 'request_processing', 'response_formatting'])
        
        if any(word in words for word in ['database', 'query', 'sql', 'data']):
            hypotheses.extend(['database_operations', 'data_validation', 'query_optimization'])
        
        return hypotheses[:5]  # Limit to top 5 hypotheses
    
    def _build_reasoning_chain(self, issue_description: str, hypotheses: List[str]) -> List[str]:
        """Build a reasoning chain based on issue and hypotheses."""
        reasoning_chain = []
        
        # Step 1: Analyze issue type
        if 'error' in issue_description.lower():
            reasoning_chain.append("Issue involves error handling - need to find error sources and handlers")
        
        # Step 2: Consider related components
        for hypothesis in hypotheses[:3]:
            reasoning_chain.append(f"Hypothesis: {hypothesis} - search for related components")
        
        # Step 3: Plan investigation strategy
        reasoning_chain.append("Multi-hop traversal to find interconnected components")
        reasoning_chain.append("Validate findings with detailed entity retrieval")
        
        return reasoning_chain
    
    def _validate_hypotheses(self, hypotheses: List[str]) -> List[str]:
        """Validate hypotheses based on current evidence."""
        validated = []
        
        for hypothesis in hypotheses:
            # Simple validation based on current candidates
            if any(hypothesis.split('_')[0] in entity_id.lower() 
                   for entity_id in self.candidate_entities.keys()):
                validated.append(hypothesis)
        
        return validated if validated else hypotheses[:2]  # Fallback to top 2
    
    def _extract_hypothesis_keywords(self, hypotheses: List[str]) -> List[str]:
        """Extract keywords from validated hypotheses."""
        keywords = []
        for hypothesis in hypotheses:
            # Split hypothesis into words
            words = hypothesis.replace('_', ' ').split()
            keywords.extend(words)
        
        return keywords[:10]  # Limit keywords
    
    def _extract_additional_keywords(self, reasoning_chain: List[str]) -> List[str]:
        """Extract additional keywords from reasoning chain."""
        keywords = []
        for step in reasoning_chain:
            # Extract meaningful words
            words = step.lower().split()
            keywords.extend([w for w in words if len(w) > 3 and w not in ['need', 'find', 'search']])
        
        return keywords[:5]  # Limit additional keywords
    
    def _assess_final_confidence(self, validated_hypotheses: List[str]) -> float:
        """Assess final confidence based on evidence gathered."""
        if not self.candidate_entities:
            return 0.1
        
        # Base confidence on number of candidates and their scores
        num_candidates = len(self.candidate_entities)
        avg_score = sum(self.candidate_entities.values()) / num_candidates if num_candidates > 0 else 0
        
        # Higher confidence with more candidates and higher scores
        confidence = min(0.9, 0.3 + (num_candidates * 0.1) + (avg_score * 0.3))
        
        return confidence

    def _update_state(self, observation: AgentObservation) -> None:
        """Update agent state based on observation."""
        if not observation.success:
            return
        
        # Update visited entities
        if isinstance(observation.result, list):
            for item in observation.result:
                if hasattr(item, 'entity_id'):
                    self.visited_entities.add(item.entity_id)
                elif isinstance(item, SearchResult):
                    self.visited_entities.add(item.entity_id)
                    # Update candidate scores
                    self.candidate_entities[item.entity_id] = item.score
        
        # Update candidate entities from traverse results
        elif isinstance(observation.result, TraverseResult):
            for entity in observation.result.entities:
                self.visited_entities.add(entity.entity_id)
                # Give traversed entities a base score
                if entity.entity_id not in self.candidate_entities:
                    self.candidate_entities[entity.entity_id] = 0.5
    
    def _should_stop(self) -> bool:
        """Check if agent should stop."""
        # Stop if we have high-confidence candidates
        if self.candidate_entities:
            max_score = max(self.candidate_entities.values())
            if max_score >= self.confidence_threshold:
                return True
        
        # Stop if we've reached max steps
        if self.current_step >= self.max_steps:
            return True
        
        return False
    
    async def _generate_final_result(self, issue_description: str) -> LocalizationResult:
        """Generate final localization result."""
        # Get relevant entities
        relevant_entities = []
        for entity_id in self.candidate_entities:
            entity = self.graph.get_entity(entity_id)
            if entity:
                relevant_entities.append(entity)
        
        # Sort by confidence score
        relevant_entities.sort(
            key=lambda e: self.candidate_entities.get(e.entity_id, 0.0), 
            reverse=True
        )
        
        # Create final ranking
        final_ranking = [
            (entity.entity_id, self.candidate_entities.get(entity.entity_id, 0.0))
            for entity in relevant_entities
        ]
        
        return LocalizationResult(
            issue_description=issue_description,
            relevant_entities=relevant_entities,
            confidence_scores=self.candidate_entities.copy(),
            reasoning_trace=self.observations.copy(),
            final_ranking=final_ranking
        )
    
    def get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state for debugging/monitoring."""
        return {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "visited_entities": list(self.visited_entities),
            "candidate_entities": self.candidate_entities.copy(),
            "num_observations": len(self.observations)
        }
