"""
DiPaCo: Distributed Path Composition for modular training.

Implements coarse routing, discriminative re-sharding, and path composition.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


class RoutingStrategy(Enum):
    """Routing strategies for DiPaCo."""
    GENERATIVE = "generative"      # K-means clustering
    DISCRIMINATIVE = "discriminative"  # Classifier-based
    HYBRID = "hybrid"             # Combination of both


@dataclass
class PathConfig:
    """Configuration for a DiPaCo path."""
    path_id: str
    modules: List[str]  # Module IDs in this path
    shared_modules: List[str]  # Shared module IDs
    unshared_modules: List[str]  # Unshared module IDs
    routing_key: str  # Key for routing decisions
    priority: float = 1.0  # Path priority
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    path_id: str
    confidence: float
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class PathPerformance:
    """Performance metrics for a path."""
    path_id: str
    accuracy: float
    latency: float
    throughput: float
    resource_usage: float
    last_updated: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DiPaCo_Router:
    """
    DiPaCo Router for distributed path composition.
    
    Implements coarse routing, discriminative re-sharding,
    and path composition for modular training.
    """
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.HYBRID):
        self.strategy = strategy
        self.paths = {}
        self.path_performance = {}
        self.routing_models = {}
        self.shard_assignments = {}
        
        # Routing components
        self.generative_router = None
        self.discriminative_router = None
        self.re_sharding_engine = None
        
        # Statistics
        self.routing_statistics = {
            "total_routes": 0,
            "successful_routes": 0,
            "re_sharding_events": 0,
            "average_latency": 0.0
        }
    
    async def initialize(self, paths: List[PathConfig]) -> None:
        """
        Initialize the DiPaCo router.
        
        Args:
            paths: List of path configurations
        """
        print("Initializing DiPaCo Router...")
        
        # Initialize paths
        for path_config in paths:
            self.paths[path_config.path_id] = path_config
            self.path_performance[path_config.path_id] = PathPerformance(
                path_id=path_config.path_id,
                accuracy=0.5,  # Initial accuracy
                latency=100.0,  # Initial latency
                throughput=1.0,  # Initial throughput
                resource_usage=0.5,  # Initial resource usage
                last_updated=time.time()
            )
        
        # Initialize routing components
        await self._initialize_routing_components()
        
        print(f"DiPaCo Router initialized with {len(paths)} paths")
    
    async def _initialize_routing_components(self) -> None:
        """Initialize routing components based on strategy."""
        
        if self.strategy in [RoutingStrategy.GENERATIVE, RoutingStrategy.HYBRID]:
            # Initialize generative router (K-means)
            self.generative_router = KMeans(n_clusters=4, random_state=42)
            print("  Generative router (K-means) initialized")
        
        if self.strategy in [RoutingStrategy.DISCRIMINATIVE, RoutingStrategy.HYBRID]:
            # Initialize discriminative router (Random Forest)
            self.discriminative_router = RandomForestClassifier(
                n_estimators=100, random_state=42
            )
            print("  Discriminative router (Random Forest) initialized")
        
        # Initialize re-sharding engine
        self.re_sharding_engine = ReShardingEngine()
        await self.re_sharding_engine.initialize()
        print("  Re-sharding engine initialized")
    
    async def route_request(self, request_features: Dict[str, Any]) -> RoutingDecision:
        """
        Route a request to the best path.
        
        Args:
            request_features: Features of the request
        
        Returns:
            Routing decision
        """
        # Extract routing features
        routing_vector = self._extract_routing_features(request_features)
        
        # Get routing decision based on strategy
        if self.strategy == RoutingStrategy.GENERATIVE:
            path_id = await self._generative_route(routing_vector)
        elif self.strategy == RoutingStrategy.DISCRIMINATIVE:
            path_id = await self._discriminative_route(routing_vector)
        else:  # HYBRID
            path_id = await self._hybrid_route(routing_vector)
        
        # Calculate confidence
        confidence = self._calculate_confidence(path_id, routing_vector)
        
        # Update statistics
        self.routing_statistics["total_routes"] += 1
        if confidence > 0.7:
            self.routing_statistics["successful_routes"] += 1
        
        return RoutingDecision(
            path_id=path_id,
            confidence=confidence,
            reasoning=f"Routed to {path_id} with {confidence:.2f} confidence",
            metadata={"routing_vector": routing_vector}
        )
    
    async def _generative_route(self, routing_vector: np.ndarray) -> str:
        """Route using generative approach (K-means clustering)."""
        if self.generative_router is None:
            # Fallback to random selection
            return list(self.paths.keys())[0]
        
        # Cluster the routing vector
        cluster_id = self.generative_router.predict([routing_vector])[0]
        
        # Map cluster to path (simplified)
        path_ids = list(self.paths.keys())
        path_id = path_ids[cluster_id % len(path_ids)]
        
        return path_id
    
    async def _discriminative_route(self, routing_vector: np.ndarray) -> str:
        """Route using discriminative approach (classifier)."""
        if self.discriminative_router is None:
            # Fallback to random selection
            return list(self.paths.keys())[0]
        
        # Predict best path
        path_id = self.discriminative_router.predict([routing_vector])[0]
        
        return path_id
    
    async def _hybrid_route(self, routing_vector: np.ndarray) -> str:
        """Route using hybrid approach."""
        # Get predictions from both routers
        generative_path = await self._generative_route(routing_vector)
        discriminative_path = await self._discriminative_route(routing_vector)
        
        # Combine predictions (weighted average)
        if generative_path == discriminative_path:
            return generative_path
        else:
            # Choose based on path performance
            gen_perf = self.path_performance[generative_path].accuracy
            disc_perf = self.path_performance[discriminative_path].accuracy
            
            return generative_path if gen_perf > disc_perf else discriminative_path
    
    def _extract_routing_features(self, request_features: Dict[str, Any]) -> np.ndarray:
        """Extract routing features from request."""
        # Convert request features to numerical vector
        features = []
        
        # Add basic features
        features.append(request_features.get("complexity", 0.5))
        features.append(request_features.get("urgency", 0.5))
        features.append(request_features.get("resource_requirements", 0.5))
        features.append(request_features.get("latency_requirements", 0.5))
        
        # Add derived features
        features.append(len(request_features.get("input_data", "")))
        features.append(request_features.get("batch_size", 1))
        
        return np.array(features)
    
    def _calculate_confidence(self, path_id: str, routing_vector: np.ndarray) -> float:
        """Calculate confidence for routing decision."""
        path_perf = self.path_performance[path_id]
        
        # Base confidence on path performance
        base_confidence = path_perf.accuracy
        
        # Adjust based on routing vector characteristics
        complexity = routing_vector[0] if len(routing_vector) > 0 else 0.5
        urgency = routing_vector[1] if len(routing_vector) > 1 else 0.5
        
        # Higher complexity and urgency reduce confidence
        confidence = base_confidence * (1.0 - complexity * 0.2) * (1.0 - urgency * 0.1)
        
        return max(0.1, min(1.0, confidence))
    
    async def update_path_performance(self, path_id: str, 
                                    performance_metrics: Dict[str, float]) -> None:
        """Update performance metrics for a path."""
        if path_id not in self.path_performance:
            return
        
        path_perf = self.path_performance[path_id]
        
        # Update metrics
        path_perf.accuracy = performance_metrics.get("accuracy", path_perf.accuracy)
        path_perf.latency = performance_metrics.get("latency", path_perf.latency)
        path_perf.throughput = performance_metrics.get("throughput", path_perf.throughput)
        path_perf.resource_usage = performance_metrics.get("resource_usage", path_perf.resource_usage)
        path_perf.last_updated = time.time()
        
        # Update routing models if needed
        await self._update_routing_models()
    
    async def _update_routing_models(self) -> None:
        """Update routing models based on performance data."""
        # Collect training data
        training_data = []
        labels = []
        
        for path_id, path_perf in self.path_performance.items():
            # Create feature vector from path characteristics
            path_config = self.paths[path_id]
            features = [
                len(path_config.modules),
                len(path_config.shared_modules),
                path_config.priority
            ]
            
            training_data.append(features)
            labels.append(path_id)
        
        if len(training_data) < 2:
            return
        
        # Update discriminative router
        if self.discriminative_router is not None:
            try:
                self.discriminative_router.fit(training_data, labels)
            except Exception as e:
                print(f"Error updating discriminative router: {e}")
        
        # Update generative router
        if self.generative_router is not None:
            try:
                self.generative_router.fit(training_data)
            except Exception as e:
                print(f"Error updating generative router: {e}")
    
    async def re_shard_paths(self, performance_threshold: float = 0.7) -> Dict[str, Any]:
        """
        Re-shard paths based on performance.
        
        Args:
            performance_threshold: Threshold for re-sharding
        
        Returns:
            Re-sharding results
        """
        print("Performing discriminative re-sharding...")
        
        # Identify underperforming paths
        underperforming_paths = [
            path_id for path_id, perf in self.path_performance.items()
            if perf.accuracy < performance_threshold
        ]
        
        if not underperforming_paths:
            return {"re_sharded": False, "reason": "No underperforming paths"}
        
        # Perform re-sharding
        re_sharding_result = await self.re_sharding_engine.re_shard(
            underperforming_paths, self.paths, self.path_performance
        )
        
        # Update statistics
        self.routing_statistics["re_sharding_events"] += 1
        
        return {
            "re_sharded": True,
            "underperforming_paths": underperforming_paths,
            "result": re_sharding_result
        }
    
    def get_path_statistics(self) -> Dict[str, Any]:
        """Get statistics about paths and routing."""
        return {
            "total_paths": len(self.paths),
            "routing_strategy": self.strategy.value,
            "routing_statistics": self.routing_statistics,
            "path_performance": {
                path_id: {
                    "accuracy": perf.accuracy,
                    "latency": perf.latency,
                    "throughput": perf.throughput,
                    "resource_usage": perf.resource_usage
                }
                for path_id, perf in self.path_performance.items()
            }
        }
    
    def add_path(self, path_config: PathConfig) -> None:
        """Add a new path to the router."""
        self.paths[path_config.path_id] = path_config
        self.path_performance[path_config.path_id] = PathPerformance(
            path_id=path_config.path_id,
            accuracy=0.5,
            latency=100.0,
            throughput=1.0,
            resource_usage=0.5,
            last_updated=time.time()
        )
    
    def remove_path(self, path_id: str) -> None:
        """Remove a path from the router."""
        if path_id in self.paths:
            del self.paths[path_id]
        if path_id in self.path_performance:
            del self.path_performance[path_id]


class ReShardingEngine:
    """Engine for discriminative re-sharding of paths."""
    
    def __init__(self):
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the re-sharding engine."""
        self.is_initialized = True
        print("  Re-sharding engine initialized")
    
    async def re_shard(self, underperforming_paths: List[str], 
                      paths: Dict[str, PathConfig],
                      path_performance: Dict[str, PathPerformance]) -> Dict[str, Any]:
        """
        Re-shard underperforming paths.
        
        Args:
            underperforming_paths: List of underperforming path IDs
            paths: All path configurations
            path_performance: Performance metrics
        
        Returns:
            Re-sharding results
        """
        if not self.is_initialized:
            raise ValueError("Re-sharding engine not initialized")
        
        re_sharding_results = {
            "re_sharded_paths": [],
            "new_assignments": {},
            "performance_improvements": {}
        }
        
        for path_id in underperforming_paths:
            if path_id not in paths:
                continue
            
            # Analyze path characteristics
            path_config = paths[path_id]
            current_perf = path_performance[path_id]
            
            # Generate new module assignments
            new_assignments = await self._generate_new_assignments(
                path_config, current_perf
            )
            
            # Update path
            paths[path_id] = PathConfig(
                path_id=path_config.path_id,
                modules=new_assignments.get("modules", path_config.modules),
                shared_modules=new_assignments.get("shared_modules", path_config.shared_modules),
                unshared_modules=new_assignments.get("unshared_modules", path_config.unshared_modules),
                routing_key=path_config.routing_key,
                priority=path_config.priority,
                metadata=path_config.metadata
            )
            
            re_sharding_results["re_sharded_paths"].append(path_id)
            re_sharding_results["new_assignments"][path_id] = new_assignments
            
            # Estimate performance improvement
            estimated_improvement = await self._estimate_performance_improvement(
                path_id, new_assignments
            )
            re_sharding_results["performance_improvements"][path_id] = estimated_improvement
        
        return re_sharding_results
    
    async def _generate_new_assignments(self, path_config: PathConfig, 
                                      current_perf: PathPerformance) -> Dict[str, Any]:
        """Generate new module assignments for a path."""
        # Simple re-sharding strategy: redistribute modules
        modules = path_config.modules.copy()
        shared_modules = path_config.shared_modules.copy()
        unshared_modules = path_config.unshared_modules.copy()
        
        # Move some modules between shared and unshared
        if len(shared_modules) > 1:
            # Move one module from shared to unshared
            moved_module = shared_modules.pop(0)
            unshared_modules.append(moved_module)
        
        return {
            "modules": modules,
            "shared_modules": shared_modules,
            "unshared_modules": unshared_modules
        }
    
    async def _estimate_performance_improvement(self, path_id: str, 
                                              new_assignments: Dict[str, Any]) -> Dict[str, float]:
        """Estimate performance improvement from re-sharding."""
        # Simple estimation based on module distribution
        shared_ratio = len(new_assignments["shared_modules"]) / len(new_assignments["modules"])
        
        return {
            "estimated_accuracy_improvement": shared_ratio * 0.1,
            "estimated_latency_reduction": shared_ratio * 0.05,
            "estimated_throughput_improvement": shared_ratio * 0.15
        }
