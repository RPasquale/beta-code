"""
Evolutionary Database: MAP-elites selection and lineage tracking.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class ProgramRecord:
    """Record of a program in the evolutionary database."""
    program_id: str
    program: str
    score: float
    generation: int
    parent_id: str = None
    metadata: Dict[str, Any] = None


class EvolutionaryDB:
    """Evolutionary database for tracking programs and their lineage."""
    
    def __init__(self):
        self.programs = {}
        self.lineage = {}
        self.generation_count = 0
    
    def add_program(self, program: str, score: float, parent_id: str = None) -> str:
        """Add a program to the database."""
        program_id = f"prog_{len(self.programs)}"
        
        record = ProgramRecord(
            program_id=program_id,
            program=program,
            score=score,
            generation=self.generation_count,
            parent_id=parent_id
        )
        
        self.programs[program_id] = record
        
        if parent_id:
            if parent_id not in self.lineage:
                self.lineage[parent_id] = []
            self.lineage[parent_id].append(program_id)
        
        return program_id
    
    def get_program(self, program_id: str) -> ProgramRecord:
        """Get a program record by ID."""
        return self.programs.get(program_id)
    
    def get_lineage(self, program_id: str) -> List[str]:
        """Get the lineage (children) of a program."""
        return self.lineage.get(program_id, [])
    
    def get_best_programs(self, n: int = 10) -> List[ProgramRecord]:
        """Get the best N programs by score."""
        sorted_programs = sorted(
            self.programs.values(),
            key=lambda x: x.score,
            reverse=True
        )
        return sorted_programs[:n]
    
    def reset(self):
        """Reset the database."""
        self.programs.clear()
        self.lineage.clear()
        self.generation_count = 0


class MAP_Elites_Selector:
    """MAP-elites selector for maintaining diversity."""
    
    def __init__(self):
        self.elites = {}
        self.feature_dimensions = 2  # Simplified for demo
    
    def select_parents(self, population: List[Tuple[str, float]], n: int) -> List[Tuple[str, float]]:
        """Select parents using MAP-elites strategy."""
        # Sort by score and return top N
        sorted_pop = sorted(population, key=lambda x: x[1], reverse=True)
        return sorted_pop[:n]
    
    def get_elites(self) -> List[Tuple[str, float]]:
        """Get current elite programs."""
        return list(self.elites.items())
    
    def reset(self):
        """Reset the selector."""
        self.elites.clear()
