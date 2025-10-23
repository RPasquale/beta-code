"""
Distributed Training: Streaming DiLoCo and DiPaCo implementation.

Implements distributed training with fragmented synchronization,
overlap, and quantized outer gradients.
"""

from .streaming_diloco import StreamingDiLoCo, FragmentConfig, WorkerConfig
from .dipaco import DiPaCo_Router, PathConfig, RoutingStrategy
from .fragments import FragmentManager, FragmentScheduler
from .quantization import QuantizationManager, FP4Quantizer
from .communication import CommunicationManager, OverlapManager

__all__ = [
    "StreamingDiLoCo",
    "FragmentConfig",
    "WorkerConfig",
    "DiPaCo_Router",
    "PathConfig",
    "RoutingStrategy",
    "FragmentManager",
    "FragmentScheduler",
    "QuantizationManager",
    "FP4Quantizer",
    "CommunicationManager",
    "OverlapManager",
]
