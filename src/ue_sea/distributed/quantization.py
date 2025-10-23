"""
Quantization: FP4 quantization for outer gradients.

Implements E3M0 quantization scheme for bandwidth efficiency.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum


class QuantizationScheme(Enum):
    """Quantization schemes."""
    E3M0 = "E3M0"  # 3 exponent bits, 0 mantissa bits
    FP4 = "FP4"    # 4-bit floating point
    INT4 = "INT4"  # 4-bit integer


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    scheme: QuantizationScheme = QuantizationScheme.E3M0
    scale_factor: float = 1.0
    zero_point: int = 0
    min_val: float = -8.0
    max_val: float = 7.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class FP4Quantizer:
    """FP4 quantizer for gradients."""
    
    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize the quantizer."""
        self.is_initialized = True
        print(f"FP4 Quantizer initialized with scheme: {self.config.scheme.value}")
    
    async def quantize(self, tensor: torch.Tensor, scheme: str = None) -> torch.Tensor:
        """
        Quantize tensor to FP4.
        
        Args:
            tensor: Input tensor
            scheme: Quantization scheme (overrides config)
        
        Returns:
            Quantized tensor
        """
        if not self.is_initialized:
            raise ValueError("Quantizer not initialized")
        
        scheme = scheme or self.config.scheme.value
        
        if scheme == "E3M0":
            return self._quantize_e3m0(tensor)
        elif scheme == "FP4":
            return self._quantize_fp4(tensor)
        elif scheme == "INT4":
            return self._quantize_int4(tensor)
        else:
            raise ValueError(f"Unknown quantization scheme: {scheme}")
    
    def _quantize_e3m0(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize using E3M0 scheme (3 exponent, 0 mantissa)."""
        # E3M0: 1 sign bit + 3 exponent bits + 0 mantissa bits
        # Range: -8 to 7 (approximately)
        
        # Clamp to range
        tensor_clamped = torch.clamp(tensor, self.config.min_val, self.config.max_val)
        
        # Scale to 4-bit range
        scale = 15.0 / (self.config.max_val - self.config.min_val)
        tensor_scaled = tensor_clamped * scale
        
        # Round to nearest integer
        tensor_rounded = torch.round(tensor_scaled)
        
        # Convert to 4-bit integer
        tensor_4bit = tensor_rounded.to(torch.int8)
        
        return tensor_4bit
    
    def _quantize_fp4(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize using FP4 scheme."""
        # FP4: 1 sign bit + 3 exponent bits + 0 mantissa bits
        # Similar to E3M0 but with different scaling
        
        # Clamp to range
        tensor_clamped = torch.clamp(tensor, -8.0, 7.0)
        
        # Scale to 4-bit range
        tensor_scaled = tensor_clamped * 2.0
        
        # Round to nearest integer
        tensor_rounded = torch.round(tensor_scaled)
        
        # Convert to 4-bit integer
        tensor_4bit = tensor_rounded.to(torch.int8)
        
        return tensor_4bit
    
    def _quantize_int4(self, tensor: torch.Tensor) -> torch.Tensor:
        """Quantize using INT4 scheme."""
        # INT4: 4-bit signed integer
        # Range: -8 to 7
        
        # Clamp to range
        tensor_clamped = torch.clamp(tensor, -8, 7)
        
        # Round to nearest integer
        tensor_rounded = torch.round(tensor_clamped)
        
        # Convert to 4-bit integer
        tensor_4bit = tensor_rounded.to(torch.int8)
        
        return tensor_4bit
    
    async def dequantize(self, quantized_tensor: torch.Tensor, scheme: str = None) -> torch.Tensor:
        """
        Dequantize tensor from FP4.
        
        Args:
            quantized_tensor: Quantized tensor
            scheme: Quantization scheme
        
        Returns:
            Dequantized tensor
        """
        if not self.is_initialized:
            raise ValueError("Quantizer not initialized")
        
        scheme = scheme or self.config.scheme.value
        
        if scheme == "E3M0":
            return self._dequantize_e3m0(quantized_tensor)
        elif scheme == "FP4":
            return self._dequantize_fp4(quantized_tensor)
        elif scheme == "INT4":
            return self._dequantize_int4(quantized_tensor)
        else:
            raise ValueError(f"Unknown quantization scheme: {scheme}")
    
    def _dequantize_e3m0(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize using E3M0 scheme."""
        # Convert back to float
        tensor_float = quantized_tensor.float()
        
        # Scale back
        scale = (self.config.max_val - self.config.min_val) / 15.0
        tensor_scaled = tensor_float / scale
        
        return tensor_scaled
    
    def _dequantize_fp4(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize using FP4 scheme."""
        # Convert back to float
        tensor_float = quantized_tensor.float()
        
        # Scale back
        tensor_scaled = tensor_float / 2.0
        
        return tensor_scaled
    
    def _dequantize_int4(self, quantized_tensor: torch.Tensor) -> torch.Tensor:
        """Dequantize using INT4 scheme."""
        # Convert back to float
        tensor_float = quantized_tensor.float()
        
        return tensor_float
    
    def get_compression_ratio(self) -> float:
        """Get compression ratio achieved by quantization."""
        # FP4 is 4 bits vs 32 bits for float32
        return 32.0 / 4.0  # 8x compression
    
    def get_bandwidth_savings(self) -> float:
        """Get bandwidth savings percentage."""
        return (1.0 - 1.0 / self.get_compression_ratio()) * 100.0  # 87.5% savings


class QuantizationManager:
    """Manager for quantization operations."""
    
    def __init__(self):
        self.quantizers = {}
        self.is_initialized = False
    
    async def initialize(self) -> None:
        """Initialize quantization manager."""
        # Initialize different quantization schemes
        self.quantizers["E3M0"] = FP4Quantizer(QuantizationConfig(scheme=QuantizationScheme.E3M0))
        self.quantizers["FP4"] = FP4Quantizer(QuantizationConfig(scheme=QuantizationScheme.FP4))
        self.quantizers["INT4"] = FP4Quantizer(QuantizationConfig(scheme=QuantizationScheme.INT4))
        
        # Initialize all quantizers
        for quantizer in self.quantizers.values():
            await quantizer.initialize()
        
        self.is_initialized = True
        print("Quantization Manager initialized")
    
    async def quantize(self, tensor: torch.Tensor, scheme: str = "E3M0") -> torch.Tensor:
        """Quantize tensor using specified scheme."""
        if not self.is_initialized:
            raise ValueError("Quantization manager not initialized")
        
        if scheme not in self.quantizers:
            raise ValueError(f"Unknown quantization scheme: {scheme}")
        
        return await self.quantizers[scheme].quantize(tensor, scheme)
    
    async def dequantize(self, tensor: torch.Tensor, scheme: str = "E3M0") -> torch.Tensor:
        """Dequantize tensor using specified scheme."""
        if not self.is_initialized:
            raise ValueError("Quantization manager not initialized")
        
        if scheme not in self.quantizers:
            raise ValueError(f"Unknown quantization scheme: {scheme}")
        
        return await self.quantizers[scheme].dequantize(tensor, scheme)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get quantization statistics."""
        return {
            "is_initialized": self.is_initialized,
            "available_schemes": list(self.quantizers.keys()),
            "compression_ratio": self.quantizers["E3M0"].get_compression_ratio(),
            "bandwidth_savings": self.quantizers["E3M0"].get_bandwidth_savings()
        }
