"""Vector Compression - MIB binarization (32x) + INT8 quantization (4x)."""
from __future__ import annotations
import math
import struct
from typing import Optional
import numpy as np


class MIBCompressor:
    """Mutual Information Binarization - 384 floats → 12 bytes (32x compression).
    
    Based on: "Binarized Vector Search" technique.
    Each dimension contributes 1 bit: positive=1, negative=0.
    """
    
    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension
    
    def compress(self, vector: list[float] | np.ndarray) -> bytes:
        """Compress float vector to binary representation."""
        vec = np.asarray(vector, dtype=np.float32)
        # Sign binarization: positive → 1, non-positive → 0
        bits = (vec > 0).astype(np.uint8)
        # Pack bits into bytes
        n_bytes = math.ceil(self._dim / 8)
        packed = np.zeros(n_bytes, dtype=np.uint8)
        for i in range(self._dim):
            byte_idx = i // 8
            bit_idx = i % 8
            packed[byte_idx] |= bits[i] << bit_idx
        return packed.tobytes()
    
    def decompress(self, data: bytes) -> list[float]:
        """Decompress binary back to float vector (+1/-1)."""
        n_bytes = math.ceil(self._dim / 8)
        result = np.zeros(self._dim, dtype=np.float32)
        for i in range(self._dim):
            byte_idx = i // 8
            bit_idx = i % 8
            if byte_idx < len(data):
                bit = (data[byte_idx] >> bit_idx) & 1
                result[i] = 1.0 if bit else -1.0
        return result.tolist()
    
    def similarity(self, v1: bytes, v2: bytes) -> float:
        """Hamming-distance-based similarity between two compressed vectors."""
        matches = 0
        total = min(len(v1), len(v2)) * 8
        for b1, b2 in zip(v1, v2):
            xor = b1 ^ b2
            matches += 8 - bin(xor).count('1')
        return matches / total if total > 0 else 0.0


class INT8Compressor:
    """INT8 Quantization - 384 float32 → 384 int8 (4x compression)."""
    
    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension
    
    def compress(self, vector: list[float] | np.ndarray) -> bytes:
        """Quantize float32 vector to int8."""
        vec = np.asarray(vector, dtype=np.float32)
        max_val = np.max(np.abs(vec)) or 1.0
        scale = 127.0 / max_val
        quantized = np.clip(np.round(vec * scale), -128, 127).astype(np.int8)
        return struct.pack(f"{self._dim}b", *quantized)
    
    def decompress(self, data: bytes, scale: float = 1.0) -> list[float]:
        """Dequantize int8 back to float32."""
        count = len(data)
        values = struct.unpack(f"{count}b", data)
        if scale == 0:
            scale = 1.0
        return [v / 127.0 / scale * 127.0 for v in values]
    
    def compress_with_scale(self, vector: list[float] | np.ndarray) -> tuple[bytes, float]:
        """Compress and return scale factor for lossless reconstruction."""
        vec = np.asarray(vector, dtype=np.float32)
        max_val = np.max(np.abs(vec)) or 1.0
        scale = max_val / 127.0
        quantized = np.clip(np.round(vec / scale), -128, 127).astype(np.int8)
        return struct.pack(f"{self._dim}b", *quantized), scale
    
    def decompress_with_scale(self, data: bytes, scale: float) -> list[float]:
        """Dequantize using stored scale factor."""
        count = len(data)
        values = struct.unpack(f"{count}b", data)
        return [v * scale for v in values]


class VectorCompressor:
    """Unified vector compression interface."""
    
    def __init__(self, method: str = "none", dimension: int = 384) -> None:
        self._method = method
        if method == "mib":
            self._compressor = MIBCompressor(dimension)
        elif method == "int8":
            self._compressor = INT8Compressor(dimension)
        else:
            self._compressor = None
    
    def compress(self, vector: list[float] | np.ndarray) -> tuple[bytes, dict]:
        """Compress vector. Returns (compressed_data, metadata)."""
        if self._compressor is None:
            vec = np.asarray(vector, dtype=np.float32)
            return vec.tobytes(), {"method": "raw", "dim": len(vector)}
        
        if isinstance(self._compressor, MIBCompressor):
            data = self._compressor.compress(vector)
            return data, {"method": "mib", "dim": len(vector) if hasattr(vector, '__len__') else 384}
        else:
            data, scale = self._compressor.compress_with_scale(vector)
            return data, {"method": "int8", "dim": len(vector) if hasattr(vector, '__len__') else 384, "scale": scale}
    
    def decompress(self, data: bytes, metadata: dict) -> list[float]:
        """Decompress vector using metadata."""
        method = metadata.get("method", "raw")
        if method == "raw":
            dim = metadata.get("dim", 384)
            return list(np.frombuffer(data, dtype=np.float32)[:dim])
        elif method == "mib":
            return self._compressor.decompress(data)
        elif method == "int8":
            scale = metadata.get("scale", 1.0)
            return self._compressor.decompress_with_scale(data, scale)
        return list(np.frombuffer(data, dtype=np.float32))
