"""
Text2Loc Language Encoder with Ollama Integration

Replaces T5 model with Ollama API for sentence embeddings using qwen3-embedding:0.6b model.
Maintains the same transformer architecture for intra-sentence and inter-sentence processing.
"""

from typing import List, Optional, Dict, Any
import json
import time
import hashlib
import re
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests
from numpy.linalg import norm

from nltk import tokenize as text_tokenize

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# CARE: This has a trailing ReLU!!
def get_mlp(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct an MLP for use in other models.

    Args:
        channels (List[int]): List of number of channels in each layer.
        add_batchnorm (bool, optional): Whether to add BatchNorm after each layer. Defaults to True.

    Returns:
        nn.Sequential: Output MLP
    """
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU()
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                for i in range(1, len(channels))
            ]
        )


def get_mlp2(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct an MLP for use in other models without RELU in the final layer.

    Args:
        channels (List[int]): List of number of channels in each layer.
        add_batchnorm (bool, optional): Whether to add BatchNorm after each layer. Defaults to True.

    Returns:
        nn.Sequential: Output MLP
    """
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU()
                ) if i < len(channels) - 1
                else
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i])
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                if i < len(channels) - 1
                else nn.Sequential(nn.Linear(channels[i - 1], channels[i]))
                for i in range(1, len(channels))
            ]
        )


class OllamaEmbeddingClient:
    """Client for Ollama embedding API using qwen3-embedding:0.6b model."""

    def __init__(self,
                 model_name: str = "qwen3-embedding:0.6b",
                 ollama_url: str = "http://localhost:11434",
                 timeout: int = 30,
                 embedding_dim: int = 1024,
                 mock_mode: bool = False,
                 cache_enabled: bool = True,
                 max_cache_size: int = 1000):
        """
        Initialize Ollama embedding client.

        Args:
            model_name: Ollama model name
            ollama_url: Ollama server URL
            timeout: API timeout in seconds
            embedding_dim: Expected embedding dimension (used for mock mode)
            mock_mode: Use mock embeddings for testing
            cache_enabled: Enable embedding cache
            max_cache_size: Maximum cache size
        """
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.embedding_dim = embedding_dim
        self.mock_mode = mock_mode
        self.cache_enabled = cache_enabled
        self.max_cache_size = max_cache_size

        # Cache system
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self.cache_hits = 0
        self.cache_misses = 0

        # Performance stats
        self.total_calls = 0
        self.total_time = 0.0

        # HTTP session
        self.session = requests.Session() if not mock_mode else None

        if not mock_mode:
            self._test_connection()

        logger.info(f"OllamaEmbeddingClient initialized: model={model_name}, mock={mock_mode}")

    def _test_connection(self):
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"Connected to Ollama at {self.ollama_url}")
                # Check if model is available
                models = response.json().get("models", [])
                model_available = any(model["name"] == self.model_name for model in models)
                if not model_available:
                    logger.warning(f"Model {self.model_name} not found. Using first available model.")
                    if models:
                        self.model_name = models[0]["name"]
                        logger.info(f"Using model: {self.model_name}")
            else:
                logger.warning(f"Ollama connection issue: {response.status_code}")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            if not self.mock_mode:
                raise ConnectionError(f"Cannot connect to Ollama: {e}")

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model_name}:{text}".encode('utf-8')).hexdigest()

    def _get_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache."""
        if not self.cache_enabled:
            return None

        cache_key = self._get_cache_key(text)
        if cache_key in self.embedding_cache:
            self.cache_hits += 1
            return self.embedding_cache[cache_key]

        self.cache_misses += 1
        return None

    def _add_to_cache(self, text: str, embedding: np.ndarray):
        """Add embedding to cache."""
        if not self.cache_enabled:
            return

        # Clean cache if too large
        if len(self.embedding_cache) >= self.max_cache_size:
            keys_to_remove = list(self.embedding_cache.keys())[:self.max_cache_size // 2]
            for key in keys_to_remove:
                del self.embedding_cache[key]
            logger.info(f"Cleaned cache: removed {len(keys_to_remove)} entries")

        cache_key = self._get_cache_key(text)
        self.embedding_cache[cache_key] = embedding

    def _call_ollama_api(self, text: str) -> np.ndarray:
        """
        Call Ollama embedding API.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        if self.mock_mode:
            return self._mock_embedding(text)

        try:
            payload = {
                "model": self.model_name,
                "prompt": text
            }

            response = self.session.post(
                f"{self.ollama_url}/api/embeddings",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result.get("embedding", []), dtype=np.float32)

                if len(embedding) == 0:
                    raise ValueError("Empty embedding returned")

                # Normalize embedding
                embedding = embedding / norm(embedding)

                # Update embedding dimension based on actual response
                if len(embedding) != self.embedding_dim:
                    logger.info(f"Updating embedding dimension from {self.embedding_dim} to {len(embedding)}")
                    self.embedding_dim = len(embedding)

                return embedding
            else:
                logger.error(f"API error {response.status_code}: {response.text}")
                raise Exception(f"API error: {response.status_code}")

        except requests.exceptions.Timeout:
            logger.error(f"API timeout after {self.timeout}s")
            raise TimeoutError(f"API timeout after {self.timeout}s")
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise

    def _mock_embedding(self, text: str) -> np.ndarray:
        """Generate mock embedding for testing."""
        # Use text hash as seed for reproducibility
        seed = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
        np.random.seed(seed)

        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        embedding = embedding / norm(embedding)

        return embedding

    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Get embedding for a single text.

        Args:
            text: Text to embed
            use_cache: Use cache if available

        Returns:
            Embedding vector
        """
        start_time = time.time()

        try:
            # Check cache
            if use_cache:
                cached = self._get_from_cache(text)
                if cached is not None:
                    return cached

            # Get embedding
            embedding = self._call_ollama_api(text)

            # Add to cache
            if use_cache:
                self._add_to_cache(text, embedding)

            # Update stats
            self.total_calls += 1
            self.total_time += (time.time() - start_time)

            return embedding

        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            # Return zero vector on error
            return np.zeros(self.embedding_dim, dtype=np.float32)

    def embed_texts(self, texts: List[str], batch_delay: float = 0.05, use_cache: bool = True) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_delay: Delay between API calls (seconds)
            use_cache: Use cache if available

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for i, text in enumerate(texts):
            # Add delay to avoid overwhelming API
            if i > 0 and not self.mock_mode:
                time.sleep(batch_delay)

            embedding = self.embed_text(text, use_cache=use_cache)
            embeddings.append(embedding)

            # Log progress
            if (i + 1) % 10 == 0:
                logger.debug(f"Embedding progress: {i + 1}/{len(texts)}")

        return embeddings

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        avg_time = self.total_time / self.total_calls if self.total_calls > 0 else 0

        return {
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "average_time": avg_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.embedding_cache),
            "hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses)
                       if (self.cache_hits + self.cache_misses) > 0 else 0.0,
            "embedding_dim": self.embedding_dim,
            "model": self.model_name
        }

    def clear_cache(self):
        """Clear embedding cache."""
        self.embedding_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Embedding cache cleared")


class LanguageEncoder(torch.nn.Module):
    def __init__(self, embedding_dim,
                 hungging_model: str = "qwen3-embedding:0.6b",
                 fixed_embedding: bool = False,
                 intra_module_num_layers: int = 2,
                 intra_module_num_heads: int = 4,
                 is_fine: bool = False,
                 inter_module_num_layers: int = 2,
                 inter_module_num_heads: int = 4,
                 mock_mode: bool = False,
                 cache_enabled: bool = True,
                 ollama_url: str = "http://localhost:11434"):
        """
        Language encoder using Ollama API for sentence embeddings.
        Backward compatible with original Text2Loc API signature.

        Args:
            embedding_dim: Output embedding dimension
            hungging_model: Now interpreted as Ollama model name (e.g., "qwen3-embedding:0.6b")
                           Maps "t5-large", "t5-base", "t5-small" to "qwen3-embedding:0.6b"
            fixed_embedding: If True, embeddings are fixed (not trainable)
            intra_module_num_layers: Number of transformer layers for intra-sentence processing
            intra_module_num_heads: Number of attention heads for intra-sentence processing
            is_fine: Whether this is for fine localization
            inter_module_num_layers: Number of transformer layers for inter-sentence processing
            inter_module_num_heads: Number of attention heads for inter-sentence processing
            mock_mode: Use mock embeddings (for testing without Ollama)
            cache_enabled: Enable embedding cache
            ollama_url: Ollama server URL (new parameter for flexibility)
        """
        super(LanguageEncoder, self).__init__()

        self.is_fine = is_fine
        self.fixed_embedding = fixed_embedding
        self.embedding_dim = embedding_dim

        # Map HuggingFace model names to Ollama equivalents
        ollama_model_map = {
            "t5-large": "qwen3-embedding:0.6b",
            "t5-base": "qwen3-embedding:0.6b",
            "t5-small": "qwen3-embedding:0.6b",
        }

        ollama_model = ollama_model_map.get(hungging_model, hungging_model)

        # Initialize Ollama client
        self.embedding_client = OllamaEmbeddingClient(
            model_name=ollama_model,
            ollama_url=ollama_url,
            embedding_dim=1024,  # Default for qwen3-embedding:0.6b
            mock_mode=mock_mode,
            cache_enabled=cache_enabled
        )

        # We'll get actual embedding dimension after first call
        self.input_dim = 1024  # Will be updated

        # Intra-sentence transformer (processes each sentence embedding)
        # Note: With Ollama embeddings (sentence-level), we treat each sentence
        # as a sequence of length 1
        self.intra_module = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.input_dim,
                nhead=intra_module_num_heads,
                dim_feedforward=self.input_dim * 4,
                batch_first=False  # Expects (seq_len, batch, features)
            )
            for _ in range(intra_module_num_layers)
        ])

        # MLP to map from Ollama embedding dimension to desired embedding_dim
        self.inter_mlp = get_mlp2([self.input_dim, embedding_dim], add_batchnorm=True)

        # Inter-sentence transformer (for coarse localization only)
        if not is_fine:
            self.inter_module = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=inter_module_num_heads,
                    dim_feedforward=embedding_dim * 4,
                    batch_first=False
                )
                for _ in range(inter_module_num_layers)
            ])

    def _update_input_dim(self, actual_dim: int):
        """Update input dimension based on actual Ollama embedding dimension."""
        if self.input_dim != actual_dim:
            logger.info(f"Updating LanguageEncoder input_dim from {self.input_dim} to {actual_dim}")
            self.input_dim = actual_dim

            # Recreate intra_module with correct dimension
            intra_num_layers = len(self.intra_module)
            intra_num_heads = self.intra_module[0].nhead if intra_num_layers > 0 else 4

            self.intra_module = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.input_dim,
                    nhead=intra_num_heads,
                    dim_feedforward=self.input_dim * 4,
                    batch_first=False
                )
                for _ in range(intra_num_layers)
            ])

            # Recreate inter_mlp with correct input dimension
            self.inter_mlp = get_mlp2([self.input_dim, self.embedding_dim], add_batchnorm=True)

            # Move new modules to correct device
            if hasattr(self, '_device'):
                self.intra_module.to(self._device)
                self.inter_mlp.to(self._device)

    def forward(self, descriptions: List[str]) -> torch.Tensor:
        """
        Encode language descriptions.

        Args:
            descriptions: List of text descriptions

        Returns:
            Tensor of shape [batch_size, num_sentences, embedding_dim] for fine localization,
            or [batch_size, embedding_dim] for coarse localization
        """
        # Split each description into sentences
        split_union_sentences = []
        for description in descriptions:
            try:
                split_union_sentences.extend(text_tokenize.sent_tokenize(description))
            except Exception as e:
                # If tokenizer fails, use simple split by period as last resort
                logger.warning(f"Tokenizer failed: {e}, using simple period split")
                # Simple split by period, question mark, exclamation
                simple_sentences = re.split(r'[.!?]+', description)
                for sent in simple_sentences:
                    sent = sent.strip()
                    if sent:
                        # Add period back if it was split by period
                        if sent and not sent.endswith(('.', '!', '?')):
                            split_union_sentences.append(sent + '.')
                        else:
                            split_union_sentences.append(sent)

        batch_size = len(descriptions)
        num_sentences = len(split_union_sentences) // batch_size

        # Get embeddings from Ollama
        embeddings_list = self.embedding_client.embed_texts(split_union_sentences)

        # Convert to tensor and move to device
        embeddings_np = np.stack(embeddings_list, axis=0)  # [num_total_sentences, embedding_dim]

        # Update input dimension if needed
        actual_dim = embeddings_np.shape[1]
        if actual_dim != self.input_dim:
            self._update_input_dim(actual_dim)

        # Convert to tensor
        description_encodings = torch.from_numpy(embeddings_np).float().to(self.device)

        # If fixed_embedding, detach from computation graph
        if self.fixed_embedding:
            description_encodings = description_encodings.detach()

        # Reshape for transformer: [seq_len=1, batch=num_total_sentences, features]
        # Since each sentence is a single embedding, we treat it as sequence of length 1
        description_encodings = description_encodings.unsqueeze(0)  # [1, num_total_sentences, input_dim]

        # Intra-sentence transformer processing
        # (With seq_len=1, this is essentially applying self-attention to a single token,
        # which still allows feature transformation through feed-forward layers)
        for layer in self.intra_module:
            description_encodings = layer(description_encodings)

        # Remove sequence dimension and get sentence embeddings
        # Since seq_len=1, we just take the first (and only) element
        description_encodings = description_encodings.squeeze(0)  # [num_total_sentences, input_dim]

        # Apply MLP to map to desired embedding dimension
        description_encodings = self.inter_mlp(description_encodings)  # [num_total_sentences, embedding_dim]

        # Reshape to [batch_size, num_sentences, embedding_dim]
        description_encodings = description_encodings.view(batch_size, num_sentences, -1)

        # For fine localization, return all sentence embeddings
        if self.is_fine:
            return description_encodings

        # For coarse localization, apply inter-sentence transformer
        # Reshape to [seq_len=num_sentences, batch=batch_size, embedding_dim]
        description_encodings = description_encodings.permute(1, 0, 2)

        # Inter-sentence transformer processing
        for layer in self.inter_module:
            # Residual connection
            description_encodings = description_encodings + layer(description_encodings)

        # Max pooling over sentences to get single embedding per batch
        description_encodings = description_encodings.max(dim=0)[0]  # [batch_size, embedding_dim]

        return description_encodings

    @property
    def device(self) -> torch.device:
        """Get the device of the model parameters."""
        if not hasattr(self, '_device'):
            self._device = next(self.inter_mlp.parameters()).device
        return self._device

    def get_stats(self) -> Dict[str, Any]:
        """Get Ollama client statistics."""
        return self.embedding_client.get_stats()

    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_client.clear_cache()


# Backward compatibility wrapper for existing code
class LanguageEncoderLegacy(LanguageEncoder):
    """Legacy wrapper for compatibility with existing code that uses 'hungging_model' parameter."""

    def __init__(self, embedding_dim,
                 hungging_model: str = "qwen3-embedding:0.6b",  # Now used as Ollama model name
                 fixed_embedding: bool = False,
                 intra_module_num_layers: int = 2,
                 intra_module_num_heads: int = 4,
                 is_fine: bool = False,
                 inter_module_num_layers: int = 2,
                 inter_module_num_heads: int = 4,
                 mock_mode: bool = False,
                 cache_enabled: bool = True):
        """
        Legacy interface for backward compatibility.

        Args:
            hungging_model: Now interpreted as Ollama model name
            Other parameters same as LanguageEncoder
        """
        # Map "t5-large" and other T5 models to appropriate Ollama models
        ollama_model_map = {
            "t5-large": "qwen3-embedding:0.6b",  # Map T5-large to qwen3-embedding
            "t5-base": "qwen3-embedding:0.6b",
            "t5-small": "qwen3-embedding:0.6b",
        }

        ollama_model = ollama_model_map.get(hungging_model, hungging_model)

        super().__init__(
            embedding_dim=embedding_dim,
            hungging_model=ollama_model,
            ollama_url="http://localhost:11434",
            fixed_embedding=fixed_embedding,
            intra_module_num_layers=intra_module_num_layers,
            intra_module_num_heads=intra_module_num_heads,
            is_fine=is_fine,
            inter_module_num_layers=inter_module_num_layers,
            inter_module_num_heads=inter_module_num_heads,
            mock_mode=mock_mode,
            cache_enabled=cache_enabled
        )

        # Store original parameter for reference
        self.hungging_model = hungging_model
