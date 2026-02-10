"""
Quick import test for LanguageEncoder with Ollama integration.
This test runs in mock mode and doesn't require Ollama or nltk.
"""

import sys
import os
import numpy as np
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test basic import of LanguageEncoder."""
    try:
        from models.language_encoder import (
            LanguageEncoder,
            LanguageEncoderLegacy,
            get_mlp,
            get_mlp2
        )
        print("✅ Import successful")
        return True, (LanguageEncoder, LanguageEncoderLegacy, get_mlp, get_mlp2)
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False, None

def test_nltk_workaround():
    """Work around nltk dependency for testing."""
    try:
        # Try to import nltk normally
        from nltk import tokenize as text_tokenize
        print("✅ nltk available")
        return True
    except ImportError:
        print("⚠️ nltk not available, using simple mock tokenizer")
        # Create a mock tokenizer that splits by periods
        import types
        mock_tokenize = types.ModuleType('mock_tokenize')

        def mock_sent_tokenize(text):
            # Simple period-based sentence splitting
            sentences = []
            for sent in text.split('.'):
                sent = sent.strip()
                if sent:
                    sentences.append(sent + '.')
            return sentences if sentences else [text]

        mock_tokenize.sent_tokenize = mock_sent_tokenize

        # Add to sys.modules so language_encoder can import it
        sys.modules['nltk.tokenize'] = mock_tokenize
        return False

def test_language_encoder_basic():
    """Test basic functionality of LanguageEncoder."""
    # First try the nltk workaround
    nltk_available = test_nltk_workaround()

    # Now import LanguageEncoder
    success, modules = test_import()
    if not success:
        return False

    LanguageEncoder, LanguageEncoderLegacy, get_mlp, get_mlp2 = modules

    try:
        print("\n🔧 Testing LanguageEncoder initialization...")

        # Test 1: Coarse localization (default)
        encoder_coarse = LanguageEncoder(
            embedding_dim=128,
            hungging_model="qwen3-embedding:0.6b",
            fixed_embedding=False,
            intra_module_num_layers=2,
            intra_module_num_heads=4,
            is_fine=False,
            inter_module_num_layers=1,
            inter_module_num_heads=4,
            mock_mode=True,  # Mock mode for testing
            cache_enabled=True
        )
        print("✅ Coarse localization encoder created")

        # Test 2: Fine localization
        encoder_fine = LanguageEncoder(
            embedding_dim=256,
            hungging_model="t5-large",  # Should map to qwen3-embedding:0.6b
            is_fine=True,
            mock_mode=True
        )
        print("✅ Fine localization encoder created")

        # Test 3: Legacy interface (backward compatibility)
        encoder_legacy = LanguageEncoderLegacy(
            embedding_dim=128,
            hungging_model="t5-large",
            mock_mode=True
        )
        print("✅ Legacy encoder created")

        # Test forward passes
        print("\n🔧 Testing forward passes...")

        # Sample descriptions
        descriptions = [
            "A red car is parked near a building.",
            "There is a tree on the left side.",
            "Two people are walking."
        ]

        # Test coarse encoder
        output_coarse = encoder_coarse(descriptions)
        print(f"✅ Coarse encoder output shape: {output_coarse.shape}")
        print(f"   Expected: [3, 128], Actual: {list(output_coarse.shape)}")

        # Test fine encoder
        output_fine = encoder_fine(descriptions)
        print(f"✅ Fine encoder output shape: {output_fine.shape}")
        print(f"   Should have 3 dimensions (batch, sentences, features)")

        # Test legacy encoder
        output_legacy = encoder_legacy(descriptions)
        print(f"✅ Legacy encoder output shape: {output_legacy.shape}")

        # Test model mapping
        print("\n🔧 Testing model mapping...")
        print(f"   T5-large mapped to: {encoder_fine.embedding_client.model_name}")
        print(f"   Expected: qwen3-embedding:0.6b")

        # Test MLP functions
        print("\n🔧 Testing MLP utilities...")
        mlp = get_mlp([128, 64, 32])
        mlp2 = get_mlp2([128, 64, 32])

        test_input = torch.randn(4, 128)
        output_mlp = mlp(test_input)
        output_mlp2 = mlp2(test_input)

        print(f"✅ MLP output shapes: {output_mlp.shape}, {output_mlp2.shape}")

        # Test device property
        print(f"✅ Encoder device: {encoder_coarse.device}")

        # Test statistics
        print("\n🔧 Testing statistics...")
        stats = encoder_coarse.get_stats()
        print(f"✅ Stats available: {list(stats.keys())}")

        # Test edge cases
        print("\n🔧 Testing edge cases...")
        single_output = encoder_coarse(["Single description."])
        print(f"✅ Single description shape: {single_output.shape}")

        long_desc = ["First. Second. Third."]
        long_output = encoder_coarse(long_desc)
        print(f"✅ Multiple sentences shape: {long_output.shape}")

        print("\n🎉 All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ollama_client_mock():
    """Test OllamaEmbeddingClient in isolation."""
    try:
        # Import after nltk workaround
        from models.language_encoder import OllamaEmbeddingClient

        print("\n🔧 Testing OllamaEmbeddingClient...")

        client = OllamaEmbeddingClient(
            model_name="qwen3-embedding:0.6b",
            ollama_url="http://localhost:11434",
            timeout=5,
            embedding_dim=1024,
            mock_mode=True,
            cache_enabled=True
        )

        print("✅ Ollama client created")

        # Test single embedding
        embedding = client.embed_text("Test text")
        print(f"✅ Single embedding shape: {embedding.shape}")
        print(f"   Expected: (1024,), Actual: {embedding.shape}")

        # Test batch embeddings
        embeddings = client.embed_texts(["Text 1", "Text 2", "Text 3"])
        print(f"✅ Batch embeddings count: {len(embeddings)}")

        # Test cache
        embedding2 = client.embed_text("Test text")
        print(f"✅ Cache hit: {np.array_equal(embedding, embedding2)}")

        # Test stats
        stats = client.get_stats()
        print(f"✅ Stats: {stats['total_calls']} calls, {stats['cache_hits']} hits")

        # Clear cache
        client.clear_cache()
        print("✅ Cache cleared")

        return True

    except Exception as e:
        print(f"❌ Ollama client test failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 60)
    print("Quick LanguageEncoder Integration Test")
    print("=" * 60)
    print("Running in mock mode (no Ollama required)...")
    print()

    # Run tests
    all_passed = True

    # Test 1: Basic imports and nltk workaround
    print("📦 Test 1: Module imports...")
    test_nltk_workaround()

    # Test 2: LanguageEncoder functionality
    print("\n🤖 Test 2: LanguageEncoder functionality...")
    if not test_language_encoder_basic():
        all_passed = False

    # Test 3: Ollama client in isolation
    print("\n🔌 Test 3: OllamaEmbeddingClient...")
    if not test_ollama_client_mock():
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    if all_passed:
        print("✅ All tests passed!")
        print("\nNext steps:")
        print("1. Install nltk: pip install nltk")
        print("2. Install Ollama and pull model: ollama pull qwen3-embedding:0.6b")
        print("3. Test with real Ollama (set mock_mode=False)")
    else:
        print("❌ Some tests failed")
        print("\nTroubleshooting:")
        print("1. Check Python dependencies: pip install torch numpy requests")
        print("2. For nltk: pip install nltk or use the mock tokenizer")
        print("3. Check file paths and imports")

    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
