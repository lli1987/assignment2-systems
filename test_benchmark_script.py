"""
Quick test to verify the benchmark script structure is correct.
This is just a syntax/import check - actual benchmarking requires GPU.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")

    try:
        from cs336_systems.benchmark_flash_attention import (
            pytorch_attention,
            benchmark_forward,
            benchmark_backward,
            benchmark_end_to_end,
            run_benchmark,
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_pytorch_attention():
    """Test PyTorch attention function on CPU."""
    print("\nTesting PyTorch attention function...")

    try:
        import torch
        from cs336_systems.benchmark_flash_attention import pytorch_attention

        # Small test on CPU
        Q = torch.randn(1, 8, 16, requires_grad=True)
        K = torch.randn(1, 8, 16, requires_grad=True)
        V = torch.randn(1, 8, 16, requires_grad=True)

        # Test forward
        O = pytorch_attention(Q, K, V, is_causal=True)
        assert O.shape == Q.shape, f"Output shape mismatch: {O.shape} vs {Q.shape}"

        # Test backward
        dO = torch.randn_like(O)
        O.backward(dO)
        assert Q.grad is not None, "Q gradient is None"
        assert K.grad is not None, "K gradient is None"
        assert V.grad is not None, "V gradient is None"

        print("✓ PyTorch attention works correctly")
        return True

    except Exception as e:
        print(f"✗ PyTorch attention failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_script_structure():
    """Test that the script has correct structure."""
    print("\nTesting script structure...")

    try:
        import cs336_systems.benchmark_flash_attention as bench

        # Check main function exists
        assert hasattr(bench, 'main'), "main() function not found"

        # Check all benchmark functions exist
        required_functions = [
            'pytorch_attention',
            'benchmark_forward',
            'benchmark_backward',
            'benchmark_end_to_end',
            'run_benchmark',
        ]

        for fn_name in required_functions:
            assert hasattr(bench, fn_name), f"{fn_name} function not found"

        print("✓ Script structure is correct")
        return True

    except Exception as e:
        print(f"✗ Script structure check failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("FlashAttention Benchmark Script - Validation Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Script Structure", test_script_structure()))
    results.append(("PyTorch Attention", test_pytorch_attention()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All validation tests passed!")
        print("\nThe benchmark script is ready to run on a GPU machine.")
        print("Use: uv run python cs336_systems/benchmark_flash_attention.py")
    else:
        print("\n✗ Some validation tests failed. Please fix the issues above.")
        sys.exit(1)
