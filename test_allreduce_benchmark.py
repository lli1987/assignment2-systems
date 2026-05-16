"""
Quick test to verify the all-reduce benchmark script structure.
Tests CPU version (Gloo) only since GPU may not be available.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all imports work."""
    print("Testing imports...")

    try:
        from cs336_systems.benchmark_allreduce import (
            setup_process_group,
            cleanup,
            benchmark_allreduce,
            run_single_config,
            plot_results,
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_script_structure():
    """Test that the script has correct structure."""
    print("\nTesting script structure...")

    try:
        import cs336_systems.benchmark_allreduce as bench

        # Check main function exists
        assert hasattr(bench, 'main'), "main() function not found"

        # Check all required functions exist
        required_functions = [
            'setup_process_group',
            'cleanup',
            'benchmark_allreduce',
            'run_benchmark_wrapper',
            'run_single_config',
            'plot_results',
        ]

        for fn_name in required_functions:
            assert hasattr(bench, fn_name), f"{fn_name} function not found"

        print("✓ Script structure is correct")
        return True

    except Exception as e:
        print(f"✗ Script structure check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_allreduce():
    """Test simple all-reduce with 2 processes on CPU."""
    print("\nTesting simple all-reduce (Gloo/CPU, 2 processes, 1MB)...")

    try:
        from cs336_systems.benchmark_allreduce import run_single_config

        # Run a simple configuration
        result = run_single_config(
            world_size=2,
            backend='gloo',
            data_size_mb=1,
            warmup_iters=2,
            benchmark_iters=5,
            port=29600,  # Use different port
        )

        # Check result structure
        assert 'world_size' in result, "Missing 'world_size' in result"
        assert 'backend' in result, "Missing 'backend' in result"
        assert 'mean_time_ms' in result, "Missing 'mean_time_ms' in result"
        assert 'bandwidth_mbps' in result, "Missing 'bandwidth_mbps' in result"

        # Check values are reasonable
        assert result['world_size'] == 2, "Incorrect world_size"
        assert result['backend'] == 'gloo', "Incorrect backend"
        assert result['mean_time_ms'] > 0, "Mean time should be positive"
        assert result['bandwidth_mbps'] > 0, "Bandwidth should be positive"

        print(f"✓ All-reduce test passed")
        print(f"  Mean time: {result['mean_time_ms']:.3f} ms")
        print(f"  Bandwidth: {result['bandwidth_mbps']:.2f} MB/s")
        return True

    except Exception as e:
        print(f"✗ All-reduce test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_plot_creation():
    """Test that plotting functions work with dummy data."""
    print("\nTesting plot creation...")

    try:
        import pandas as pd
        import tempfile
        from pathlib import Path
        from cs336_systems.benchmark_allreduce import plot_results

        # Create dummy data
        dummy_data = {
            'world_size': [2, 2, 4, 4],
            'backend': ['gloo', 'nccl', 'gloo', 'nccl'],
            'data_size_mb': [1, 1, 10, 10],
            'mean_time_ms': [0.5, 0.1, 5.0, 1.0],
            'std_time_ms': [0.05, 0.01, 0.5, 0.1],
            'bandwidth_mbps': [2.0, 10.0, 2.0, 10.0],
        }
        df = pd.DataFrame(dummy_data)

        # Create temporary directory for plots
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            plot_results(df, tmp_path)

            # Check that plots were created
            expected_plots = [
                'allreduce_latency_by_backend.png',
                'allreduce_bandwidth_by_processes.png',
                'allreduce_gloo_vs_nccl.png',
            ]

            for plot_name in expected_plots:
                plot_file = tmp_path / plot_name
                assert plot_file.exists(), f"Plot {plot_name} was not created"

        print("✓ Plot creation test passed")
        return True

    except Exception as e:
        print(f"✗ Plot creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("All-Reduce Benchmark Script - Validation Tests")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Script Structure", test_script_structure()))
    results.append(("Plot Creation", test_plot_creation()))
    results.append(("Simple All-Reduce", test_simple_allreduce()))

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
        print("\nThe benchmark script is ready to run.")
        print("Usage: uv run python cs336_systems/benchmark_allreduce.py")
    else:
        print("\n✗ Some validation tests failed. Please fix the issues above.")
        sys.exit(1)
