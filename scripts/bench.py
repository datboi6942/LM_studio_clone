"""Benchmark script for performance testing."""

import time
from typing import Any, Callable, List

import structlog

logger = structlog.get_logger(__name__)


def benchmark_function(func: Callable[..., Any], *args: Any, **kwargs: Any) -> float:
    """Benchmark a function execution time.
    
    Args:
        func: Function to benchmark.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.
        
    Returns:
        Execution time in seconds.
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    return end_time - start_time


def run_benchmarks() -> None:
    """Run all benchmarks."""
    logger.info("Running performance benchmarks...")
    
    # Example benchmark for string operations
    def test_string_concat(n: int) -> str:
        """Test string concatenation performance."""
        result = ""
        for i in range(n):
            result += str(i)
        return result
    
    def test_string_join(n: int) -> str:
        """Test string join performance."""
        return "".join(str(i) for i in range(n))
    
    # Run benchmarks
    n = 10000
    
    concat_time = benchmark_function(test_string_concat, n)
    join_time = benchmark_function(test_string_join, n)
    
    logger.info(
        "String operations benchmark",
        concat_time=f"{concat_time:.4f}s",
        join_time=f"{join_time:.4f}s",
        speedup=f"{concat_time/join_time:.2f}x"
    )
    
    # Performance assertions
    assert join_time < concat_time, "Join should be faster than concatenation"
    assert join_time < 0.1, f"Join operation too slow: {join_time}s"
    
    logger.info("All benchmarks passed!")


if __name__ == "__main__":
    run_benchmarks() 