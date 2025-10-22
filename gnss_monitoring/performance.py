"""
Performance Optimization Module
================================

Tools for performance profiling, optimization, and batch processing.
"""

import time
import functools
import cProfile
import pstats
import io
from typing import Callable, Any, Dict, List, Optional
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

from .logger import get_logger


logger = get_logger(__name__)


class PerformanceProfiler:
    """
    Performance profiling and analysis tool.
    """

    def __init__(self):
        """Initialize profiler."""
        self.profiler = cProfile.Profile()
        self.stats = None
        self.results = {}

    def profile(self, func: Callable) -> Callable:
        """
        Decorator to profile a function.

        Usage:
            @profiler.profile
            def my_function():
                pass
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.profiler.enable()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.profiler.disable()
        return wrapper

    def get_stats(self, sort_by: str = 'cumulative', limit: int = 20) -> str:
        """
        Get profiling statistics.

        Args:
            sort_by: Sort criterion ('cumulative', 'time', 'calls').
            limit: Number of functions to display.

        Returns:
            Formatted statistics string.
        """
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats(sort_by)
        ps.print_stats(limit)
        return s.getvalue()

    def save_stats(self, output_path: str):
        """Save profiling stats to file."""
        self.profiler.dump_stats(output_path)
        logger.info(f"Profiling stats saved to {output_path}")


class PerformanceTimer:
    """
    Context manager for timing code blocks.

    Usage:
        with PerformanceTimer("My operation"):
            # code to time
            pass
    """

    def __init__(self, name: str = "Operation", verbose: bool = True):
        """
        Initialize timer.

        Args:
            name: Name of the operation.
            verbose: Print timing information.
        """
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        if self.verbose:
            logger.info(f"⏱️  Starting: {self.name}")
        return self

    def __exit__(self, *args):
        """Stop timer and print result."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        if self.verbose:
            logger.info(f"✅ Completed: {self.name} in {self.elapsed:.3f}s")

    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed if self.elapsed else 0.0


def measure_performance(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Usage:
        @measure_performance
        def my_function():
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"Function '{func.__name__}' took {elapsed:.3f}s")
        return result
    return wrapper


class BatchProcessor:
    """
    Batch processing with parallel execution support.
    """

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False
    ):
        """
        Initialize batch processor.

        Args:
            max_workers: Maximum number of workers. If None, uses CPU count.
            use_processes: If True, use ProcessPoolExecutor, else ThreadPoolExecutor.
        """
        if max_workers is None:
            max_workers = mp.cpu_count()

        self.max_workers = max_workers
        self.use_processes = use_processes
        self.executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        logger.info(
            f"BatchProcessor initialized: {max_workers} workers, "
            f"mode={'processes' if use_processes else 'threads'}"
        )

    def process_batch(
        self,
        func: Callable,
        items: List[Any],
        *args,
        **kwargs
    ) -> List[Any]:
        """
        Process items in parallel.

        Args:
            func: Function to apply to each item.
            items: List of items to process.
            *args, **kwargs: Additional arguments for func.

        Returns:
            List of results.
        """
        logger.info(f"Processing batch of {len(items)} items with {self.max_workers} workers")

        with self.executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(func, item, *args, **kwargs) for item in items]

            # Collect results
            results = []
            for i, future in enumerate(futures, 1):
                try:
                    result = future.result()
                    results.append(result)
                    if i % 10 == 0:
                        logger.debug(f"Processed {i}/{len(items)} items")
                except Exception as e:
                    logger.error(f"Error processing item {i}: {e}")
                    results.append(None)

        logger.info(f"Batch processing complete: {len(results)} results")
        return results

    def process_files(
        self,
        func: Callable,
        file_paths: List[str],
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process multiple files in parallel.

        Args:
            func: Function to apply to each file.
            file_paths: List of file paths.
            *args, **kwargs: Additional arguments for func.

        Returns:
            Dictionary mapping file paths to results.
        """
        logger.info(f"Processing {len(file_paths)} files")

        results = self.process_batch(func, file_paths, *args, **kwargs)

        # Create result dictionary
        result_dict = {
            path: result
            for path, result in zip(file_paths, results)
        }

        return result_dict


class MemoryOptimizer:
    """
    Tools for memory optimization.
    """

    @staticmethod
    def chunk_dataframe(df, chunk_size: int = 10000):
        """
        Generator to process DataFrame in chunks.

        Args:
            df: DataFrame to chunk.
            chunk_size: Size of each chunk.

        Yields:
            DataFrame chunks.
        """
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i + chunk_size]

    @staticmethod
    def reduce_memory_usage(df):
        """
        Reduce DataFrame memory usage by optimizing dtypes.

        Args:
            df: DataFrame to optimize.

        Returns:
            Optimized DataFrame.
        """
        start_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage: {start_mem:.2f} MB")

        for col in df.columns:
            col_type = df[col].dtype

            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()

                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)

        end_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Memory usage after optimization: {end_mem:.2f} MB")
        logger.info(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

        return df


class CacheManager:
    """
    Simple caching mechanism for expensive operations.
    """

    def __init__(self, cache_dir: str = './cache'):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"CacheManager initialized: {cache_dir}")

    def cache_result(self, key: str, ttl: int = 3600):
        """
        Decorator to cache function results.

        Args:
            key: Cache key.
            ttl: Time to live in seconds.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                cache_file = self.cache_dir / f"{key}.pkl"

                # Check if cache exists and is fresh
                if cache_file.exists():
                    age = time.time() - cache_file.stat().st_mtime
                    if age < ttl:
                        logger.debug(f"Cache hit for '{key}'")
                        import pickle
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)

                # Compute result
                logger.debug(f"Cache miss for '{key}', computing...")
                result = func(*args, **kwargs)

                # Save to cache
                import pickle
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)

                return result
            return wrapper
        return decorator

    def clear_cache(self):
        """Clear all cached files."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir()
        logger.info("Cache cleared")


# Import numpy for memory optimizer
import numpy as np


if __name__ == '__main__':
    # Test performance tools
    print("Testing Performance Optimization Module...")
    print("-" * 70)

    # Test timer
    print("\n1. Testing PerformanceTimer...")
    with PerformanceTimer("Sleep test"):
        time.sleep(0.1)

    # Test decorator
    print("\n2. Testing @measure_performance decorator...")
    @measure_performance
    def slow_function():
        time.sleep(0.05)
        return "Done"

    result = slow_function()

    # Test batch processor
    print("\n3. Testing BatchProcessor...")
    def square(x):
        return x ** 2

    processor = BatchProcessor(max_workers=2)
    numbers = list(range(10))
    results = processor.process_batch(square, numbers)
    print(f"   Results: {results}")

    print("\n" + "=" * 70)
    print("✅ Performance tools test completed!")
