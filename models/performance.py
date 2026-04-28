"""Performance monitoring utilities for training and inference.

Measures: wall-clock time, peak GPU/MPS memory, CPU RAM usage, samples/sec.
Saves results to performance.json alongside results.json / final_adapter/.

Usage (training):
    monitor = PerformanceMonitor(device)
    monitor.start()
    trainer.train()
    stats = monitor.stop(n_samples=len(train_dataset))
    monitor.save(output_dir / "performance.json", extra={"mode": "training"})

Usage (inference):
    monitor = PerformanceMonitor(device)
    monitor.start()
    for example in val_set:
        monitor.start_example()
        model.generate(...)
        monitor.end_example()
    stats = monitor.stop(n_samples=len(val_set))
    monitor.save(output_dir / "performance.json", extra={"mode": "inference"})
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _get_peak_memory_mb(device: str) -> float:
    """Return peak memory usage in MB for the given device."""
    try:
        import torch
        if device == "cuda":
            return torch.cuda.max_memory_allocated() / 1024 ** 2
        elif device == "mps":
            return torch.mps.current_allocated_memory() / 1024 ** 2
    except Exception:
        pass
    return 0.0


def _get_cpu_ram_mb() -> float:
    """Return current process RSS memory in MB."""
    try:
        import psutil, os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 2
    except ImportError:
        pass
    return 0.0


def _reset_peak_memory(device: str) -> None:
    try:
        import torch
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass


class PerformanceMonitor:
    """Context-manager-style performance tracker.

    Records:
        - total_time_sec       — wall-clock time from start() to stop()
        - peak_gpu_memory_mb   — peak GPU/MPS memory allocated
        - cpu_ram_mb           — process RAM at stop() time
        - throughput_per_sec   — samples per second (if n_samples provided)
        - avg_latency_ms       — average per-example latency (inference only)
        - min_latency_ms       — fastest example
        - max_latency_ms       — slowest example
    """

    def __init__(self, device: str):
        self.device = device
        self._start_time: float = 0.0
        self._example_start: float = 0.0
        self._example_latencies: list[float] = []
        self.stats: dict[str, Any] = {}

    def start(self) -> None:
        _reset_peak_memory(self.device)
        self._start_time = time.perf_counter()

    def start_example(self) -> None:
        """Call before each inference example to track per-example latency."""
        self._example_start = time.perf_counter()

    def end_example(self) -> None:
        """Call after each inference example."""
        elapsed_ms = (time.perf_counter() - self._example_start) * 1000
        self._example_latencies.append(elapsed_ms)

    def stop(self, n_samples: int | None = None) -> dict[str, Any]:
        total_time = time.perf_counter() - self._start_time
        peak_gpu = _get_peak_memory_mb(self.device)
        cpu_ram = _get_cpu_ram_mb()

        self.stats = {
            "device": self.device,
            "total_time_sec": round(total_time, 2),
            "peak_gpu_memory_mb": round(peak_gpu, 1),
            "cpu_ram_mb": round(cpu_ram, 1),
        }

        if n_samples and total_time > 0:
            self.stats["throughput_per_sec"] = round(n_samples / total_time, 3)

        if self._example_latencies:
            self.stats["avg_latency_ms"] = round(
                sum(self._example_latencies) / len(self._example_latencies), 2
            )
            self.stats["min_latency_ms"] = round(min(self._example_latencies), 2)
            self.stats["max_latency_ms"] = round(max(self._example_latencies), 2)

        return self.stats

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        """Save performance stats to a JSON file."""
        payload = {**(extra or {}), **self.stats}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        self._print_summary()
        print(f"Performance stats saved to {path}")

    def _print_summary(self) -> None:
        s = self.stats
        print("\nPERFORMANCE SUMMARY")
        print(f"  Device:              {s.get('device')}")
        print(f"  Total time:          {s.get('total_time_sec')} sec")
        print(f"  Peak GPU/MPS memory: {s.get('peak_gpu_memory_mb')} MB")
        print(f"  CPU RAM:             {s.get('cpu_ram_mb')} MB")
        if "throughput_per_sec" in s:
            print(f"  Throughput:          {s.get('throughput_per_sec')} samples/sec")
        if "avg_latency_ms" in s:
            print(f"  Avg latency:         {s.get('avg_latency_ms')} ms/example")
            print(f"  Min / Max latency:   {s.get('min_latency_ms')} / {s.get('max_latency_ms')} ms")
