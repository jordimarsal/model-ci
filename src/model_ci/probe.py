# src/model_ci/probe.py
"""probe — drift math, machine guards, stabilized-generation measurement."""

from statistics import median


def drift(baseline: float, current: float, tolerance: float = 0.10) -> bool:
    """Return True when |current - baseline| / baseline > tolerance.

    Both directions count: a fleet that got faster also deserves a full bench.
    baseline <= 0 means "no baseline yet" → False.
    """
    if baseline <= 0:
        return False
    return abs(current - baseline) / baseline > tolerance


def stabilized_tps(
    complete: callable[[int], float],
    generations: int = 3,
    tokens: int = 512,
) -> float:
    """Run `complete(tokens)` *generations* times and return the MEDIAN tps.

    Median, never mean: Strata's micro-generations swing ±20% and a single
    outlier is a lie.
    """
    samples = [complete(tokens) for _ in range(generations)]
    return median(samples)


def parse_compute_apps(csv_text: str) -> list[tuple[int, int]]:
    """Parse ``nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`` output.

    Lines look like ``1234, 392 MiB`` (comma-space separated, MiB suffix).
    Blank lines are skipped.
    """
    results: list[tuple[int, int]] = []
    for line in csv_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_str, mem_str = stripped.split(",", 1)
        pid = int(pid_str.strip())
        # strip "MiB" suffix and any whitespace
        mem_str = mem_str.strip().removesuffix(" MiB")
        memory = int(mem_str)
        results.append((pid, memory))
    return results


def quiet_machine(
    load1: float,
    nproc: int,
    compute_apps: list[tuple[int, int]],
    allow: set[int] = frozenset(),
    max_load_ratio: float = 0.5,
    min_free_mib: int = 2000,
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` about whether the machine is quiet enough to bench.

    ok when:
      - load1 <= nproc * max_load_ratio
      - every compute-app PID NOT in ``allow`` holds < min_free_mib (MiB).
    """
    if load1 > nproc * max_load_ratio:
        return False, f"load {load1:.1f} exceeds threshold {nproc * max_load_ratio:.1f}"

    for pid, memory in compute_apps:
        if pid not in allow and memory >= min_free_mib:
            return False, f"PID {pid} using {memory} MiB (exceeds {min_free_mib})"

    return True, "quiet"


def gpu_rule(llama_alive: bool, strata_alive: bool) -> tuple[bool, str]:
    """Golden rule: both alive → conflict; anything else → ok."""
    if llama_alive and strata_alive:
        return False, "both llama and strata are running on GPU — cannot bench"
    return True, "gpu clear"
