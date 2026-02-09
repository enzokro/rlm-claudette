"""Tests for rlm.sandbox: SandboxBudget thread safety and limit enforcement."""

import threading
from rlm.sandbox import SandboxBudget


def test_acquire_within_budget():
    budget = SandboxBudget(max_sandboxes=3)
    assert budget.acquire() is True
    assert budget.acquire() is True
    assert budget.acquire() is True
    assert budget.remaining == 0


def test_acquire_exceeds_budget():
    budget = SandboxBudget(max_sandboxes=2)
    assert budget.acquire() is True
    assert budget.acquire() is True
    assert budget.acquire() is False


def test_remaining_decrements():
    budget = SandboxBudget(max_sandboxes=5)
    assert budget.remaining == 5
    budget.acquire()
    assert budget.remaining == 4
    budget.acquire()
    assert budget.remaining == 3


def test_can_acquire():
    budget = SandboxBudget(max_sandboxes=1)
    assert budget.can_acquire() is True
    budget.acquire()
    assert budget.can_acquire() is False


def test_zero_budget():
    budget = SandboxBudget(max_sandboxes=0)
    assert budget.acquire() is False
    assert budget.remaining == 0


def test_thread_safety():
    """Concurrent acquires should never exceed the budget."""
    budget = SandboxBudget(max_sandboxes=100)
    results = []
    barrier = threading.Barrier(20)

    def worker():
        barrier.wait()
        for _ in range(10):
            results.append(budget.acquire())

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 20 threads x 10 attempts = 200 attempts, but budget is only 100
    assert results.count(True) == 100
    assert results.count(False) == 100
    assert budget.remaining == 0
