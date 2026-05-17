"""Test CollectTreasure parallel API compliance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pettingzoo.test import parallel_api_test

from cpp_mpe2.collect_treasure.collect_treasure import parallel_env


def test_parallel_api_default():
    parallel_api_test(parallel_env(), num_cycles=100)


def test_parallel_api_continuous():
    parallel_api_test(parallel_env(continuous_actions=True), num_cycles=100)


def test_parallel_api_small():
    parallel_api_test(
        parallel_env(num_collectors=2, num_deposits=1, num_treasures=2), num_cycles=100
    )


def test_parallel_api_three_types():
    parallel_api_test(
        parallel_env(num_collectors=3, num_deposits=3, num_treasures=4), num_cycles=100
    )


def test_parallel_api_short_episodes():
    parallel_api_test(parallel_env(max_cycles=5), num_cycles=20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
