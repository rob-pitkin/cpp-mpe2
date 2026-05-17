"""Test SimpleLine parallel API compliance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pettingzoo.test import parallel_api_test

from cpp_pettingzoo.simple_line.simple_line import parallel_env


def test_parallel_api_default():
    parallel_api_test(parallel_env(N=4), num_cycles=100)


def test_parallel_api_continuous():
    parallel_api_test(parallel_env(N=4, continuous_actions=True), num_cycles=100)


def test_parallel_api_n2():
    parallel_api_test(parallel_env(N=2), num_cycles=100)


def test_parallel_api_n6():
    parallel_api_test(parallel_env(N=6), num_cycles=100)


def test_parallel_api_terminate():
    parallel_api_test(parallel_env(N=4, terminate_on_success=True), num_cycles=100)


def test_parallel_api_short_episodes():
    parallel_api_test(parallel_env(N=4, max_cycles=5), num_cycles=20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
