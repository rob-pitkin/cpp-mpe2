"""Test AEC wrapper for SimpleAdversary environment."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pettingzoo.test import api_test
from cpp_mpe2.simple_adversary import env as aec_env


def test_aec_api():
    """Test that AEC wrapper complies with PettingZoo API."""
    environment = aec_env(N=2)
    api_test(environment, num_cycles=100, verbose_progress=False)


def test_aec_discrete():
    """Test AEC wrapper with discrete actions."""
    environment = aec_env(N=2, continuous_actions=False)
    environment.reset(seed=42)

    for agent in environment.agent_iter(max_iter=20):
        observation, reward, termination, truncation, info = environment.last()

        if termination or truncation:
            action = None
        else:
            action = environment.action_space(agent).sample()

        environment.step(action)

    environment.close()


def test_aec_continuous():
    """Test AEC wrapper with continuous actions."""
    environment = aec_env(N=2, continuous_actions=True)
    environment.reset(seed=42)

    for agent in environment.agent_iter(max_iter=20):
        observation, reward, termination, truncation, info = environment.last()

        if termination or truncation:
            action = None
        else:
            action = environment.action_space(agent).sample()

        environment.step(action)

    environment.close()


def test_aec_n_parameter():
    """Test AEC wrapper with different N values."""
    for N in [1, 2, 3]:
        environment = aec_env(N=N)
        environment.reset(seed=42)

        for agent in environment.agent_iter(max_iter=10):
            observation, reward, termination, truncation, info = environment.last()

            if termination or truncation:
                action = None
            else:
                action = 0  # noop

            environment.step(action)

        environment.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
