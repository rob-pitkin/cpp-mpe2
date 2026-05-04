"""Test PettingZoo API compliance for SimpleAdversary environment."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from pettingzoo.test import parallel_api_test
from cpp_pettingzoo.simple_adversary import parallel_env


def test_api():
    """Test that environment complies with PettingZoo parallel API."""
    env = parallel_env(N=2)
    parallel_api_test(env, num_cycles=100)


def test_action_spaces():
    """Test that action spaces are correct."""
    env = parallel_env(N=2, continuous_actions=False)
    env.reset(seed=42)

    for agent in env.agents:
        assert env.action_space(agent).n == 5, \
            f"{agent} should have 5 discrete actions (movement only)"

    env.close()


def test_continuous_action_spaces():
    """Test that continuous action spaces are correct."""
    env = parallel_env(N=2, continuous_actions=True)
    env.reset(seed=42)

    for agent in env.agents:
        assert env.action_space(agent).shape == (5,), \
            f"{agent} should have 5-dim continuous actions"

    env.close()


def test_observation_spaces():
    """Test that observation spaces are asymmetric."""
    env = parallel_env(N=2)
    env.reset(seed=42)

    assert env.observation_space("adversary_0").shape == (8,), \
        "Adversary observation should be 8 (no goal)"
    assert env.observation_space("agent_0").shape == (10,), \
        "Good agent observation should be 10 (with goal)"
    assert env.observation_space("agent_1").shape == (10,), \
        "Good agent observation should be 10 (with goal)"

    env.close()


def test_num_agents():
    """Test that environment has correct number of agents."""
    env = parallel_env(N=2)
    env.reset(seed=42)

    assert len(env.agents) == 3, "Should have 3 agents with N=2"
    assert "adversary_0" in env.agents, "Should have adversary_0"
    assert "agent_0" in env.agents, "Should have agent_0"
    assert "agent_1" in env.agents, "Should have agent_1"

    env.close()


def test_n_parameter():
    """Test that N parameter controls agents and landmarks."""
    for N in [1, 2, 3]:
        env = parallel_env(N=N)
        env.reset(seed=42)

        assert len(env.agents) == N + 1, f"Should have {N+1} agents"
        env.close()


def test_max_cycles():
    """Test that max_cycles parameter controls episode length."""
    env = parallel_env(N=2, max_cycles=10)
    env.reset(seed=42)

    for _ in range(10):
        actions = {"adversary_0": 0, "agent_0": 0, "agent_1": 0}
        _, _, _, truncations, _ = env.step(actions)

    assert all(truncations.values()), "Episode should truncate after max_cycles"

    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
