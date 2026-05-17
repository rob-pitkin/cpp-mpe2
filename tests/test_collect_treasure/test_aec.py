"""Test CollectTreasure AEC API compatibility."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from pettingzoo.test import api_test as aec_api_test

from cpp_pettingzoo.collect_treasure.collect_treasure import env as treasure_env


def test_aec_api():
    aec_api_test(treasure_env(), num_cycles=10, verbose_progress=False)


def test_aec_basic_attributes():
    env = treasure_env()
    env.reset()
    assert hasattr(env, 'agents')
    assert hasattr(env, 'num_agents')
    assert hasattr(env, 'agent_selection')
    assert hasattr(env, 'rewards')
    assert hasattr(env, 'terminations')
    assert hasattr(env, 'truncations')
    assert hasattr(env, 'infos')
    env.close()


def test_aec_reset():
    env = treasure_env()
    env.reset(seed=42)
    assert len(env.agents) == 8  # 6 collectors + 2 deposits
    assert env.agent_selection in env.agents
    env.close()


def test_aec_step_cycle():
    env = treasure_env(max_cycles=5)
    env.reset(seed=42)
    steps = 0
    while env.agents and steps < 5 * 8:
        agent = env.agent_selection
        _, _, term, trunc, _ = env.last()
        env.step(None if (term or trunc) else env.action_space(agent).sample())
        steps += 1
    env.close()


def test_aec_deterministic():
    env1 = treasure_env()
    env2 = treasure_env()
    env1.reset(seed=123); env2.reset(seed=123)
    obs1, _, _, _, _ = env1.last()
    obs2, _, _, _, _ = env2.last()
    np.testing.assert_allclose(obs1, obs2)
    env1.close(); env2.close()


def test_aec_observation_action_spaces():
    env = treasure_env()
    env.reset()
    for a in [f"collector_{i}" for i in range(6)]:
        assert env.observation_space(a).shape == (86,)
        assert env.action_space(a).n == 5
    for a in [f"deposit_{i}" for i in range(2)]:
        assert env.observation_space(a).shape == (84,)
        assert env.action_space(a).n == 5
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
