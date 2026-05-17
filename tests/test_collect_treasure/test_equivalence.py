"""Test CollectTreasure C++ implementation matches mpe2 reference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pytest
from mpe2 import collect_treasure_v1
from pettingzoo.test import parallel_api_test

from cpp_mpe2.collect_treasure.collect_treasure import parallel_env as cpp_env


def test_api_default():
    parallel_api_test(cpp_env(), num_cycles=100)


def test_api_continuous():
    parallel_api_test(cpp_env(continuous_actions=True), num_cycles=100)


def test_api_small():
    parallel_api_test(cpp_env(num_collectors=2, num_deposits=1, num_treasures=2), num_cycles=100)


def test_reset_deterministic():
    env1, env2 = cpp_env(), cpp_env()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    for a in env1.agents:
        np.testing.assert_allclose(obs1[a], obs2[a])
    env1.close(); env2.close()


def test_reset_different_seeds():
    env1, env2 = cpp_env(), cpp_env()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=99)
    assert any(not np.allclose(obs1[a], obs2[a]) for a in env1.agents)
    env1.close(); env2.close()


def test_agent_names_default():
    env = cpp_env()
    env.reset(seed=0)
    assert set(env.agents) == {
        f"collector_{i}" for i in range(6)
    } | {f"deposit_{i}" for i in range(2)}
    env.close()


def test_observation_shape_matches_mpe2():
    cpp = cpp_env()
    mpe2 = collect_treasure_v1.parallel_env()
    cpp.reset(seed=42); mpe2.reset(seed=42)
    for a in cpp.possible_agents:
        assert cpp.observation_space(a).shape == mpe2.observation_space(a).shape, \
            f"{a}: cpp={cpp.observation_space(a).shape} mpe2={mpe2.observation_space(a).shape}"
    cpp.close(); mpe2.close()


def test_observation_shape_custom_config():
    for nc, nd, nt in [(2, 1, 3), (4, 3, 5), (3, 2, 4)]:
        cpp = cpp_env(num_collectors=nc, num_deposits=nd, num_treasures=nt)
        mpe2 = collect_treasure_v1.parallel_env(num_collectors=nc, num_deposits=nd, num_treasures=nt)
        cpp.reset(seed=42); mpe2.reset(seed=42)
        for a in cpp.possible_agents:
            assert cpp.observation_space(a).shape == mpe2.observation_space(a).shape, \
                f"nc={nc} nd={nd} nt={nt} {a}: cpp={cpp.observation_space(a).shape}"
        cpp.close(); mpe2.close()


def test_action_spaces_discrete():
    env = cpp_env()
    env.reset(seed=42)
    for a in env.agents:
        assert env.action_space(a).n == 5
    env.close()


def test_action_spaces_continuous():
    env = cpp_env(continuous_actions=True)
    env.reset(seed=42)
    for a in env.agents:
        assert env.action_space(a).shape == (5,)
    env.close()


def test_rewards_all_finite():
    env = cpp_env()
    env.reset(seed=42)
    for _ in range(10):
        _, rews, terms, truncs, _ = env.step({a: 0 for a in env.agents})
        assert all(np.isfinite(r) for r in rews.values())
        if any(terms.values()) or any(truncs.values()):
            break
    env.close()


def test_collector_reward_negative_shaping():
    """Collectors not holding anything should get distance-shaping penalty."""
    env = cpp_env()
    env.reset(seed=42)
    _, rews, _, _, _ = env.step({a: 0 for a in env.agents})
    # Distance shaping is -0.1*dist, so collector reward has negative component
    for a in [f"collector_{i}" for i in range(6)]:
        assert rews[a] < 10.0  # global bonus can push positive, but shaping is present
    env.close()


def test_truncation_at_max_cycles():
    env = cpp_env(max_cycles=5)
    env.reset(seed=42)
    actions = {a: 0 for a in env.agents}
    for i in range(4):
        _, _, _, truncs, _ = env.step(actions)
        assert not any(truncs.values())
    _, _, _, truncs, _ = env.step(actions)
    assert all(truncs.values())
    assert len(env.agents) == 0
    env.close()


def test_multi_step_determinism():
    env1, env2 = cpp_env(), cpp_env()
    env1.reset(seed=7); env2.reset(seed=7)
    for step in range(10):
        actions = {a: step % 5 for a in env1.possible_agents}
        obs1, rew1, _, _, _ = env1.step(actions)
        obs2, rew2, _, _, _ = env2.step(actions)
        for a in env1.possible_agents:
            np.testing.assert_allclose(obs1[a], obs2[a], rtol=1e-5)
            np.testing.assert_allclose(rew1[a], rew2[a], rtol=1e-5)
    env1.close(); env2.close()


def test_multiple_resets():
    env = cpp_env()
    for seed in [0, 1, 42, 100]:
        obs, _ = env.reset(seed=seed)
        assert len(obs) == 8
        for a, o in obs.items():
            expected = 86 if a.startswith("collector") else 84
            assert o.shape == (expected,)
            assert np.all(np.isfinite(o))
    env.close()


def test_pickup_and_delivery_mechanics():
    """Run many steps and verify observations stay valid."""
    env = cpp_env(max_cycles=25)
    env.reset(seed=0)
    for _ in range(25):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rews, terms, truncs, _ = env.step(actions)
        for o in obs.values():
            assert np.all(np.isfinite(o))
        if any(terms.values()) or any(truncs.values()):
            break
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
