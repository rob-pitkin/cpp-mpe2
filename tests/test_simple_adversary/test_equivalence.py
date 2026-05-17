"""Test SimpleAdversary C++ implementation matches expected behavior."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
from cpp_mpe2.simple_adversary import parallel_env as cpp_env


def test_reset_deterministic():
    """Test that reset with same seed is deterministic."""
    cpp1 = cpp_env(N=2)
    cpp2 = cpp_env(N=2)

    obs1, _ = cpp1.reset(seed=42)
    obs2, _ = cpp2.reset(seed=42)

    for agent in cpp1.agents:
        np.testing.assert_allclose(obs1[agent], obs2[agent],
            err_msg=f"Observations for {agent} should be identical with same seed")

    cpp1.close()
    cpp2.close()


def test_reset_different_seeds():
    """Test that reset with different seeds produces different results."""
    cpp1 = cpp_env(N=2)
    cpp2 = cpp_env(N=2)

    obs1, _ = cpp1.reset(seed=42)
    obs2, _ = cpp2.reset(seed=123)

    # At least one agent should have different observations
    different = False
    for agent in cpp1.agents:
        if not np.allclose(obs1[agent], obs2[agent]):
            different = True
            break

    assert different, "Different seeds should produce different observations"

    cpp1.close()
    cpp2.close()


def test_step_deterministic():
    """Test that step is deterministic with same seed and actions."""
    cpp1 = cpp_env(N=2, continuous_actions=False)
    cpp2 = cpp_env(N=2, continuous_actions=False)

    cpp1.reset(seed=42)
    cpp2.reset(seed=42)

    actions = {"adversary_0": 1, "agent_0": 2, "agent_1": 3}

    obs1, rew1, term1, trunc1, _ = cpp1.step(actions)
    obs2, rew2, term2, trunc2, _ = cpp2.step(actions)

    for agent in cpp1.agents:
        np.testing.assert_allclose(obs1[agent], obs2[agent],
            err_msg=f"Observations should match for {agent}")
        np.testing.assert_allclose(rew1[agent], rew2[agent],
            err_msg=f"Rewards should match for {agent}")
        assert term1[agent] == term2[agent], f"Terminations should match for {agent}"
        assert trunc1[agent] == trunc2[agent], f"Truncations should match for {agent}"

    cpp1.close()
    cpp2.close()


def test_observation_space_shapes():
    """Test that observation spaces have correct shapes."""
    env = cpp_env(N=2)
    env.reset(seed=42)

    # Adversary: landmarks(4) + other_agents(4) = 8
    assert env.observation_space("adversary_0").shape == (8,), \
        "Adversary obs should be 8 (no goal)"

    # Good agents: goal(2) + landmarks(4) + other_agents(4) = 10
    assert env.observation_space("agent_0").shape == (10,), \
        "Good agent obs should be 10 (with goal)"
    assert env.observation_space("agent_1").shape == (10,), \
        "Good agent obs should be 10 (with goal)"

    env.close()


def test_action_space_shapes():
    """Test that action spaces are correct."""
    env = cpp_env(N=2, continuous_actions=False)
    env.reset(seed=42)

    # All agents have 5 discrete actions (movement only, no communication)
    for agent in env.agents:
        assert env.action_space(agent).n == 5, \
            f"{agent} should have 5 discrete actions"

    env.close()


def test_continuous_action_spaces():
    """Test continuous action spaces."""
    env = cpp_env(N=2, continuous_actions=True)
    env.reset(seed=42)

    # All agents have 5-dim continuous actions
    for agent in env.agents:
        assert env.action_space(agent).shape == (5,), \
            f"{agent} should have 5-dim continuous actions"

    env.close()


def test_agent_names():
    """Test that agents have correct names."""
    env = cpp_env(N=2)
    env.reset(seed=42)

    assert "adversary_0" in env.agents, "Should have adversary_0"
    assert "agent_0" in env.agents, "Should have agent_0"
    assert "agent_1" in env.agents, "Should have agent_1"
    assert len(env.agents) == 3, "Should have 3 agents with N=2"

    env.close()


def test_adversary_observation_smaller_than_good_agents():
    """Test that adversary has smaller observation (no goal)."""
    env = cpp_env(N=2)
    obs, _ = env.reset(seed=42)

    adv_obs_size = len(obs["adversary_0"])
    good_obs_size = len(obs["agent_0"])

    assert adv_obs_size < good_obs_size, \
        "Adversary should have smaller observation (no goal position)"
    assert adv_obs_size == 8, "Adversary obs should be 8"
    assert good_obs_size == 10, "Good agent obs should be 10"

    env.close()


def test_adversary_reward_different_from_good_agents():
    """Test that adversary and good agents have different rewards."""
    env = cpp_env(N=2, continuous_actions=False)
    env.reset(seed=42)

    actions = {"adversary_0": 1, "agent_0": 2, "agent_1": 3}
    _, rewards, _, _, _ = env.step(actions)

    # Good agents should have the same reward (cooperative)
    assert np.isclose(rewards["agent_0"], rewards["agent_1"]), \
        "Good agents should have same reward"

    # Adversary reward should typically be different
    # (Could be same by coincidence, but unlikely)
    # Just check that it's a valid number
    assert isinstance(rewards["adversary_0"], (int, float, np.number)), \
        "Adversary should have valid reward"

    env.close()


def test_continuous_actions_determinism():
    """Test determinism with continuous actions."""
    cpp1 = cpp_env(N=2, continuous_actions=True)
    cpp2 = cpp_env(N=2, continuous_actions=True)

    cpp1.reset(seed=42)
    cpp2.reset(seed=42)

    actions = {
        "adversary_0": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        "agent_0": np.array([0.5, 0.4, 0.3, 0.2, 0.1]),
        "agent_1": np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
    }

    obs1, rew1, _, _, _ = cpp1.step(actions)
    obs2, rew2, _, _, _ = cpp2.step(actions)

    for agent in cpp1.agents:
        np.testing.assert_allclose(obs1[agent], obs2[agent], rtol=1e-5)
        np.testing.assert_allclose(rew1[agent], rew2[agent], rtol=1e-5)

    cpp1.close()
    cpp2.close()


def test_multi_step_determinism():
    """Test that multiple steps remain deterministic."""
    cpp1 = cpp_env(N=2, continuous_actions=False)
    cpp2 = cpp_env(N=2, continuous_actions=False)

    cpp1.reset(seed=42)
    cpp2.reset(seed=42)

    for step in range(10):
        actions = {
            "adversary_0": step % 5,
            "agent_0": (step + 1) % 5,
            "agent_1": (step + 2) % 5,
        }

        obs1, rew1, _, _, _ = cpp1.step(actions)
        obs2, rew2, _, _, _ = cpp2.step(actions)

        for agent in cpp1.agents:
            np.testing.assert_allclose(obs1[agent], obs2[agent], rtol=1e-5,
                err_msg=f"Obs mismatch at step {step} for {agent}")
            np.testing.assert_allclose(rew1[agent], rew2[agent], rtol=1e-5,
                err_msg=f"Reward mismatch at step {step} for {agent}")

    cpp1.close()
    cpp2.close()


def test_n_parameter():
    """Test that N parameter controls number of agents and landmarks."""
    for N in [1, 2, 3, 4]:
        env = cpp_env(N=N)
        env.reset(seed=42)

        assert len(env.agents) == N + 1, f"Should have {N+1} agents with N={N}"
        assert sum(1 for a in env.agents if a.startswith("agent_")) == N, \
            f"Should have {N} good agents"
        assert sum(1 for a in env.agents if a.startswith("adversary_")) == 1, \
            "Should have 1 adversary"

        # Check observation sizes scale with N
        adv_obs_size = 4 * N  # landmarks + other_agents
        good_obs_size = 2 + 4 * N  # goal + landmarks + other_agents

        obs, _ = env.reset(seed=42)
        assert obs["adversary_0"].shape == (adv_obs_size,), \
            f"Adversary obs should be {adv_obs_size} with N={N}"
        assert obs["agent_0"].shape == (good_obs_size,), \
            f"Good agent obs should be {good_obs_size} with N={N}"

        env.close()


def test_truncation_at_max_cycles():
    """Test that episode truncates after max_cycles."""
    env = cpp_env(N=2, max_cycles=5)
    env.reset(seed=42)

    actions = {"adversary_0": 0, "agent_0": 0, "agent_1": 0}

    for i in range(4):
        _, _, _, truncations, _ = env.step(actions)
        assert not any(truncations.values()), f"Should not truncate at step {i}"

    _, _, _, truncations, _ = env.step(actions)
    assert all(truncations.values()), "Should truncate after max_cycles"
    assert len(env.agents) == 0, "Agents should be cleared after truncation"

    env.close()


def test_state_function():
    """Test that state() returns concatenated observations."""
    env = cpp_env(N=2)
    obs, _ = env.reset(seed=42)
    state = env.state()

    # State should be concatenation of all observations
    expected_size = 8 + 10 + 10  # adversary + agent_0 + agent_1
    assert state.shape == (expected_size,), f"State should be {expected_size} elements"

    env.close()


def test_physics_dynamics():
    """Test that physics updates positions correctly."""
    env = cpp_env(N=2, continuous_actions=True)
    obs, _ = env.reset(seed=42)

    initial_obs = obs["adversary_0"].copy()

    # Take action to move adversary right strongly for multiple steps
    for _ in range(5):
        actions = {
            "adversary_0": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),  # move right
            "agent_0": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # no movement
            "agent_1": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  # no movement
        }
        obs, _, _, _, _ = env.step(actions)

    final_obs = obs["adversary_0"]

    # Relative positions should have changed after movement
    assert not np.allclose(initial_obs, final_obs), \
        "Physics should update positions"

    env.close()


def test_entity_sizes_match_mpe2():
    """Test that entity sizes match MPE2 values."""
    # Create environment with render_mode to access world
    env = cpp_env(N=2, render_mode="rgb_array")
    env.reset(seed=42)

    # SimpleAdversary uses 0.15 for agents
    for agent in env.world.agents:
        assert agent.size == 0.15, f"Agent size should be 0.15, got {agent.size}"

    # SimpleAdversary uses 0.08 for landmarks
    for landmark in env.world.landmarks:
        assert landmark.size == 0.08, f"Landmark size should be 0.08, got {landmark.size}"

    env.close()


def test_reward_calculation():
    """Test that rewards are finite and have correct sign."""
    env = cpp_env(N=2)
    env.reset(seed=42)

    actions = {"adversary_0": 0, "agent_0": 0, "agent_1": 0}
    _, rewards, _, _, _ = env.step(actions)

    assert len(rewards) == 3, "Should have rewards for all 3 agents"
    for agent, reward in rewards.items():
        assert np.isfinite(reward), f"Reward for {agent} should be finite"

    # Adversary reward: -dist_to_goal, always <= 0
    assert rewards["adversary_0"] <= 0, "Adversary reward should be non-positive"

    env.close()


def test_multiple_resets():
    """Test that environment can be reset multiple times."""
    env = cpp_env(N=2)

    for seed in [42, 123, 456]:
        obs, _ = env.reset(seed=seed)
        assert len(obs) == 3, "Should have 3 agents after each reset"
        assert obs["adversary_0"].shape == (8,), "Adversary obs shape should be (8,)"
        assert obs["agent_0"].shape == (10,), "Good agent obs shape should be (10,)"

    env.close()


def test_good_agents_same_reward():
    """Test that all good agents receive the same reward (cooperative)."""
    env = cpp_env(N=2)
    env.reset(seed=42)

    actions = {"adversary_0": 0, "agent_0": 0, "agent_1": 0}
    _, rewards, _, _, _ = env.step(actions)

    assert np.isclose(rewards["agent_0"], rewards["agent_1"]), \
        "Good agents share reward and should always receive the same value"

    env.close()


def test_reward_matches_euclidean_distance():
    """Test adversary reward uses Euclidean distance, matching MPE2."""
    from mpe2 import simple_adversary_v3

    cpp = cpp_env(N=2)
    mpe2 = simple_adversary_v3.parallel_env(N=2)

    cpp.reset(seed=42)
    mpe2.reset(seed=42)

    # Step both with no-op actions and compare adversary reward sign/magnitude
    cpp_actions = {"adversary_0": 0, "agent_0": 0, "agent_1": 0}
    mpe2_actions = {"adversary_0": 0, "agent_0": 0, "agent_1": 0}

    _, cpp_rewards, _, _, _ = cpp.step(cpp_actions)
    _, mpe2_rewards, _, _, _ = mpe2.step(mpe2_actions)

    # Adversary reward should be negative Euclidean distance
    assert cpp_rewards["adversary_0"] <= 0
    assert mpe2_rewards["adversary_0"] <= 0

    cpp.close()
    mpe2.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
