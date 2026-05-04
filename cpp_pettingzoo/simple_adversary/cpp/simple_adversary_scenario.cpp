#include <cmath>
#include <random>

#include "core/entity.h"
#include "simple_adversary_scenario.h"

namespace cpp_pettingzoo::simple_adversary {

SimpleAdversaryScenario::SimpleAdversaryScenario(size_t n) : n_(n) {}

void SimpleAdversaryScenario::make_world(core::World& w) {
  w.dim_c = 2;

  core::Agent adversary = core::Agent("adversary_0", w.dim_c);
  adversary.adversary = true;
  adversary.collide = false;
  adversary.silent = true;
  adversary.size = 0.15f;
  w.agents.push_back(std::move(adversary));

  for (size_t i = 0; i < n_; ++i) {
    core::Agent a = core::Agent("agent_" + std::to_string(i), w.dim_c);
    a.collide = false;
    a.silent = true;
    a.size = 0.15f;
    w.agents.push_back(std::move(a));

    core::Landmark l = core::Landmark("landmark " + std::to_string(i));
    l.collide = false;
    l.movable = false;
    l.size = 0.08f;
    w.landmarks.push_back(std::move(l));
  }
}

void SimpleAdversaryScenario::reset_world(core::World& w) {
  auto& rng = w.get_rng();
  auto dist = std::uniform_int_distribution<>(0, n_ - 1);
  auto float_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);

  w.agents[0].color = {0.85f, 0.35f, 0.35f};
  w.agents[0].state.p_vel = {0.0f, 0.0f};
  w.agents[0].state.p_pos = {float_dist(rng), float_dist(rng)};

  auto* goal_landmark = &w.landmarks[dist(rng)];
  w.agents[0].goal_a = goal_landmark;

  for (size_t i = 1; i < n_ + 1; ++i) {
    w.agents[i].color = {0.35f, 0.35f, 0.85f};
    w.agents[i].goal_a = goal_landmark;
    w.agents[i].state.p_vel = {0.0f, 0.0f};
    w.agents[i].state.p_pos = {float_dist(rng), float_dist(rng)};
  }

  for (size_t i = 0; i < n_; ++i) {
    if (goal_landmark == &w.landmarks[i]) {
      w.landmarks[i].color = {0.15f, 0.65f, 0.15f};
    } else {
      w.landmarks[i].color = {0.15f, 0.15f, 0.15f};
    }
    w.landmarks[i].state.p_pos = {float_dist(rng), float_dist(rng)};
    w.landmarks[i].state.p_vel = {0.0f, 0.0f};
  }
}

void SimpleAdversaryScenario::compute_reward_cache(
    const core::World& world) const {
  if (reward_cache_valid_) return;

  // min distance from any good agent to the goal
  cached_min_good_dist_ = std::numeric_limits<float>::max();
  for (size_t i = 1; i < n_ + 1; ++i) {
    float dx = world.agents[i].state.p_pos[0] -
               world.agents[i].goal_a->state.p_pos[0];
    float dy = world.agents[i].state.p_pos[1] -
               world.agents[i].goal_a->state.p_pos[1];
    cached_min_good_dist_ =
        std::min(cached_min_good_dist_, std::sqrt(dx * dx + dy * dy));
  }

  // adversary distance to goal
  float adv_dx =
      world.agents[0].state.p_pos[0] - world.agents[0].goal_a->state.p_pos[0];
  float adv_dy =
      world.agents[0].state.p_pos[1] - world.agents[0].goal_a->state.p_pos[1];
  cached_adv_dist_ = std::sqrt(adv_dx * adv_dx + adv_dy * adv_dy);

  reward_cache_valid_ = true;
}

float SimpleAdversaryScenario::reward(const core::Agent& agent,
                                      const core::World& world) const {
  if (agent.adversary) {
    // Adversary is always first; invalidate cache so good agents recompute fresh
    reward_cache_valid_ = false;
    float dx = agent.state.p_pos[0] - agent.goal_a->state.p_pos[0];
    float dy = agent.state.p_pos[1] - agent.goal_a->state.p_pos[1];
    return -std::sqrt(dx * dx + dy * dy);
  }
  compute_reward_cache(world);
  return -cached_min_good_dist_ + cached_adv_dist_;
}

float SimpleAdversaryScenario::global_reward(const core::World& w) const {
  // Invalidate cache at the start of a new reward computation pass
  reward_cache_valid_ = false;
  float glob_reward = 0.0f;
  for (const auto& agent : w.agents) {
    glob_reward += reward(agent, w);
  }
  reward_cache_valid_ = false;  // reset for next call
  return glob_reward / static_cast<float>(w.agents.size());
}

std::vector<float> SimpleAdversaryScenario::observation(
    const core::Agent& agent, const core::World& world) const {
  std::vector<float> obs;

  if (agent.adversary) {
    obs.reserve(4 * n_);
  } else {
    obs.reserve(2 + 4 * n_);
    float goal_x = agent.goal_a->state.p_pos[0] - agent.state.p_pos[0];
    float goal_y = agent.goal_a->state.p_pos[1] - agent.state.p_pos[1];
    obs.push_back(goal_x);
    obs.push_back(goal_y);
  }

  for (const auto& l : world.landmarks) {
    float rel_x = l.state.p_pos[0] - agent.state.p_pos[0];
    float rel_y = l.state.p_pos[1] - agent.state.p_pos[1];
    obs.push_back(rel_x);
    obs.push_back(rel_y);
  }

  for (const auto& a : world.agents) {
    if (&a == &agent) {
      continue;
    }
    float rel_x = a.state.p_pos[0] - agent.state.p_pos[0];
    float rel_y = a.state.p_pos[1] - agent.state.p_pos[1];
    obs.push_back(rel_x);
    obs.push_back(rel_y);
  }
  return obs;
}

}  // namespace cpp_pettingzoo::simple_adversary
