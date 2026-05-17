#include <algorithm>
#include <cmath>

#include "core/entity.h"
#include "core/third_party/munkres/munkres.h"
#include "simple_line_scenario.h"

namespace cpp_pettingzoo::simple_line {

SimpleLineScenario::SimpleLineScenario(bool terminate_on_success)
    : terminate_on_success_(terminate_on_success) {}

void SimpleLineScenario::make_world(core::World& w, int N) {
  w.dim_c = 0;
  w.agents.reserve(N);
  for (int i = 0; i < N; ++i) {
    core::Agent agent("agent_" + std::to_string(i), 0);
    agent.collide = true;
    agent.silent = true;
    agent.size = 0.03f;
    w.agents.push_back(std::move(agent));
  }

  for (int i = 0; i < 2; ++i) {
    core::Landmark lm("landmark_" + std::to_string(i));
    lm.collide = false;
    lm.movable = false;
    lm.size = 0.02f;
    w.landmarks.push_back(std::move(lm));
  }
}

void SimpleLineScenario::reset_world(core::World& w) {
  auto& rng = w.get_rng();
  auto agent_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  auto lm_dist = std::uniform_real_distribution<float>(-0.25f, 0.25f);
  auto theta_dist = std::uniform_real_distribution<float>(
      0.0f, 2.0f * static_cast<float>(M_PI));

  for (auto& agent : w.agents) {
    agent.color = {0.35f, 0.35f, 0.85f};
    agent.state.p_pos = {agent_dist(rng), agent_dist(rng)};
    agent.state.p_vel = {0.0f, 0.0f};
  }

  for (auto& lm : w.landmarks) {
    lm.color = {0.25f, 0.25f, 0.25f};
    lm.state.p_vel = {0.0f, 0.0f};
  }

  // Place landmark 0 near centre
  w.landmarks[0].state.p_pos = {lm_dist(rng), lm_dist(rng)};

  // Rotate landmark 1 outward from landmark 0 until it lands in [-1, 1]^2
  float theta = theta_dist(rng);
  const float step = 5.0f * static_cast<float>(M_PI) / 180.0f;
  float lm1x, lm1y;
  do {
    float dx = std::cos(theta) * TOTAL_SEP;
    float dy = std::sin(theta) * TOTAL_SEP;
    lm1x = w.landmarks[0].state.p_pos[0] + dx;
    lm1y = w.landmarks[0].state.p_pos[1] + dy;
    theta += step;
  } while (std::abs(lm1x) >= 1.0f || std::abs(lm1y) >= 1.0f);
  w.landmarks[1].state.p_pos = {lm1x, lm1y};

  // Compute evenly-spaced target positions along the segment (fixed for episode)
  const int N = static_cast<int>(w.agents.size());
  const float lm0x = w.landmarks[0].state.p_pos[0];
  const float lm0y = w.landmarks[0].state.p_pos[1];
  const float dirx = (lm1x - lm0x) / TOTAL_SEP;
  const float diry = (lm1y - lm0y) / TOTAL_SEP;
  const float ideal_sep = (N > 1) ? TOTAL_SEP / (N - 1) : 0.0f;

  expected_positions_.resize(N);
  for (int i = 0; i < N; ++i) {
    expected_positions_[i] = {lm0x + i * ideal_sep * dirx,
                              lm0y + i * ideal_sep * diry};
  }

  cache_valid_ = false;
  joint_reward_ = 0.0f;
  delta_dists_.clear();
}

void SimpleLineScenario::compute_line(const core::World& world) const {
  if (cache_valid_) return;

  const int N = static_cast<int>(world.agents.size());

  // N×N cost matrix: cost[i][j] = dist(agent_i, target_j)
  Matrix<float> cost(N, N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float dx = world.agents[i].state.p_pos[0] - expected_positions_[j][0];
      float dy = world.agents[i].state.p_pos[1] - expected_positions_[j][1];
      cost(i, j) = std::sqrt(dx * dx + dy * dy);
    }
  }

  Matrix<float> cost_copy = cost;
  Munkres<float> solver;
  solver.solve(cost);  // 0 = assigned, -1 = not assigned

  delta_dists_.resize(N);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (cost(i, j) == 0) {
        delta_dists_[i] = cost_copy(i, j);
        break;
      }
    }
  }

  float sum = 0.0f;
  for (float d : delta_dists_) sum += std::min(d, 2.0f);
  joint_reward_ = -(sum / N);

  cache_valid_ = true;
}

float SimpleLineScenario::global_reward(const core::World& world) const {
  compute_line(world);
  return joint_reward_;
}

bool SimpleLineScenario::is_terminal(const core::World& world) const {
  if (!terminate_on_success_) return false;
  compute_line(world);
  for (float d : delta_dists_) {
    if (d >= DIST_THRESHOLD) return false;
  }
  return true;
}

std::vector<float> SimpleLineScenario::observation(
    const core::Agent& agent, const core::World& world) const {
  std::vector<float> obs;
  obs.push_back(agent.state.p_vel[0]);
  obs.push_back(agent.state.p_vel[1]);
  obs.push_back(agent.state.p_pos[0]);
  obs.push_back(agent.state.p_pos[1]);
  // Relative positions to both landmarks
  for (const auto& lm : world.landmarks) {
    obs.push_back(lm.state.p_pos[0] - agent.state.p_pos[0]);
    obs.push_back(lm.state.p_pos[1] - agent.state.p_pos[1]);
  }
  return obs;
}

}  // namespace cpp_pettingzoo::simple_line
