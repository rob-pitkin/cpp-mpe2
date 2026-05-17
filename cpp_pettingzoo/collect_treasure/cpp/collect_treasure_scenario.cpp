#include <algorithm>
#include <cmath>
#include <numeric>

#include "core/entity.h"
#include "collect_treasure_scenario.h"

namespace cpp_pettingzoo::collect_treasure {

CollectTreasureScenario::CollectTreasureScenario(int num_collectors,
                                                 int num_deposits,
                                                 int num_treasures)
    : num_collectors_(num_collectors),
      num_deposits_(num_deposits),
      num_treasures_(num_treasures) {}

void CollectTreasureScenario::make_world(core::World& w) {
  w.dim_c = 2;

  // Collectors first, then deposits
  w.agents.reserve(num_collectors_ + num_deposits_);
  for (int i = 0; i < num_collectors_; ++i) {
    core::Agent a("collector_" + std::to_string(i), 0);
    a.collector = true;
    a.collide = false;
    a.silent = true;
    a.size = 0.05f;
    a.accel = 1.5f;
    a.max_speed = 1.0f;
    a.initial_mass = 1.0f;
    a.color = {0.85f, 0.85f, 0.85f};
    w.agents.push_back(std::move(a));
  }
  for (int i = 0; i < num_deposits_; ++i) {
    core::Agent a("deposit_" + std::to_string(i), 0);
    a.collector = false;
    a.deposit_type = i;
    a.collide = false;
    a.silent = true;
    a.size = 0.075f;
    a.accel = 1.5f;
    a.max_speed = 1.0f;
    a.initial_mass = 2.25f;
    const auto& c = TYPE_COLORS[i % MAX_TYPES];
    a.color = {c[0] * 0.35f, c[1] * 0.35f, c[2] * 0.35f};
    w.agents.push_back(std::move(a));
  }

  // Treasure landmarks
  w.landmarks.reserve(num_treasures_);
  for (int i = 0; i < num_treasures_; ++i) {
    core::Landmark lm("treasure_" + std::to_string(i));
    lm.collide = false;
    lm.movable = false;
    lm.size = 0.025f;
    lm.alive = true;
    lm.treasure_type = 0;
    const auto& c = TYPE_COLORS[0];
    lm.color = {c[0], c[1], c[2]};
    w.landmarks.push_back(std::move(lm));
  }
}

void CollectTreasureScenario::reset_world(core::World& w) {
  auto& rng = w.get_rng();
  auto pos_dist = std::uniform_real_distribution<float>(-1.0f, 1.0f);
  auto bound_dist = std::uniform_real_distribution<float>(-0.95f, 0.95f);
  auto type_dist = std::uniform_int_distribution<int>(0, num_deposits_ - 1);

  for (auto& agent : w.agents) {
    agent.state.p_pos = {pos_dist(rng), pos_dist(rng)};
    agent.state.p_vel = {0.0f, 0.0f};
    agent.holding = std::nullopt;
    if (agent.collector) {
      agent.color = {0.85f, 0.85f, 0.85f};
    }
  }

  for (auto& lm : w.landmarks) {
    lm.treasure_type = type_dist(rng);
    const auto& c = TYPE_COLORS[lm.treasure_type % MAX_TYPES];
    lm.color = {c[0], c[1], c[2]};
    lm.state.p_pos = {bound_dist(rng), bound_dist(rng)};
    lm.state.p_vel = {0.0f, 0.0f};
    lm.alive = true;
  }

  cached_global_collecting_ = std::nullopt;
  cached_global_deposit_ = std::nullopt;
}

void CollectTreasureScenario::post_step(core::World& w) {
  // Invalidate reward cache for this step
  cached_global_collecting_ = std::nullopt;
  cached_global_deposit_ = std::nullopt;

  auto& rng = w.get_rng();
  auto bound_dist = std::uniform_real_distribution<float>(-0.95f, 0.95f);
  auto type_dist = std::uniform_int_distribution<int>(0, num_deposits_ - 1);

  // --- Treasure pickup ---
  for (auto& lm : w.landmarks) {
    if (lm.alive) {
      for (auto& agent : w.agents) {
        if (agent.collector && !agent.holding.has_value() &&
            is_collision(lm, agent)) {
          lm.alive = false;
          agent.holding = lm.treasure_type;
          const auto& c = TYPE_COLORS[lm.treasure_type % MAX_TYPES];
          agent.color = {c[0] * 0.85f, c[1] * 0.85f, c[2] * 0.85f};
          lm.state.p_pos = {-999.0f, -999.0f};
          break;
        }
      }
    } else {
      // Respawn (respawn_prob == 1.0)
      lm.treasure_type = type_dist(rng);
      const auto& c = TYPE_COLORS[lm.treasure_type % MAX_TYPES];
      lm.color = {c[0], c[1], c[2]};
      lm.state.p_pos = {bound_dist(rng), bound_dist(rng)};
      lm.alive = true;
    }
  }

  // --- Deposit delivery ---
  for (auto& agent : w.agents) {
    if (agent.collector && agent.holding.has_value()) {
      for (auto& deposit : w.agents) {
        if (!deposit.collector && deposit.deposit_type == agent.holding.value() &&
            is_collision(agent, deposit)) {
          agent.holding = std::nullopt;
          agent.color = {0.85f, 0.85f, 0.85f};
          break;
        }
      }
    }
  }
}

bool CollectTreasureScenario::is_collision(const core::Entity& a,
                                           const core::Entity& b) const {
  float dx = a.state.p_pos[0] - b.state.p_pos[0];
  float dy = a.state.p_pos[1] - b.state.p_pos[1];
  float dist = std::sqrt(dx * dx + dy * dy);
  return dist < (a.size + b.size);
}

float CollectTreasureScenario::global_collecting_reward(
    const core::World& w) const {
  if (cached_global_collecting_.has_value()) return *cached_global_collecting_;
  float rew = 0.0f;
  for (const auto& lm : w.landmarks) {
    if (lm.alive) {
      for (const auto& agent : w.agents) {
        if (agent.collector && !agent.holding.has_value() &&
            is_collision(lm, agent)) {
          rew += 5.0f;
        }
      }
    }
  }
  cached_global_collecting_ = rew;
  return rew;
}

float CollectTreasureScenario::global_deposit_reward(
    const core::World& w) const {
  if (cached_global_deposit_.has_value()) return *cached_global_deposit_;
  float rew = 0.0f;
  for (const auto& deposit : w.agents) {
    if (!deposit.collector) {
      for (const auto& agent : w.agents) {
        if (agent.collector && agent.holding.has_value() &&
            agent.holding.value() == deposit.deposit_type &&
            is_collision(agent, deposit)) {
          rew += 5.0f;
        }
      }
    }
  }
  cached_global_deposit_ = rew;
  return rew;
}

float CollectTreasureScenario::global_reward(const core::World& w) const {
  return global_collecting_reward(w) + global_deposit_reward(w);
}

float CollectTreasureScenario::reward(const core::Agent& agent,
                                      const core::World& w) const {
  float rew = 0.0f;

  if (agent.collector) {
    // Penalise overlaps with other collectors
    for (const auto& other : w.agents) {
      if (&other != &agent && other.collector && is_collision(agent, other)) {
        rew -= 5.0f;
      }
    }

    // Distance shaping
    if (!agent.holding.has_value()) {
      float min_dist = std::numeric_limits<float>::infinity();
      for (const auto& lm : w.landmarks) {
        if (lm.alive) {
          float dx = lm.state.p_pos[0] - agent.state.p_pos[0];
          float dy = lm.state.p_pos[1] - agent.state.p_pos[1];
          min_dist = std::min(min_dist, std::sqrt(dx * dx + dy * dy));
        }
      }
      if (std::isfinite(min_dist)) rew -= 0.1f * min_dist;
    } else {
      float min_dist = std::numeric_limits<float>::infinity();
      for (const auto& deposit : w.agents) {
        if (!deposit.collector && deposit.deposit_type == agent.holding.value()) {
          float dx = deposit.state.p_pos[0] - agent.state.p_pos[0];
          float dy = deposit.state.p_pos[1] - agent.state.p_pos[1];
          min_dist = std::min(min_dist, std::sqrt(dx * dx + dy * dy));
        }
      }
      if (std::isfinite(min_dist)) rew -= 0.1f * min_dist;
    }
  } else {
    // Deposit agent shaping
    bool has_carrier = false;
    float min_dist = std::numeric_limits<float>::infinity();
    for (const auto& collector : w.agents) {
      if (collector.collector && collector.holding.has_value() &&
          collector.holding.value() == agent.deposit_type) {
        has_carrier = true;
        float dx = collector.state.p_pos[0] - agent.state.p_pos[0];
        float dy = collector.state.p_pos[1] - agent.state.p_pos[1];
        min_dist = std::min(min_dist, std::sqrt(dx * dx + dy * dy));
      }
    }

    if (has_carrier) {
      rew -= 0.1f * min_dist;
    } else {
      // Move toward centroid of all collectors
      float cx = 0.0f, cy = 0.0f;
      int n = 0;
      for (const auto& collector : w.agents) {
        if (collector.collector) {
          cx += collector.state.p_pos[0];
          cy += collector.state.p_pos[1];
          ++n;
        }
      }
      if (n > 0) {
        cx /= n; cy /= n;
        float dx = cx - agent.state.p_pos[0];
        float dy = cy - agent.state.p_pos[1];
        rew -= 0.1f * std::sqrt(dx * dx + dy * dy);
      }
    }
  }

  rew += global_reward(w);
  return rew;
}

std::vector<float> CollectTreasureScenario::agent_encoding(
    const core::Agent& a) const {
  std::vector<float> enc(2 * num_deposits_, 0.0f);
  if (!a.collector) {
    // deposit_type one-hot in first half
    if (a.deposit_type >= 0 && a.deposit_type < num_deposits_)
      enc[a.deposit_type] = 1.0f;
  } else {
    // holding one-hot in second half
    if (a.holding.has_value() && a.holding.value() < num_deposits_)
      enc[num_deposits_ + a.holding.value()] = 1.0f;
  }
  return enc;
}

std::vector<float> CollectTreasureScenario::observation(
    const core::Agent& agent, const core::World& w) const {
  std::vector<float> obs;

  // Self: pos + vel
  obs.push_back(agent.state.p_pos[0]);
  obs.push_back(agent.state.p_pos[1]);
  obs.push_back(agent.state.p_vel[0]);
  obs.push_back(agent.state.p_vel[1]);

  // Collectors encode holding one-hot
  if (agent.collector) {
    for (int t = 0; t < num_deposits_; ++t)
      obs.push_back((agent.holding.has_value() && agent.holding.value() == t) ? 1.0f : 0.0f);
  }

  // Other agents sorted by distance (closest first)
  std::vector<const core::Agent*> others;
  others.reserve(w.agents.size() - 1);
  for (const auto& other : w.agents) {
    if (&other != &agent) others.push_back(&other);
  }
  std::sort(others.begin(), others.end(),
            [&](const core::Agent* a, const core::Agent* b) {
              auto dist = [&](const core::Agent* x) {
                float dx = x->state.p_pos[0] - agent.state.p_pos[0];
                float dy = x->state.p_pos[1] - agent.state.p_pos[1];
                return dx * dx + dy * dy;
              };
              return dist(a) < dist(b);
            });

  for (const auto* other : others) {
    obs.push_back(other->state.p_pos[0] - agent.state.p_pos[0]);
    obs.push_back(other->state.p_pos[1] - agent.state.p_pos[1]);
    obs.push_back(other->state.p_vel[0]);
    obs.push_back(other->state.p_vel[1]);
    auto enc = agent_encoding(*other);
    obs.insert(obs.end(), enc.begin(), enc.end());
  }

  // Treasures: alive ones sorted by distance first, then dead ones (zero-padded)
  std::vector<const core::Landmark*> alive_lms, dead_lms;
  for (const auto& lm : w.landmarks) {
    if (lm.alive) alive_lms.push_back(&lm);
    else dead_lms.push_back(&lm);
  }
  std::sort(alive_lms.begin(), alive_lms.end(),
            [&](const core::Landmark* a, const core::Landmark* b) {
              auto dist = [&](const core::Landmark* x) {
                float dx = x->state.p_pos[0] - agent.state.p_pos[0];
                float dy = x->state.p_pos[1] - agent.state.p_pos[1];
                return dx * dx + dy * dy;
              };
              return dist(a) < dist(b);
            });

  for (const auto* lm : alive_lms) {
    obs.push_back(lm->state.p_pos[0] - agent.state.p_pos[0]);
    obs.push_back(lm->state.p_pos[1] - agent.state.p_pos[1]);
    for (int t = 0; t < num_deposits_; ++t)
      obs.push_back(lm->treasure_type == t ? 1.0f : 0.0f);
  }
  for (size_t i = 0; i < dead_lms.size(); ++i) {
    for (int k = 0; k < 2 + num_deposits_; ++k) obs.push_back(0.0f);
  }

  return obs;
}

}  // namespace cpp_pettingzoo::collect_treasure
