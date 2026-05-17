#ifndef COLLECT_TREASURE_SCENARIO_H_
#define COLLECT_TREASURE_SCENARIO_H_

#include <array>
#include <optional>
#include <vector>

#include "../../../core/scenario.h"

namespace cpp_mpe2::collect_treasure {

// Per-type color palette (matches mpe2 _TYPE_COLORS)
static constexpr int MAX_TYPES = 6;
static constexpr std::array<std::array<float, 3>, MAX_TYPES> TYPE_COLORS = {{
    {0.212f, 0.408f, 0.776f},  // blue
    {0.945f, 0.553f, 0.000f},  // orange
    {0.169f, 0.627f, 0.173f},  // green
    {0.839f, 0.149f, 0.157f},  // red
    {0.580f, 0.404f, 0.741f},  // purple
    {0.549f, 0.337f, 0.294f},  // brown
}};

class CollectTreasureScenario : public core::Scenario {
 public:
  CollectTreasureScenario(int num_collectors = 6, int num_deposits = 2,
                          int num_treasures = 6);

  void make_world(core::World& w) override;
  void reset_world(core::World& w) override;
  void post_step(core::World& w) override;

  float reward(const core::Agent& agent,
               const core::World& world) const override;
  std::vector<float> observation(const core::Agent& agent,
                                 const core::World& world) const override;

 private:
  int num_collectors_;
  int num_deposits_;
  int num_treasures_;

  // Cached per-step global rewards (reset in post_step)
  mutable std::optional<float> cached_global_collecting_;
  mutable std::optional<float> cached_global_deposit_;

  bool is_collision(const core::Entity& a, const core::Entity& b) const;
  float global_collecting_reward(const core::World& w) const;
  float global_deposit_reward(const core::World& w) const;
  float global_reward(const core::World& w) const override;

  // Agent encoding: [deposit_type_onehot | holding_onehot] (2 * num_deposits)
  std::vector<float> agent_encoding(const core::Agent& a) const;
};

}  // namespace cpp_mpe2::collect_treasure

#endif  // COLLECT_TREASURE_SCENARIO_H_
