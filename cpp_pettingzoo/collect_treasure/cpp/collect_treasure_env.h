#ifndef COLLECT_TREASURE_ENV_H_
#define COLLECT_TREASURE_ENV_H_

#include "../../../core/base_env.h"
#include "collect_treasure_scenario.h"

namespace cpp_pettingzoo::collect_treasure {

class CollectTreasureEnv : public core::BaseEnv {
 public:
  CollectTreasureEnv(int num_collectors = 6, int num_deposits = 2,
                     int num_treasures = 6, int max_cycles = 25,
                     bool continuous_actions = false,
                     bool dynamic_rescaling = false);

 private:
  core::World world_;
  CollectTreasureScenario scenario_;
};

}  // namespace cpp_pettingzoo::collect_treasure

#endif  // COLLECT_TREASURE_ENV_H_
