#ifndef SIMPLE_ADVERSARY_ENV_H_
#define SIMPLE_ADVERSARY_ENV_H_

#include "../../../core/base_env.h"
#include "simple_adversary_scenario.h"

namespace cpp_mpe2::simple_adversary {

class SimpleAdversaryEnv : public core::BaseEnv {
 public:
  SimpleAdversaryEnv(int N = 2, int max_cycles = 25,
                     bool dynamic_rescaling = false,
                     bool continuous_actions = false);

 private:
  core::World world_;
  SimpleAdversaryScenario scenario_;
};

}  // namespace cpp_mpe2::simple_adversary

#endif  // SIMPLE_ADVERSARY_ENV_H_
