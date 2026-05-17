#ifndef SIMPLE_LINE_ENV_H_
#define SIMPLE_LINE_ENV_H_

#include "../../../core/base_env.h"
#include "simple_line_scenario.h"

namespace cpp_pettingzoo::simple_line {

class SimpleLineEnv : public core::BaseEnv {
 public:
  SimpleLineEnv(int N = 4, int max_cycles = 25,
                bool continuous_actions = false,
                bool dynamic_rescaling = false,
                bool terminate_on_success = false);

 private:
  core::World world_;
  SimpleLineScenario scenario_;
};

}  // namespace cpp_pettingzoo::simple_line

#endif  // SIMPLE_LINE_ENV_H_
