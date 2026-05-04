#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "simple_adversary_env.h"

PYBIND11_MODULE(_simple_adversary, m) {
  m.doc() = "Simple Adversary Environment C++ Implementation";

  pybind11::class_<cpp_pettingzoo::simple_adversary::SimpleAdversaryEnv>(
      m, "SimpleAdversaryEnv")
      .def(pybind11::init<int, int, bool, bool>(), pybind11::arg("N") = 2,
           pybind11::arg("max_cycles") = 25,
           pybind11::arg("dynamic_rescaling") = false,
           pybind11::arg("continuous_actions") = false)
      .def("get_agents",
           &cpp_pettingzoo::simple_adversary::SimpleAdversaryEnv::get_agents)
      .def("get_state",
           &cpp_pettingzoo::simple_adversary::SimpleAdversaryEnv::get_state)
      .def("get_render_state", &cpp_pettingzoo::simple_adversary::
                                   SimpleAdversaryEnv::get_render_state)
      .def(
          "reset",
          [](cpp_pettingzoo::simple_adversary::SimpleAdversaryEnv& self,
             std::optional<int> seed) {
            auto obs = self.reset(seed);
            // Return (observations, infos) tuple to match PettingZoo API
            pybind11::dict infos;
            for (const auto& [agent_name, o] : obs) {
              infos[agent_name.c_str()] = pybind11::dict();
            }
            return pybind11::make_tuple(obs, infos);
          },
          pybind11::arg("seed") = pybind11::none())
      .def("step",
           [](cpp_pettingzoo::simple_adversary::SimpleAdversaryEnv& self,
              const cpp_pettingzoo::ActionMap& actions) {
             auto state = self.step(actions);
             // Return (observations, rewards, terminations, truncations, infos)
             pybind11::dict infos;
             for (const auto& [agent_name, o] : state.observations) {
               infos[agent_name.c_str()] = pybind11::dict();
             }
             return pybind11::make_tuple(state.observations, state.rewards,
                                         state.terminations, state.truncations,
                                         infos);
           });
}
