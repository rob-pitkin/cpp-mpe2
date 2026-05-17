#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "collect_treasure_env.h"

PYBIND11_MODULE(_collect_treasure, m) {
  m.doc() = "Collect Treasure Environment C++ Implementation";

  pybind11::class_<cpp_pettingzoo::collect_treasure::CollectTreasureEnv>(
      m, "CollectTreasureEnv")
      .def(pybind11::init<int, int, int, int, bool, bool>(),
           pybind11::arg("num_collectors") = 6,
           pybind11::arg("num_deposits") = 2,
           pybind11::arg("num_treasures") = 6,
           pybind11::arg("max_cycles") = 25,
           pybind11::arg("continuous_actions") = false,
           pybind11::arg("dynamic_rescaling") = false)
      .def("get_agents",
           &cpp_pettingzoo::collect_treasure::CollectTreasureEnv::get_agents)
      .def("get_state",
           &cpp_pettingzoo::collect_treasure::CollectTreasureEnv::get_state)
      .def("get_render_state",
           &cpp_pettingzoo::collect_treasure::CollectTreasureEnv::get_render_state)
      .def(
          "reset",
          [](cpp_pettingzoo::collect_treasure::CollectTreasureEnv& self,
             std::optional<int> seed) {
            auto obs = self.reset(seed);
            pybind11::dict infos;
            for (const auto& [name, o] : obs) {
              infos[name.c_str()] = pybind11::dict();
            }
            return pybind11::make_tuple(obs, infos);
          },
          pybind11::arg("seed") = pybind11::none())
      .def("step",
           [](cpp_pettingzoo::collect_treasure::CollectTreasureEnv& self,
              const cpp_pettingzoo::ActionMap& actions) {
             auto state = self.step(actions);
             pybind11::dict infos;
             for (const auto& [name, o] : state.observations) {
               infos[name.c_str()] = pybind11::dict();
             }
             return pybind11::make_tuple(state.observations, state.rewards,
                                         state.terminations, state.truncations,
                                         infos);
           });
}
