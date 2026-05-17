# cpp-mpe2

Fast C++ implementations of [Farama mpe2](https://mpe2.farama.org/) (Multi-Agent Particle Environments) with a [PettingZoo](https://pettingzoo.farama.org/)-compatible Python API. Drop-in replacement for `mpe2` with **6–58× faster** rollouts.

## What's here

Every environment in `mpe2` has a C++ implementation built on a shared `core::BaseEnv` + `core::World` + `core::Scenario` abstraction, exposed to Python via [pybind11](https://github.com/pybind/pybind11) and wrapped to match `PettingZoo`'s `ParallelEnv` and AEC APIs.

| Environment | Description | Step speedup vs mpe2 |
|---|---|---|
| **Simple** | 1 agent, 1 landmark — the basic single-agent task | 15.13× |
| **SimpleSpread** | 3 cooperative agents covering 3 landmarks | 20.89× |
| **SimpleReference** | 2 agents with 10-word communication to identify goals | 15.20× |
| **SimpleSpeakerListener** | Asymmetric speaker/listener task | 16.01× |
| **SimpleAdversary** | 1 adversary + N good agents racing to a goal | 14.93× |
| **SimpleTag** | 3 adversaries chase 1 good agent (predator-prey) | 17.93× |
| **SimplePush** | 1 adversary tries to push 1 good agent off-goal | 13.81× |
| **SimpleFormation** | 4 agents form a moving formation (Hungarian matching) | 21.60× |
| **SimpleLine** | 4 agents arrange themselves along a line | 19.85× |
| **CollectTreasure** | 6 collectors + 2 deposits + 6 respawning treasures | 39.72× |
| **SimpleWorldComm** | Leader adversary + comms + forest visibility | **58.64×** |
| **SimpleCrypto** | Alice/Bob/Eve — encrypted message passing | 14.29× |

**Average step speedup: 20.0× | Range: 6.2× – 58.6×**

Full results (reset/step/episode breakdown, hardware, methodology) in [`BENCHMARKS.md`](BENCHMARKS.md).

## Installation

Requires Python 3.10+, CMake 3.15+, a C++17 compiler, and [`uv`](https://docs.astral.sh/uv/) for Python dependency management.

```bash
git clone https://github.com/rob-pitkin/cpp-mpe2.git
cd cpp-mpe2
uv sync
uv run cmake -S . -B build
uv run cmake --build build -j8
```

This produces `_simple_core.so`, `_simple_spread.so`, etc. inside `build/`, which the Python wrappers locate at import time.

## Usage

The Python API is intentionally a drop-in replacement for `mpe2`. Swap your import and everything else stays the same.

```python
# Before (mpe2)
from mpe2 import simple_spread_v3
env = simple_spread_v3.parallel_env()

# After (cpp-mpe2)
from cpp_mpe2.simple_spread.simple_spread import parallel_env
env = parallel_env()

obs, _ = env.reset(seed=0)
for _ in range(25):
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
```

Both `ParallelEnv` and AEC APIs are supported. `env(...)` returns the AEC-wrapped form, `parallel_env(...)` returns the parallel form.

## Project structure

```
cpp-mpe2/
├── core/                       # Shared C++ runtime
│   ├── entity.h                # Agent, Landmark base classes
│   ├── world.{h,cpp}           # Physics integration, collisions
│   ├── scenario.h              # Scenario interface (env-specific logic)
│   ├── base_env.{h,cpp}        # Step/reset loop, action decomposition
│   └── types.h                 # ObservationMap, ActionMap, State, etc.
├── cpp_mpe2/                   # Python package
│   ├── core/                   # Python-side Agent/Landmark/World shims for rendering
│   ├── simple/                 # One subdir per environment
│   │   ├── simple.py           # PettingZoo wrapper (ParallelEnv + AEC + render)
│   │   └── cpp/                # C++ scenario + env + pybind11 bindings
│   │       ├── simple_scenario.{h,cpp}
│   │       ├── simple_env.{h,cpp}
│   │       └── bindings.cpp
│   ├── simple_spread/  ...     # (same pattern for all 12 envs)
│   └── benchmark_*.py          # One benchmark script per env
├── tests/                      # 411 tests covering API, AEC, equivalence, render
├── CMakeLists.txt              # Builds one pybind11 module per env
├── pyproject.toml
└── BENCHMARKS.md               # Detailed perf analysis
```

## Architecture notes

A few design choices worth knowing if you're extending this:

- **One `core::BaseEnv` for all envs.** It handles the step/reset loop, max-cycles truncation, observation/reward collection, and discrete-action decomposition (`action % 5` → movement, `action / 5` → comm one-hot). Adding a new env means writing a `Scenario` subclass; the loop is shared.
- **`core::Scenario` interface** has `make_world`, `reset_world`, `reward`, `observation`, plus optional hooks: `global_reward` (for local/global blending), `is_terminal`, and `post_step` (for envs like CollectTreasure that mutate world state between physics and reward computation).
- **Landmark color is a render-only concept.** It's set Python-side at `__init__`, never round-tripped through C++. Forgetting this causes `None * 200` render crashes — every env must explicitly assign landmark colors in its Python wrapper.
- **No new C++ for asymmetric action spaces.** The `Discrete(20)` leader action in `simple_world_comm` and the `Discrete(4)` pure-comm action in `simple_crypto` both worked out of the box because `BaseEnv::step` already handled the general `movement % mdim` / `action / mdim` decomposition. Action space asymmetry lives entirely in the Python wrapper's `_action_spaces` dict.
- **`Landmark::subtype`** is a generic int tag used by `simple_world_comm` (obstacle/food/forest) and could be reused by other envs that pack multiple landmark categories into one vector.

## Running tests

```bash
uv run pytest tests/                     # All 411 tests
uv run pytest tests/test_simple_spread/  # One env's tests
uv run pytest -k equivalence             # Just the mpe2-parity checks
```

Each environment has three test files:
- `test_api.py` / `test_equivalence.py` — `parallel_api_test` + shape/action-space parity with `mpe2`
- `test_aec.py` — AEC API compliance via `pettingzoo.test.api_test`
- `test_render.py` — RGB array + human-mode rendering

## Running benchmarks

```bash
uv run python cpp_mpe2/benchmark_simple_spread.py
```

Each benchmark runs 1M resets, 1M steps, and 100K episodes against both backends and reports the per-op speedup. See [`BENCHMARKS.md`](BENCHMARKS.md) for the full table and per-environment analysis.

## Dependencies

- **Build:** CMake 3.15+, C++17 compiler, pybind11 (auto-installed via uv), [`munkres-cpp`](https://github.com/saebyn/munkres-cpp) (vendored under `core/third_party/` for Hungarian matching in `SimpleFormation` / `SimpleLine`)
- **Runtime:** Python 3.10+, `pettingzoo>=1.24`, `gymnasium>=0.29`, `numpy>=1.24`, `pygame-ce` (for rendering)
- **Dev:** `pytest`, `mpe2==1.1.0` (used as the equivalence-test oracle)

All managed via `uv sync` from `pyproject.toml`.
