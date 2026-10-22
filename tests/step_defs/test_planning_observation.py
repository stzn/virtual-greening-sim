"""計画観測（ステップ定義は実装時に追加）。"""

from pytest_bdd import scenarios

scenarios("world/planning_observation.feature", "sim/planning_observation_sim.feature")
