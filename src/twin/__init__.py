"""Physics digital twin for power transformers.

The twin couples four physically-grounded simulators:

- :mod:`src.twin.thermal`       – IEEE C57.91 thermal model (top-oil + hot-spot)
- :mod:`src.twin.aging`         – Arrhenius insulation aging integrator
- :mod:`src.twin.gas_evolution` – first-order DGA gas generation kinetics
- :mod:`src.twin.calibration`   – per-asset fit of thermal coefficients to OT

The orchestrator (:class:`src.twin.twin.TransformerTwin`) loads an asset's
historical state, runs a scenario through the simulators, and emits a
:class:`TwinState` snapshot. State can optionally be mirrored to Azure Digital
Twins via :mod:`src.twin.adt_sync`.
"""

from __future__ import annotations

from src.twin.aging import AgingTwin, hours_to_years, normal_insulation_life_hours
from src.twin.calibration import CalibrationResult, calibrate_thermal_twin
from src.twin.gas_evolution import GasEvolutionTwin
from src.twin.scenarios import SCENARIO_LIBRARY, ScenarioInput, build_scenario
from src.twin.thermal import ThermalCoefficients, ThermalTwin
from src.twin.twin import TransformerTwin, TwinState

__all__ = [
    "AgingTwin",
    "CalibrationResult",
    "GasEvolutionTwin",
    "SCENARIO_LIBRARY",
    "ScenarioInput",
    "ThermalCoefficients",
    "ThermalTwin",
    "TransformerTwin",
    "TwinState",
    "build_scenario",
    "calibrate_thermal_twin",
    "hours_to_years",
    "normal_insulation_life_hours",
]
