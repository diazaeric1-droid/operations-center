"""Shared data models for AFE Copilot.

Kept dependency-light (stdlib only) so modules like docx_builder can render an
AFE without importing the Anthropic SDK that drafter.py pulls in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AFEDiagnosis:
    """Input to the drafter. Typically produced by Project 1 (Production Engineer Copilot)."""
    well_id: str
    api_number: str
    field: str
    operator: str
    intervention: str                    # Must match a key in cost_db.COST_TEMPLATES
    primary_diagnosis: str               # Free-form, e.g., "Scale + low intake + below POR"
    incremental_rate_bopd: float
    expected_uplift_decline_per_yr: float = 0.6
    requested_by: str = "Senior Production Engineer"

    @classmethod
    def from_json(cls, path: str | Path) -> "AFEDiagnosis":
        with Path(path).open() as f:
            return cls(**json.load(f))

    # ---- validation + Production-Engineer-Copilot chain ----------------------

    # The 8 interventions the cost DB / drafter / risk register know how to price.
    # Kept here as a literal so models.py stays import-light (no cost_db import at
    # module load), but assert-checked against cost_db in the test suite.
    VALID_INTERVENTIONS = (
        "acid_stimulation",
        "scale_treatment",
        "esp_swap",
        "esp_to_beam_conversion",
        "rod_pump_workover",
        "gas_lift_optimization",
        "paraffin_treatment",
        "p_and_a",
    )

    _REQUIRED_STR_FIELDS = ("well_id", "api_number", "field", "operator",
                            "intervention", "primary_diagnosis")

    def validate(self) -> list[str]:
        """Return a list of human-readable problems. Empty list means valid.

        Stdlib only — no pydantic. This is the same contract the Production
        Engineer Copilot (Project 1) needs to satisfy before its diagnosis can
        be turned into an AFE.
        """
        problems: list[str] = []

        if self.intervention not in self.VALID_INTERVENTIONS:
            problems.append(
                f"intervention '{self.intervention}' is not one of the supported "
                f"types: {', '.join(self.VALID_INTERVENTIONS)}."
            )

        # required non-empty string fields
        for field_name in self._REQUIRED_STR_FIELDS:
            value = getattr(self, field_name, None)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"'{field_name}' must be a non-empty string.")

        # incremental rate must be a positive number
        try:
            rate = float(self.incremental_rate_bopd)
            if not rate > 0:
                problems.append("'incremental_rate_bopd' must be greater than 0.")
        except (TypeError, ValueError):
            problems.append("'incremental_rate_bopd' must be a number greater than 0.")

        # decline must be in the open interval (0, 2)
        try:
            decline = float(self.expected_uplift_decline_per_yr)
            if not (0 < decline < 2):
                problems.append(
                    "'expected_uplift_decline_per_yr' must be between 0 and 2 (exclusive)."
                )
        except (TypeError, ValueError):
            problems.append("'expected_uplift_decline_per_yr' must be a number in (0, 2).")

        return problems

    @classmethod
    def from_pe_copilot(cls, d: dict) -> "AFEDiagnosis":
        """Build an AFEDiagnosis from a Production Engineer Copilot export dict.

        Accepts the PE-Copilot export schema (same field names as this dataclass).
        Unknown keys are ignored so the upstream app can add fields without
        breaking the chain. Raises ValueError listing ALL validation problems if
        the resulting diagnosis is invalid.
        """
        if not isinstance(d, dict):
            raise ValueError("Production Engineer Copilot export must be a JSON object (dict).")

        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}

        # surface missing required keys clearly before dataclass construction
        missing = [k for k in cls._REQUIRED_STR_FIELDS if k not in filtered] + \
                  (["incremental_rate_bopd"] if "incremental_rate_bopd" not in filtered else [])
        if missing:
            raise ValueError(
                "Production Engineer Copilot export is missing required field(s): "
                + ", ".join(missing)
            )

        diagnosis = cls(**filtered)
        problems = diagnosis.validate()
        if problems:
            raise ValueError(
                "Production Engineer Copilot diagnosis failed validation:\n"
                + "\n".join(f"  • {p}" for p in problems)
            )
        return diagnosis
