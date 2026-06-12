"""AFE Drafter agent: takes a well-diagnosis JSON, produces a draft AFE as .docx + markdown.

The agent uses Claude to write technical justification + scope narrative, and
calls deterministic tools for cost lookup, economics, and risk-register pull —
so engineering numbers stay trusted while the language stays human.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .cost_db import COST_TEMPLATES, lookup_cost_template, total_estimate
from .economics import compute_economics
from .models import AFEDiagnosis  # dataclass lives in models.py (no anthropic dependency)
from .risk_register import lookup_risks


class MissingAPIKey(RuntimeError):
    """Raised when drafting is attempted without ANTHROPIC_API_KEY set. The cost,
    economics, and risk tools (and Monte-Carlo / .docx) all work without a key —
    only the LLM narrative needs one."""


# ---------- tool schemas (Anthropic tool-use API) ----------------------------

TOOL_SCHEMAS = [
    {
        "name": "lookup_cost_template",
        "description": "Return the canonical line-item cost template for an intervention type. "
                       "Use this once at the start to seed cost line items.",
        "input_schema": {
            "type": "object",
            "properties": {"intervention": {"type": "string", "enum": list(COST_TEMPLATES)}},
            "required": ["intervention"],
        },
    },
    {
        "name": "compute_economics",
        "description": "Compute NPV @ 10%, payout months, and $/incremental-bbl for the AFE.",
        "input_schema": {
            "type": "object",
            "properties": {
                "treatment_cost_usd": {"type": "number"},
                "incremental_rate_bopd": {"type": "number"},
                "uplift_decline_per_yr": {"type": "number", "default": 0.6},
                "realized_price_per_bbl": {"type": "number", "default": 65.0},
            },
            "required": ["treatment_cost_usd", "incremental_rate_bopd"],
        },
    },
    {
        "name": "lookup_risks",
        "description": "Return the standard risk register entries for an intervention type.",
        "input_schema": {
            "type": "object",
            "properties": {"intervention": {"type": "string", "enum": list(COST_TEMPLATES)}},
            "required": ["intervention"],
        },
    },
]


class ToolExecutor:
    def __init__(self, diagnosis: AFEDiagnosis):
        self.diagnosis = diagnosis
        self.last_cost_template = None

    def dispatch(self, name: str, args: dict) -> str:
        try:
            result = getattr(self, f"_tool_{name}")(**args)
            return json.dumps(result, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else float(o), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e), "tool": name})

    def _tool_lookup_cost_template(self, intervention: str) -> list[dict]:
        items = lookup_cost_template(intervention)
        self.last_cost_template = items
        return [{**asdict(i), "total_usd": i.total_usd} for i in items]

    def _tool_compute_economics(self, **kwargs) -> dict:
        # P&A / pure-cost jobs have no production uplift — production economics
        # (NPV, payout, $/bbl) are meaningless and would render as $inf/bbl. Return
        # a structured note so the model frames it as abandonment cost vs. liability.
        rate = kwargs.get("incremental_rate_bopd", 0) or 0
        if rate <= 0:
            return {
                "applicable": False,
                "note": ("No production uplift — this is a cost-only / P&A job. Do NOT "
                         "report NPV, payout, or $/bbl. Justify against remaining "
                         "liability, plugging-bond release, and avoided idle-well carrying "
                         "cost / regulatory exposure instead."),
                "treatment_cost_usd": kwargs.get("treatment_cost_usd"),
            }
        econ = compute_economics(**kwargs)
        return {"applicable": True, **asdict(econ)}

    def _tool_lookup_risks(self, intervention: str) -> list[dict]:
        return [asdict(r) for r in lookup_risks(intervention)]


# ---------- agent prompt -----------------------------------------------------

SYSTEM_PROMPT = """You are an AFE Drafter for a Permian Basin E&P operator. Given a well's diagnosis and chosen intervention, produce a complete draft AFE (Authorization for Expenditure) ready for engineering-manager review.

Today's date: {today}

Process:
1. Call `lookup_cost_template` for the intervention to get baseline line-item costs.
2. Call `compute_economics` using the line-item total as treatment_cost and the diagnosis's expected uplift. If the tool returns `"applicable": false` (a P&A or cost-only job), do NOT report NPV/payout/$ per bbl — justify the spend against remaining liability, plugging-bond release, and avoided idle-well carrying cost instead, and label the Economics section "Cost & Liability Basis".
3. Call `lookup_risks` for the intervention.
4. Write a complete AFE markdown document with these sections, in this order:

   - **AFE Header** (AFE number TBD, well ID, API, field, operator, requested-by, date)
   - **Executive Summary** (2-3 sentence rationale + total cost + NPV/payout headline)
   - **Scope of Work** (numbered steps the rig and service crews will execute)
   - **Technical Justification** (why this intervention now — references the primary diagnosis, expected outcome, alternatives considered)
   - **Cost Breakdown** (table: category | description | qty | unit | unit cost | total | vendor)
   - **Economics** (table: total cost | first-year incremental | EUR uplift | NPV @ 10% | payout months | $/incremental bbl)
   - **Risk Register** (table: category | risk | likelihood | consequence | mitigation)
   - **Schedule & Approvals** (planned execution window, approval signature block: Engineer → Engineering Manager → Operations Manager → Finance → JV Partners if applicable)

Style:
- Write the way a Staff Production Engineer would write to an engineering manager — terse, specific, decision-ready.
- Cost numbers come ONLY from `lookup_cost_template` results. If you need to deviate (e.g., longer rig days for a complex well), state the deviation in the scope and add a note in the cost breakdown.
- Risk language comes from `lookup_risks`. You may add 1-2 well-specific risks if the diagnosis warrants it.
- Never invent vendor names. If the template doesn't have one, write "TBD" or "[Procurement to select]".
- **Never invent specific well IDs, pad histories, or analogous wells.** The only well referenced should be the one in the input. If you want to argue from analogous-well experience, frame it generically: "Industry experience on similar Wolfcamp wells suggests..." — not "Wells ED-003H and ED-007H on this pad failed..."
- **Output ONLY the AFE markdown.** No preamble like "Drafting now..." or "All data in hand." The first character of your response must be the `#` of the AFE header.
"""


def run_drafter(diagnosis: AFEDiagnosis, model: str = "claude-sonnet-4-6", verbose: bool = False,
                api_key: str | None = None) -> str:
    load_dotenv()
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise MissingAPIKey(
            "ANTHROPIC_API_KEY is not set — provide a key to draft the AFE narrative. "
            "Cost tables, Monte-Carlo economics, and .docx export work without a key.")
    from anthropic import Anthropic
    client = Anthropic(api_key=key)
    console = Console()
    executor = ToolExecutor(diagnosis)

    if verbose:
        console.print(f"[bold cyan]Drafting AFE:[/] {diagnosis.well_id} | {diagnosis.intervention}")

    user_prompt = (
        f"Draft an AFE for the following:\n\n"
        f"Well: {diagnosis.well_id} ({diagnosis.api_number})\n"
        f"Field: {diagnosis.field}\n"
        f"Operator: {diagnosis.operator}\n"
        f"Intervention: {diagnosis.intervention}\n"
        f"Primary diagnosis: {diagnosis.primary_diagnosis}\n"
        f"Expected uplift: +{diagnosis.incremental_rate_bopd} BOPD, decline {diagnosis.expected_uplift_decline_per_yr}/yr\n"
        f"Requested by: {diagnosis.requested_by}"
    )
    messages = [{"role": "user", "content": user_prompt}]
    system_prompt = SYSTEM_PROMPT.format(today=date.today().isoformat())

    for _ in range(10):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = "".join(b.text for b in response.content if b.type == "text")
            # Belt-and-suspenders: strip anything before the first markdown header
            # so any residual "drafting now..." preamble doesn't leak into the file.
            first_header = text.find("\n#")
            if first_header > 0 and not text.lstrip().startswith("#"):
                text = text[first_header:].lstrip()
            return text

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    console.print(f"[dim]→ tool: {block.name}({block.input})[/]")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": executor.dispatch(block.name, block.input),
                })
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

    return "Drafter stopped without producing AFE."


def main():
    parser = argparse.ArgumentParser(description="Draft an AFE from a well-diagnosis JSON.")
    parser.add_argument("--input", required=True, help="Path to diagnosis JSON")
    parser.add_argument("--out", default="drafts", help="Output directory")
    parser.add_argument("--model", default=os.environ.get("MODEL", "claude-sonnet-4-6"))
    parser.add_argument("--docx", action="store_true", help="Also generate .docx output")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    diagnosis = AFEDiagnosis.from_json(args.input)
    markdown = run_drafter(diagnosis, model=args.model, verbose=args.verbose)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"AFE_{diagnosis.well_id}_{diagnosis.intervention}.md"
    md_path.write_text(markdown)
    print(f"\n✓ Wrote {md_path}")

    if args.docx:
        from .docx_builder import build_docx
        docx_path = out_dir / f"AFE_{diagnosis.well_id}_{diagnosis.intervention}.docx"
        build_docx(markdown, docx_path, diagnosis)
        print(f"✓ Wrote {docx_path}")


if __name__ == "__main__":
    main()
