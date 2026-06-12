"""Standard risk register per intervention type — the boilerplate every AFE
needs but engineers re-type each time. Encoded once here."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Risk:
    category: str         # HSE | Operational | Cost | Schedule | Regulatory
    description: str
    likelihood: str       # Low | Medium | High
    consequence: str      # Low | Medium | High
    mitigation: str


RISK_TEMPLATES: dict[str, list[Risk]] = {
    "acid_stimulation": [
        Risk("HSE", "HCl exposure to personnel during mixing/pumping", "Medium", "High",
             "JSA + PPE per acid handling SOP; emergency eyewash on location"),
        Risk("Operational", "Acid breakthrough into adjacent zone", "Low", "Medium",
             "Use of mechanical diverter + ball sealers; pre-job zonal isolation review"),
        Risk("Operational", "Tubing/casing corrosion from spent acid", "Low", "Medium",
             "Corrosion inhibitor in acid blend; post-treatment flush with KCl water"),
        Risk("Schedule", "Pump truck availability in tight market", "Medium", "Medium",
             "Confirm CTU + acid vendor 14 days ahead; backup vendor on standby"),
        Risk("Cost", "Acid volume over-run on tight perfs", "Medium", "Low",
             "Stage acid in 1,000 gal increments; monitor injection pressure for ISIP"),
    ],
    "scale_treatment": [
        Risk("Operational", "Inhibitor squeeze flowback before reaching pay zone", "Medium", "Medium",
             "Shut-in for 24 hr post-squeeze for adsorption; monitor inhibitor residual"),
        Risk("HSE", "Phosphonate chemical handling", "Low", "Low",
             "Standard chem handling PPE; SDS on location"),
        Risk("Cost", "Repeat squeeze needed if residual depletes <6 months", "Medium", "Medium",
             "Establish residual monitoring program; trigger re-squeeze at <5 ppm"),
    ],
    "esp_swap": [
        Risk("Operational", "Fish in hole during ESP pull", "Low", "High",
             "Pre-job tubing tally + EMI; experienced WOR crew; fishing tools on call"),
        Risk("HSE", "H2S exposure during tubing trip", "Low", "High",
             "H2S monitors + SCBA on location; PSI training current for all crew"),
        Risk("Schedule", "New ESP lead-time in tight market", "Medium", "High",
             "Confirm vendor stock 30 days out; consider rebuild from inventory if available"),
        Risk("Operational", "Wrong-sized ESP installed (POR mismatch)", "Medium", "High",
             "Post-acid inflow test required before ESP selection; senior PE sign-off on size"),
        Risk("Cost", "VSD/transformer surface work scope creep", "Medium", "Medium",
             "Pre-job site visit + photo survey; surface SOW signed off before mobilization"),
    ],
    "esp_to_beam_conversion": [
        Risk("Operational", "Beam unit foundation inadequate for soil", "Medium", "Medium",
             "Soil bearing test pre-pour; foundation design stamped by P.E."),
        Risk("Operational", "Rod string sizing for current rate window", "Low", "High",
             "Run RodStar or similar simulator; senior PE sign-off on string design"),
        Risk("Schedule", "Two trade contractors (electrical + concrete) coordination", "Medium", "Medium",
             "Single GC to manage trades; daily standup during conversion"),
        Risk("HSE", "Crane/rigging lift of beam unit", "Low", "High",
             "Certified rigger + crane operator; lift plan reviewed by Ops Manager"),
    ],
    "rod_pump_workover": [
        Risk("Operational", "Cause of failure unclear post-pull", "Low", "Medium",
             "Photograph + measure all pulled rods; tubing inspection for wear pattern"),
        Risk("Cost", "Additional rod sections needed beyond initial scope", "Medium", "Low",
             "Mobilize with 20% spare rod inventory; rod inspection determines final scope"),
    ],
    "gas_lift_optimization": [
        Risk("Operational", "Valve fails to seat after wireline change", "Low", "Medium",
             "Pressure test mandrel post-change before normalizing"),
        Risk("Schedule", "Slickline truck availability", "Low", "Low",
             "Routine slickline scope, multiple vendors available"),
    ],
    "paraffin_treatment": [
        Risk("HSE", "Hot oil burn hazard (180°F+)", "Low", "Medium",
             "PPE per hot oil SOP; barricades around treatment area"),
        Risk("Operational", "Plunger sticking recurrence", "Medium", "Low",
             "Increase continuous chemical injection rate post-treatment"),
    ],
    "p_and_a": [
        Risk("Regulatory", "Cement plug placement fails RRC inspection", "Low", "High",
             "Pre-job plug design per RRC Rule 14; tagged + tested per regulation"),
        Risk("Operational", "Unable to set cement plug at planned depth", "Medium", "Medium",
             "Junk basket runs pre-cement; contingency for additional cement volume"),
        Risk("Cost", "Surface remediation scope unclear", "Medium", "Medium",
             "Pre-P&A site walkdown with Ops + Land; surface scope locked before mobilization"),
        Risk("Regulatory", "Bond release delayed by paperwork", "Medium", "Low",
             "Pre-file W-3 90 days before P&A; track RRC submission to release"),
    ],
}


def lookup_risks(intervention: str) -> list[Risk]:
    return RISK_TEMPLATES.get(intervention, [])
