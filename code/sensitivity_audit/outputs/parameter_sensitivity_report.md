# Tai sensitivity audit

Scope: King County cached XAI audit; model and attribution files are reused.

Baseline rank-correlation between marginal and on-manifold evaluation:
- Deletion: 0.37 [-0.26, 0.89]
- Infidelity: 0.97 [0.83, 1.00]

Parameter sensitivity ranges:
- n_points / deletion: range=0.32, mean CI width=1.32
- n_points / infidelity: range=0.14, mean CI width=0.36
- mask_frac / deletion: range=0.01, mean CI width=1.12
- mask_frac / infidelity: range=0.02, mean CI width=0.21
- bg_size / deletion: range=0.46, mean CI width=1.22
- bg_size / infidelity: range=0.02, mean CI width=0.23

Causal interpretation:
- Marginal replacement asks a weak observational question: what happens if features are broken independently?
- On-manifold donor replacement is closer to a conditional/counterfactual question: what changes when a house is compared with plausible neighboring houses?
- The audit is not a causal identification proof; it checks whether explanations survive more plausible interventions.
