# Walmart-Promotion-Optimization-Project
A data-driven promotion strategy pipeline under budget constraints

# Project Description

This project builds an end-to-end promotion decision system for retail sales optimization, using historical Walmart sales data to answer a practical business question:

Given limited promotion budget, which Store × Department should receive promotion, and how much promotion intensity should be allocated to each?

Unlike simple uplift modeling or isolated response analysis, this project focuses on decision-oriented modeling, connecting prediction, robustness validation, budget allocation, and execution into a single, coherent pipeline.

The system is designed to be:

- Interpretable (promotion response curves instead of black-box scores)

- Robust (validated across time slices and sales states)

- Actionable (outputs store-level execution sheets ready for operations)

# Core Ideas

- Treat promotion intensity as a controllable decision variable (knob) rather than a fixed feature.

- Learn promotion response curves instead of single-point effects.

- Explicitly model diminishing returns and turning points.

- Allocate budget using marginal ROI–based greedy optimization, not “all-or-nothing” decisions.

- Separate decision logic (step16) from execution translation (step17).

# Pipeline Overview
## Data Preparation & Profiling (Step 1–8)

- Clean and merge Walmart sales, calendar, and markdown data.

- Construct lag features and seasonality signals.

- Segment departments into tiers (e.g. core / holiday / tail).

- Profile stores by sales structure and promotion behavior.

- Identify promotion-eligible departments.

## Forecast & Baseline Modeling (Step 9–12)

- Build supervised forecasting features.

- Train a LightGBM baseline model.

- Validate against naive baselines.

- Analyze feature importance to ensure model sanity.

## Promotion Response Curves (Step 13–14)

- Simulate what-if promotion intensity levels.

- Estimate response curves by:

- overall

- department tier

- store profile

- tier × store profile

Extract decision-ready summaries:

  - best promotion intensity

  - marginal turning point

  - expected lift vs zero promotion

## Robustness Validation (Step 15A / 15B)

- Verify strategy stability across:

  - time slices (holiday vs non-holiday)

  - sales states (low / mid / high sales)

- Ensure recommended intensities remain within safe regions under different conditions.

## Budget-Constrained Allocation (Step 16)

- Construct a candidate universe of Store × Department.

- Enforce turning-point safety constraints.

- Allocate promotion budget using incremental marginal ROI greedy allocation:

  - Start from zero promotion.

  - Increase promotion step-by-step.

  - Always fund the highest marginal ROI action.

  - Stop when budget is exhausted.

##  Execution Sheet Generation (Step 17)

- Translate strategy output into an operations-friendly execution table.

- Add:

  - allocation flags

  - risk indicators

  - human-readable action notes

- Ready for direct handoff to business or ops teams.

# Final Outputs

- Promotion allocation plan under budget constraints

- Store × Department execution sheet with:

  -  promotion intensity

  - expected lift

  - risk level

  - operational guidance

# Why This Project Matters

Most promotion models stop at prediction.
This project goes one step further — it makes decisions.

It demonstrates how to:

- bridge ML outputs with business constraints,

- avoid over-promotion and tail risk,

- design promotion systems that are deployable in real retail operations.

# Tech Stack

- Python

- pandas / numpy

- LightGBM

- Structured, step-based pipeline design

- Fully reproducible via scripts
