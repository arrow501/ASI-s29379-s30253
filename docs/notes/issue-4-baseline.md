# Issue #4: Sprint 1 baseline working notes

Issue: <https://github.com/arrow501/ASI-s29379-s30253/issues/4>

## Scope

Declare model features in `project.yaml`, implement the preprocessing and logistic
regression baseline, complete `notebooks/02_baseline.ipynb`, and generate full-data
`metrics.json`. Add a pre-commit check that validates project configuration against
the dataset contract when either YAML file changes. Keep the instructor-owned
`src/asi_project/datasets.yaml` unchanged.

## Proposed decisions

- Numeric features: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`.
- Categorical features: `job`, `marital`, `education`, `default`, `housing`, `loan`,
  `contact`, `month`, `poutcome`.
- Exclude `duration` as required by the prediction-time contract.
- Use the contract's chronological split, with the final 20% held out.
- Fit median imputation and scaling for numeric features, and most-frequent
  imputation and one-hot encoding for categorical features, inside the pipeline.
  Ignore unseen categories at prediction time.
- Start with untuned logistic regression and compare it with the most-frequent
  dummy classifier using the same split. F1 is the main metric.

Pending EDA, treat `day` as numeric, preserve `unknown` as a category, and retain
`pdays=-1` as a sentinel. Revisit these assumptions and regenerate metrics after
issue #3 is integrated. Existing data preparation functions allow baseline work
to proceed without temporary replacement code.

## Workflow and evidence

Review and refine the implementation plan before coding. After implementation,
review the diff and results, revise, then perform a final pass. Validate with
focused configuration and pipeline tests, the existing suite, pre-commit,
full-data notebook execution, the CI sample-data path, and strict automated
Sprint 1 checks. Record actual results and remaining questions here as work lands.

Implementation and validation are not yet complete. The configuration hook runs
at commit time, not on every file save.

## Later documentation

Move settled pipeline details into `docs/architecture.md` and measured results
and limitations into `docs/model_card.md`. Use `docs/adr/` for substantial
decisions that need a lasting rationale. Keep this note concise and distinguish
proposals from verified results.
