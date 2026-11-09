# S2-1 — Kedro foundation and pipeline interfaces

Owner: Arrow. Reviewer: Anton. Implements issue #10.

Kedro 1.7.0 is pinned in `requirements/preflight-s2.txt`. Install it in `asi-ml`
with `python -m pip install -r requirements/preflight-s2.txt -c constraints.txt`.
The project uses the existing `src/asi_project` package and `conf` directory.

## Shared configuration

The custom context in `context.py` exposes `params:project` to nodes. It reads
feature lists, dataset selection and split values from the selected project's
`project.yaml`. Target, positive class and split strategy come from the unchanged
`datasets.yaml` registry. `conf/base/parameters.yml` contains an empty `project`
mapping: defaults are supplied by the context, not duplicated in YAML. A bare
config loader returns only overrides; use `session.load_context().params` for
effective settings. Nodes receive the resolved parameter dictionary and should
not read configuration files themselves.
Catalog entries own file paths. `base` is shared, `local` is the default override
environment, and `ci` overrides the catalog to select the committed sample.

## Pipeline boundary

| Pipeline | Inputs | Outputs | Owner |
| --- | --- | --- | --- |
| `data_processing` | `raw_data`, `params:project` | `prepared_data` | Anton, S2-2 |
| `data_science` | `prepared_data`, `params:project` | `trained_model`, `metrics` | Arrow, S2-3 |

`raw_data` is a DataFrame loaded by the catalog with the appropriate CSV separator.
`prepared_data` is a validated, normalized and training-filtered DataFrame that
retains target `y` and row order. It contains no fitted preprocessing.
The training pipeline selects configured features, performs the chronological
split, and fits preprocessing only inside the sklearn model on training rows.
Intermediate train/test datasets can stay in memory.

S2-2 owns input catalog entries for full data and the CI sample. S2-3 owns output
entries for the persisted sklearn pipeline and JSON metrics under ignored `data/`
directories. Full-data metrics must not overwrite Sprint 1's tracked `metrics.json`.
Evaluation includes the dummy comparison and accuracy, precision, recall, F1,
and confusion matrix, using the existing metric helpers.

## Foundation verification

Run `kedro info`, `kedro registry list`, and `pytest` in `asi-ml`.
Both named pipelines and their `__default__` composition are registered. Their
bodies are deliberately empty until S2-2/S2-3 add nodes: `kedro run` cannot train
yet. S2-1 verifies project discovery and parameter resolution in local and CI;
the complete run is verified after both implementations are integrated.

Local verification on 2026-11-09: Kedro reports 1.7.0 and lists all three
registrations; the complete pytest suite passes (125 tests). Ruff lint and format
checks pass, and `pre-commit run --all-files` passes. Kedro emits upstream OmegaConf
deprecation warnings; configuration loading succeeds in both environments.

CI installs the pinned Sprint 2 requirements with the shared constraints before
checking library versions and running tests. It also runs `kedro info` and
`kedro registry list`. Full pipeline execution remains the S2-4 integration step.
Tests cover loading from a repository subdirectory, both environments, propagation
of both feature lists and split settings, and registry composition with connected
nonempty pipeline factories. Catalog contents are left to S2-2/S2-3.
These are local validation results; GitHub Actions has not run on these changes.

## Proposed PR description

Configure the existing package as a Kedro 1.7.0 project and register the two
Sprint 2 pipeline factories plus their default composition. Resolve shared node
parameters from `project.yaml` and the dataset registry, validate the effective
settings including environment and runtime overrides, and document catalog
ownership and the `prepared_data` boundary for S2-2 and S2-3.
Install the pinned Kedro requirements in CI and check project discovery there.

Validation: project discovery, registry listing, configuration tests for local/CI
and changes to project settings, and the existing pytest suite. Pipeline bodies
remain placeholders pending the subsequent implementation issues.

AI assistance: Codex assisted implementation, documentation, and local validation.

## Configuration validation

The custom Kedro context validates resolved `params:project` before parameters
can enter the catalog for node execution. The configuration CLI check shares this
validator and the existing feature-contract rules. Feature lists must contain
allowed names without duplicates or overlap. Test size must be numeric and
strictly between zero and one; the random seed must be an integer in the supported
range. Dataset ID, target, positive class, and split strategy must match the
selected contract, even when environment files or runtime parameters override
values. Valid feature subsets and split fractions remain configurable.

Parameter configuration uses deep (`soft`) merging: environment overrides retain
unspecified base fields, and runtime overrides take precedence over environment
values. Regression tests cover partial overrides in local and CI environments,
verify untouched settings survive, and check that `project.yaml` is unchanged.
Invalid environment override tests also supply only the changed fields.

This validates configuration only; dataset contents, row ordering, splitting,
fitted preprocessing and persistence remain the subsequent pipeline work.

## Project paths and cross-platform behavior

The selected Kedro session's `project_path` is the anchor for `project.yaml`.
Configuration uses `pathlib` rather than OS-specific path separators or shell
commands. Explicit roots take precedence over `ASI_PROJECT_ROOT`; scripts that
do not supply a root still use that variable, then search upward from CWD.
Changing directories cannot redirect a selected session to another checkout.
No global custom OmegaConf resolver is registered, so contexts for two projects
can coexist without sharing project defaults.

The context initialization sequence is:

1. Read team settings using the selected project path and obtain its dataset contract.
2. Construct default `project` parameters in memory.
3. Read Kedro's merged base, environment and runtime parameters.
4. Deep-merge those overrides onto the defaults and validate the result.
5. Supply the complete parameters to Kedro for node-parameter validation and the catalog.

Precedence, from lowest to highest, is project defaults → base parameters →
selected environment → runtime parameters. Lists are replaced, not appended.
Other parameter groups pass through unchanged. Contract fields cannot be changed
by any of these overrides.

For example, `conf/ci/parameters.yml` can override only the seed:

```yaml
project:
  random_state: 17
```

An eventual CLI run can override that seed again with
`kedro run --env ci --params project.random_state=23`.
This does not edit `project.yaml` or alter the default 80/20 split. A conflicting
`project.split_strategy=stratified_random` fails contract validation.
These run commands require the subsequent pipeline implementations.

Normal CLI usage starts in the selected repository (Kedro performs CLI project
discovery). A programmatic caller outside the repository selects it explicitly:

```python
from pathlib import Path
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

project_root = Path("path/to/ASI-s29379-s30253").resolve()
bootstrap_project(project_root)
with KedroSession.create(project_path=project_root, env="ci") as session:
    parameters = session.load_context().params
```

Use an absolute path when starting elsewhere. An optional absolute `conf_source`
can select a separate configuration directory without changing the project root.
Catalog relative file paths remain anchored by Kedro to the selected project.

Regression tests exercise an unrelated CWD, a second checkout containing different
settings, a conflicting `ASI_PROJECT_ROOT`, spaces and Unicode in project paths,
separate configuration directories, and two live contexts. Existing tests cover
subdirectories, local/CI overrides, runtime precedence, and contract rejection.

GitHub Actions runs the environment, discovery, pre-commit, pytest and sample
notebook checks on Ubuntu, Windows and macOS with Python 3.11. Each job installs
Kedro using the same pinned requirement and constraints. Bash is supplied by the
runner/activation action; Python code does not depend on Bash for path resolution.
`PYTHONUTF8=1` makes Python text handling consistent across runners, and Kedro
telemetry is disabled. CI uses committed sample data and requires no full-data
download or external service credentials. Pipeline execution is added in S2-4.

Only Linux has been exercised locally. Windows/macOS compatibility and fresh
runner installation remain to be confirmed by the matrix after publication.
