# Visual exploration

`notebooks/03_visual_exploration.py` is an optional marimo app, alongside the
existing Jupyter notebooks. It uses project configuration and dataset-contract
helpers without changing the model, baseline metrics, feature lists or split.
It is outside the course syllabus and is not a prerequisite for the required
Kedro pipeline, tests, or notebooks 01/02. Notebook 01 is restored to its original
committed version; the additional interactive exploration lives in notebook 03.

## Setup and launch

After the normal project environment setup, activate `asi-ml` and run:

```bash
python scripts/visual_exploration.py --setup --sample
```

This installs the pinned optional packages with the project's ML constraints,
installs the local package in editable mode without extra dependencies, checks
imports and opens the browser editor on committed sample data. It works with the
same command in Bash, PowerShell and macOS terminals; no full-data download or
editor extension is needed. It requires network only for the initial installation.

For subsequent launches, use `python scripts/visual_exploration.py`. Add `--sample`
to start on sample data, or choose the full local dataset in the app. Use
`python scripts/visual_exploration.py --check` to check dependencies and imports
without starting a server. Setup never runs implicitly; missing/mismatched optional
dependencies produce a clear instruction to rerun with `--setup`. Installation
failures stop the launcher before attempting to start marimo. The active Python
must be 3.11, matching the project environment. Core ML versions remain constrained.

The notebook uses this project environment rather than a PEP 723 sandbox.
The launcher fixes the loopback host. `--no-token` enables local pair discovery; it is
not a command for exposing a server publicly. On Windows, pair's shell helpers
need Git Bash or WSL with Bash, curl and jq. The Python notebook itself uses
portable paths and is independent of the launch directory once selected.

Open the displayed URL and keep the browser tab open. In marimo's Actions menu,
choose **Pair an agent** and copy the command for Codex, or ask the installed
marimo-pair skill to connect to that URL. A live session is the source of truth:
edit through the editor or pair, rather than editing its `.py` file behind it.

## VS Code

Install the official `marimo-team.vscode-marimo` extension (recommended by
`ASI.code-workspace`). Open `03_visual_exploration.py`, run **marimo: Open as
marimo notebook** in the Command Palette, then select the `asi-ml` Python
environment in the marimo editor. Use the existing environment, not a marimo
sandbox. Jupyter kernel selection and **Run Python File** are different paths;
running this file as a plain Python script performs a noninteractive run.

For first use, **Tasks: Run Task → ASI: set up visual exploration (optional)**
performs setup and starts with the sample. Afterwards, **ASI: visual exploration
(marimo, asi-ml)** launches the browser editor through the same Python script.
Both tasks use `conda run -n asi-ml`, independent of VS Code's selected interpreter.
Open `ASI.code-workspace` to see these tasks.
The browser editor and the VS Code editor can create separate sessions; choose
one active editor for changes and connect pair to that session's URL.

An automatic outline in the sidebar follows the notebook headings and provides
navigation across explanations, controls and working views. Long-form mathematical
text has its own readable block; controls and short tips stay near the relevant plot.

## Data and reactive boundaries

Domyślnie aplikacja wybiera pełne dane lokalne, a przy ich braku próbkę CI.
Selektor pozwala zmienić źródło. Sekcje 1–5, w tym PCA i t-SNE, używają wyłącznie
pierwszych 80% rekordów. Sekcje 6–8 opisują całą sekwencję, łącznie z testem
oglądanym już podczas wcześniejszej eksploracji i eksperymentów. Nie przywraca to
niezależności testu. Kontrakt wyklucza `duration` i target z reprezentacji.

Feature selection, numeric scaling, sampling count and sampling seed live in an
**Apply geometry** form. On startup the baseline geometry runs once. Subsequently,
unsubmitted edits do not change it. Preprocessing and PCA fit on training rows;
the displayed sample is uniform without replacement and shared by PCA and t-SNE.
The target only controls overlays and descriptive summaries.

Standard scaling reproduces baseline preprocessing; robust scaling changes the
numeric weighting via median/IQR. Categories are one-hot encoded, including
`unknown`. Neither metric is a neutral notion of customer similarity. Keeping
only up to eight PCs for the rotation tour discards the remaining variance.
Hover details and selection tables retain the original allowed feature values.

The browser performs hover, colour changes, zoom and rotations locally. Only box
selection returns row positions to Python for medians and a row table. Selection
does not refit preprocessing or PCA. An empty selection is shown explicitly.

## Rotation mathematics and interaction

Let `Z` contain the fixed, centred PCA scores in the first `d ≤ 8` components,
with no whitening. Let `B` be a `d × 2` matrix with orthonormal columns. Screen
coordinates are `Z B`. Initially its columns are `e1` and `e2`.

Each incremental angle applies a Givens rotation in a chosen ambient PC plane:

```text
B[i, :] ← cos(θ) old_B[i, :] − sin(θ) old_B[j, :]
B[j, :] ← sin(θ) old_B[i, :] + cos(θ) old_B[j, :]
```

The rotation preserves `BᵀB = I`; the cloud is fixed while its viewing plane
moves. With plane PC1 ↔ PC3, a quarter-turn gives horizontal `PC3`, vertical
`PC2`. Intermediate frames are true orthographic projections, not linear
interpolation between point positions, crossfades, refitted PCAs or axis rescaling.
The on-screen equations show the current combinations, and axis signs are arbitrary.

The variance badge is `Σ λk (Bk1² + Bk2²) / Σ λk`, using eigenvalues from the
whole fitted training set. It reports variance in the viewing plane relative to
all encoded dimensions, not explained label variance. Rotation changes projected
distances; it preserves the basis geometry, not every pairwise screen distance.

Choose two distinct components and drag or scrub the angle slider. Switching
planes retains the current basis and sets a new zero angle. A rotation plane
orthogonal to both viewing axes has no initial effect. Play advances that angle
at a controlled speed using `requestAnimationFrame`. It starts paused; Reset
stops it and restores PC1 × PC2. Arrow keys rotate and Space toggles playback.

Screen scale is fixed during motion. Default zoom uses the 98th percentile of
sample radii so extremes do not hide the cloud. The status explicitly counts
off-screen points; scroll to zoom or use **Fit all points**. No records are
discarded. Box selection pauses animation; selected identities survive subsequent
rotations and recolouring. Double-check selected groups in original features
before assigning a substantive meaning to a visible region.

## t-SNE and other views

Segment rates combine group support with bounded 95% Wilson intervals. Their
independence assumption may be optimistic for repeated contacts/campaigns.

t-SNE runs only after **Run t-SNE** submits its form, with up to 30 PCs as input
and at most 2,500 uniformly sampled points. Changing its colour does not refit.
Changing the geometry resets the t-SNE form, requiring a new submission. Compare
perplexities and seeds; island distances, sizes and apparent gaps are not global
geometry or evidence of predictive performance. Trustworthiness is measured
against the original encoded sample, including any PCA bottleneck; it has
quadratic memory cost. KL divergence should only be compared under matching
t-SNE settings. No predictive model is trained or tuned here.

## Chronological exploration

Section 5 uses source-record order rather than fabricated calendar dates.
Controls choose the trailing window length, equal-sized block count, numeric
feature and categorical feature. The positive-rate curve uses complete trailing
windows only. The numeric median and interquartile band omit missing values and
require at least 20% of a window (minimum two observations); its early windows
can therefore be partial. The band describes observed spread, not uncertainty
in the median. Literal `unknown` remains a category visible in composition plots.

The final paired view replaces a mostly empty missingness heatmap: its top panel
shows the share of records with `previous > 0`, and its lower panel compares
positive rates within the prior-contact and no-recorded-prior-contact groups.
Both share a record-order axis. Marker sizes show support, 95% Wilson intervals
show binomial uncertainty, and a support slider hides small group-blocks. Absent
groups stay gaps rather than fabricated zero rates. The groups are contact-history
records, not identified unique customers, and rate differences are not causal.

Rolling statistics are calculated on every training record before plotting at
most 1,200 positions. Block compositions sum to 100%; blocks have nearly equal
record counts, not equal durations. Hover and zoom inspect the curves without
refitting PCA or t-SNE. Overlapping windows are dependent; trends and apparent
cycles in row order do not establish calendar seasonality or causal effects.

## Diagnostyka splitów i progów — Sprint 3

Sekcja 6 pozwala porównać 64/16/20, bagging wewnątrz pierwszych 80% oraz trzy
rozszerzające się foldy chronologiczne. Zewnętrzny test zawsze pozostaje końcowymi
20%. Suwak validation oznacza procent development, więc domyślne 20% daje 16%
całości. Wcześniejsze foldy chronologiczne pokazują nieużyty przyszły development
na szaro. Losowe foldy są ilustracją stratyfikacji; AutoGluon zarządza własnym
podziałem, a RF/XT mogą korzystać z OOB. Mapa próbuje do 1800 pozycji, lecz
liczności, proporcje klas, Wilson i okno kroczące liczą się na wszystkich rekordach.

Sekcja 7 porównuje rozkłady kategorii i wskazuje kategorie nieobecne w train.
`unknown` i braki pozostają widoczne. Kolejność rekordów nie jest osią dat:
nie odtwarzamy roku ani długości kampanii z `day` i `month`.

Sekcja 8 czyta lokalne JSON-y OOF z `data/08_reporting/experiments`, bez W&B,
AutoGluon i ponownego treningu. Wybiera najnowszy ukończony run dla każdego
protokołu o zgodnym fingerprintcie danych i rozmiarze testu. Sprawdza indeksy,
etykiety i prawdopodobieństwa; odrzuca rekordy końcowego testu. Wykresy pokazują
F1 między oknami oraz precision, recall, F1 i macierz pomyłek przy wybranym progu.
Maksimum F1 na tych samych OOF jest opisem doboru progu, nie niezależnym wynikiem.
Zmiana splitu w sekcji 6 nie przelicza zapisanych eksperymentów w sekcji 8.
Przy braku zgodnych artefaktów, np. dla próbki CI, panel wyników pokazuje komunikat;
diagnostyka danych nadal działa. Kontrolki tych sekcji nie przeliczają PCA ani t-SNE.

Na pełnych danych domyślny podział ma 28 934 / 7 234 / 9 043 rekordów i odpowiednio
1 441 / 992 / 2 856 pozytywów (4,98% / 13,71% / 31,58%). To powód do sprawdzania
stabilności między okresami, nie do przesuwania granicy w poszukiwaniu maksimum testu.

## Validation

```bash
marimo check --strict notebooks/03_visual_exploration.py
python notebooks/03_visual_exploration.py
pytest -q tests/test_exploration.py tests/test_split_exploration.py
```

Set `ASI_DATA_MODE=sample` before the smoke check for a small run (PowerShell:
`$env:ASI_DATA_MODE = "sample"`; Bash: prefix the command with `ASI_DATA_MODE=sample`).
The noninteractive run checks initial views; t-SNE's submit-gated branch is tested
separately. Browser interaction needs an open notebook session.

Tests verify deterministic uniform sampling, chronological holdout exclusion,
Wilson intervals at zero/all positives, invalid inputs, and the actual JavaScript
rotation implementation. Node (optional for that test) checks quarter-turn
coordinates, inverse rotations and orthonormality after 10,000 steps.

Local wrap-up on 2026-11-10: 134 tests pass. Marimo's strict structure check,
sample-mode execution, optional dependency setup/check, and file lint/format
checks pass. Notebook 01 matches its committed version. Windows/macOS execution
and remote CI are not verified by these local Linux results. Course CI does not
need the optional marimo packages installed to run the test suite.

Sprawdzenie rozszerzenia splitów (2026-11-24): 157 testów przeszło, w tym 23 nowe
testy granic podziału, kategorii, progów i odczytu rzeczywistych JSON-ów OOF.
`marimo check --strict`, Ruff i nieinteraktywny przebieg na próbce przeszły.
W aktywnym kernelu sprawdzono protokoły, foldy, rozmiar validation, kategorię,
próg oraz przełączenie próbka → pełne dane: bez błędów komórek. Zmiany kontrolek
diagnostyki zachowały ten sam obiekt geometrii. Brak dostępnego podglądu
przeglądarki nie pozwolił zweryfikować wizualnie układu nowych paneli.

Official references: [marimo pair](https://docs.marimo.io/guides/generate_with_ai/marimo_pair/),
[interactive controls](https://docs.marimo.io/guides/interactivity/),
[custom widgets](https://docs.marimo.io/api/inputs/anywidget/),
[Plotly selections](https://docs.marimo.io/api/plotting/).
