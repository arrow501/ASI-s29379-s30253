# Pakiet `asi_project`

Wspólny kod projektu. Jeden pakiet gwarantuje, że trening i serwowanie korzystają z tej samej
logiki (np. `normalize_input` działa identycznie w notebooku, w pipeline Kedro i w API).

## Aktywne w Sprincie 1

| Moduł | Rola |
|---|---|
| `datasets.yaml` | kontrakt zatwierdzonych datasetów (prowadząca; zespół nie edytuje) |
| `config.py` | odczyt `project.yaml` i rejestru datasetów |
| `data.py` | kontrakt danych, input normalization, training-only filtering, X/y, split |
| `model.py` | DummyClassifier i gotowy sklearn Pipeline z model preprocessing |
| `metrics.py` | metryki i kontrakt `metrics.json` |
| `download.py` | pobieranie danych i odcisk SHA-256 |

## Pipeline’y S2/S3 i granica z serwowaniem

| Element | Sprint | Kontrakt |
|---|---|---|
| `pipelines/data_processing/` | S2 | nody: walidacja → input normalization → training-only filtering (funkcje z `data.py`) |
| `pipelines/data_science/` | S2 | baseline: split → trening Pipeline → ewaluacja |
| `pipelines/experiments/` | S3 | podział/foldy → trening → zamrożenie decyzji → opcjonalny test |
| `provenance.py`, `experiment_store.py` | S3 | fingerprint/SHA oraz lokalizacja wyników z catalogu |
| `pipeline_registry.py`, `settings.py` | S2 | pliki projektu Kedro (rejestracja pipeline'ów, ustawienia) |
| `model_package.py` | S3 | model i `metadata.json` (cechy, target, klasa pozytywna, próg, metryki, SHA, fingerprint, wersje); wspólny zapis/odczyt dla Kedro, smoke modelu i API |
| `tracking.py` | S3 | logowanie gotowych wyników do W&B lub zapis offline; `disabled` pomija SDK |
| `model_artifacts.py` | S3 | publikacja i pobranie W&B Artifact, zapis konkretnej wersji |
| `benchmark.py` | S3 | pomiary natywnych modeli lub pakietów w osobnych procesach |
| `api/` | S4 | usługa predykcji — patrz `api/README.md` |

Zasada: moduły aktywne w Sprincie 1 nie importują Kedro, W&B, AutoGluona, FastAPI ani innych
bibliotek przyszłych sprintów. Pilnuje tego test w `tests/test_repo_hygiene.py`.

## Kontrakt pakietu modelu

Pakiet zawiera zapisany model oraz metadane potrzebne do jego
wczytania i użycia bez ponownego treningu.

Trening, pomiary, smoke test i przyszłe API korzystają ze wspólnych
funkcji zapisu, wczytywania i predykcji z `model_package.py`.

### Struktura i tworzenie pakietu

Dla sklearn i AutoGluona stosujemy tę samą strukturę:

```text
<package_dir>/
├── model/
└── metadata.json
```

Wspólny katalog `model/` ujednolica strukturę pakietu, poprawia
czytelność i ułatwia jego obsługę. Sposób wczytywania określa pole
`model_kind` w metadanych.

Katalog pakietu powstaje podczas wywołania funkcji zapisu dla
wytrenowanego modelu. Nie tworzymy go ręcznie ani podczas importowania
modułu. Funkcja otrzymuje model, metadane i ścieżkę docelową.

Lokalne pakiety zapisujemy w `data/06_models/`, przykładowo:

```text
data/06_models/baseline_package/
├── model/
│   └── pipeline.pkl
└── metadata.json
```

Dla baseline’u sklearn zapisujemy cały Pipeline, obejmujący preprocessing
i klasyfikator. Referencyjny DummyClassifier zapisujemy samodzielnie —
nie wymaga preprocessingu cech.

Dla AutoGluona katalog `model/` zawiera kompletny zapis
TabularPredictor, wraz z preprocessingiem i wymaganymi modelami
składowymi.

Jeden pakiet reprezentuje jeden model lub jeden predictor.
Pakiet musi działać po przeniesieniu do innego katalogu,
bez dostępu do pierwotnego katalogu treningowego.

### Metadane

Plik `metadata.json` zawiera następujące wymagane pola:

| Pole | Zawartość |
| --- | --- |
| `model_kind` | `sklearn` albo `autogluon` |
| `model_files` | Lista plików modelu ze ścieżkami względem katalogu `model/` |
| `dataset_id` | Identyfikator datasetu |
| `numeric_features` | Uporządkowana lista cech liczbowych |
| `categorical_features` | Uporządkowana lista cech kategorycznych |
| `target` | Nazwa przewidywanej zmiennej |
| `negative_class` | Oryginalna etykieta klasy negatywnej |
| `positive_class` | Oryginalna etykieta klasy pozytywnej |
| `decision_threshold` | Próg klasyfikacji z zakresu `[0, 1]` |
| `metrics` | Metryki pogrupowane według zbioru, np. `validation` i `test` |
| `git_sha` | SHA kodu użytego podczas treningu |
| `git_dirty` | Wartość logiczna informująca o niezacommitowanych zmianach podczas treningu |
| `data_fingerprint` | Fingerprint danych użytych w eksperymencie |
| `versions` | Wersja Pythona i bibliotek wymaganych do wczytania modelu |

Metadane pochodzą z eksperymentu zapisywanego modelu.
Pakowanie i publikacja nie zastępują ich bieżącą konfiguracją,
aktualnym SHA ani wynikami innego modelu.

Uwzględniamy wyłącznie rzeczywiście obliczone metryki.
Brak ewaluacji testowej oznacza brak sekcji `test`.

Loader sprawdza wymagane pola, ich typy i poprawność wartości,
a także kompletność plików modelu. Niepoprawny lub niekompletny
pakiet jest odrzucany z czytelnym błędem.

### Wejście predykcji

Wspólna funkcja predykcji przyjmuje pandas DataFrame.
Jeden wiersz oznacza jeden rekord.

DataFrame musi zawierać wszystkie cechy wskazane w metadanych.
Kolejność kolumn wejściowych może być dowolna — funkcja wybiera
i porządkuje cechy zgodnie z metadanymi.

Brak wymaganej kolumny powoduje błąd. Dodatkowe kolumny,
w tym target i cechy wykluczone, nie trafiają do modelu.

Przed predykcją wykonywana jest wspólna normalizacja
`normalize_input`. Nie stosujemy filtrowania treningowego
i nie wymagamy obecności targetu.

Brakujące wartości w istniejących kolumnach obsługuje zapisany
preprocessing modelu. Wczytywanie i predykcja nie wykonują `fit`.

### Wynik predykcji

Dla sklearn i AutoGluona wspólna funkcja zwraca DataFrame:

| Kolumna | Zawartość |
| --- | --- |
| `prediction` | Oryginalna etykieta przewidzianej klasy |
| `probability_positive` | Prawdopodobieństwo klasy pozytywnej z zakresu `[0, 1]` |

Klasa pozytywna jest zwracana, gdy:

```text
probability_positive >= decision_threshold
```

W przeciwnym przypadku zwracana jest klasa negatywna.
Wewnętrzne kodowanie klas modelu jest mapowane na etykiety
zapisane w metadanych.

Wynik zachowuje liczbę, kolejność i indeks rekordów wejściowych.
Predykcja nie usuwa rekordów.

### Publikacja i pobieranie

Cały katalog pakietu publikujemy jako W&B Artifact.
Po wspólnym zatwierdzeniu wyboru modelu w S3-2 wybrana wersja
otrzymuje alias `production`.

Zapisujemy również pełny identyfikator konkretnej wersji
artefaktu. Alias `production` może z czasem wskazywać inną wersję.

Pobrany pakiet ma tę samą strukturę i jest wczytywany tym samym
loaderem co pakiet lokalny. Pobranie i wczytanie nie wymagają
ponownego treningu.


### Komendy obsługi pakietu

Polecenia poniżej są przeznaczone do PowerShell i uruchamiane z katalogu głównego repozytorium.
Przygotowanie środowiska jest opisane w głównym README. Pakiet projektu musi być zainstalowany:

```powershell
conda activate asi-ml
python -m pip install --no-deps -e .
$env:KEDRO_DISABLE_TELEMETRY = "1"
```

**Zapis z istniejącego eksperymentu.** Przykład wybiera baseline z ostatniego zakończonego
pełnego eksperymentu z wynikiem testowym. Wymaga jego oryginalnych modeli, raportów
oraz pełnego CSV. Pakowanie nie uruchamia treningu. Przy braku danych użyj
`python scripts/download_data.py`.

```powershell
$env:WANDB_MODE = "disabled"
$runDir = Get-ChildItem data/08_reporting/experiments -Directory |
    Where-Object {
        (Test-Path (Join-Path $_.FullName "selection.json")) -and
        (Test-Path (Join-Path $_.FullName "test_metrics.json")) -and
        ((Get-Content (Join-Path $_.FullName "test_metrics.json") -Raw | ConvertFrom-Json).status -eq "completed")
    } |
    Sort-Object Name -Descending |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $runDir) { throw "Brak zakonczonego pelnego eksperymentu z wynikiem testowym." }
$packageDir = Join-Path "data/06_models/manual_packages" ("baseline-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
python scripts/manage_model_package.py export --run $runDir --candidate baseline --data data/01_raw/bank_marketing/bank-full.csv --sep ";" --output $packageDir --include-test
if ($LASTEXITCODE -ne 0) { throw "Eksport pakietu nie powiodl sie." }
```

Dla próbki wskaż właściwy run w `data/08_reporting/ci/experiments/`, plik
`data/01_raw/sample.csv` i `--sep ","`. `--candidate` przyjmuje nazwę z
`selection.json`, również kandydata AutoGluona. Bez `--include-test` zapisujemy
wyłącznie metryki validation. Fingerprint musi odpowiadać danym tego eksperymentu.
Nowe uruchomienia pipeline'ów S3 zapisują pakiety automatycznie; `export` obsługuje
również wcześniejsze runy. Istniejący katalog docelowy nie jest nadpisywany.

**Wczytanie i predykcja bez treningu.** Polecenie otwiera nowy proces, wczytuje
pakiet wspólnym loaderem i zwraca klasy oraz prawdopodobieństwa dla pięciu wierszy:

```powershell
python scripts/predict_model_package.py --package $packageDir --input data/01_raw/sample.csv --rows 5
```

**Publikacja po wspólnym wyborze S3-2.** Wskaż pakiet zatwierdzony przez obie osoby.
W&B wymaga zależności z `requirements/preflight-s3.txt` i działającego dostępu do konta.

```powershell
python -m pip install -r requirements/preflight-s3.txt -c constraints.txt
$env:WANDB_MODE = "online"
wandb login
$productionPackage = Read-Host "Sciezka pakietu zatwierdzonego w S3-2"
$productionPackage = (Resolve-Path -LiteralPath $productionPackage -ErrorAction Stop).Path
python scripts/manage_model_package.py publish --package $productionPackage --name bank-marketing-model --production
if ($LASTEXITCODE -ne 0) { throw "Publikacja nie powiodla sie." }
$receiptPath = Join-Path (Split-Path $productionPackage -Parent) ((Split-Path $productionPackage -Leaf) + ".wandb.json")
$published = Get-Content $receiptPath -Raw | ConvertFrom-Json
$published.artifact_ref
```

Bez `--production` publikacja nadaje tylko alias `latest`. Ustawienia `entity` i `project`
pochodzą z konfiguracji Kedro; `--env` wybiera środowisko (domyślnie `local`).
`WANDB_ENTITY` i `WANDB_PROJECT` mogą je nadpisać. Klucz API podajemy przez `wandb login`
lub środowisko procesu, poza kodem i repozytorium.

Potwierdzenie `<nazwa_pakietu>.wandb.json` powstaje obok pakietu i zawiera konkretny
identyfikator, np. `entity/project/bank-marketing-model:v3`. Zapisz ten identyfikator
w notatce odbioru razem z decyzją S3-2. Sam alias `production` może później zmienić wskazanie.

**Pobranie i użycie na innym checkoutcie.** Wklej zapisany identyfikator wersji.
Można również podać `entity/project/bank-marketing-model:production`; potwierdzenie
pobrania zapisze wersję, na którą wskazywał alias.

```powershell
$env:WANDB_MODE = "online"
wandb login
$artifactRef = Read-Host "Pelny artifact_ref z potwierdzenia publikacji"
$downloadDir = Join-Path "data/06_models/downloaded" ("model-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
python scripts/manage_model_package.py download --artifact $artifactRef --output $downloadDir
if ($LASTEXITCODE -ne 0) { throw "Pobranie nie powiodlo sie." }
$downloadReceipt = Join-Path (Split-Path $downloadDir -Parent) ((Split-Path $downloadDir -Leaf) + ".wandb.json")
(Get-Content $downloadReceipt -Raw | ConvertFrom-Json).artifact_ref
$env:WANDB_MODE = "disabled"
python scripts/predict_model_package.py --package $downloadDir --input data/01_raw/sample.csv --rows 5
```

Pobranie wymaga nowego katalogu docelowego. Pobrany pakiet jest sprawdzany tym samym
loaderem co lokalny. Skrypt nie dopasowuje modelu ani preprocessingu. Potwierdzenie
rzeczywistej publikacji i pobrania online wymaga wykonania tych operacji; lokalne testy
CLI nie zastępują odbioru W&B `production`.


### Wspólna predykcja i odbiorcy pakietu

Pomiary S3-2, lokalne sprawdzenie pakietu i przyszłe API korzystają z tej samej ścieżki:

```python
from pathlib import Path

import pandas as pd

from asi_project.model_package import load_model_package, predict_model_package
from asi_project.pipelines.experiments.artifacts import read_json

run_dir = Path("data/08_reporting/experiments/<run_id>")
packages = read_json(run_dir / "packages.json")
model, metadata = load_model_package(packages["baseline"])
rows = pd.read_csv("data/01_raw/sample.csv").head(5)
result = predict_model_package(model, metadata, rows)
```

`load_model_package` zwraca model i metadane. Model wczytujemy raz, a następnie
wywołujemy `predict_model_package` dla kolejnych partii danych. Odbiorca nie wybiera
samodzielnie pliku pickle, katalogu AutoGluona ani progu klasyfikacji.

Predykcja normalizuje wyłącznie wymagane cechy. Zachowuje kolejność i indeks,
również przy powtarzających się indeksach. Nie wymaga targetu i nie usuwa rekordów.
Puste wejście z wymaganymi kolumnami zwraca pusty wynik z ustalonym schematem.

Lokalne użycie poza procesem treningowym, bez W&B i bez ponownego fit:

```powershell
$env:WANDB_MODE = "disabled"
python scripts/predict_model_package.py --package "data/06_models/ci/packages/<run_id>/baseline" --input data/01_raw/sample.csv --rows 5
```

Dla pełnego CSV dodaj `--sep ";"`. Opcjonalne `--output <plik.json>` zapisuje wynik
zamiast wypisywać go w terminalu. Ten lokalny przykład nie pobiera artefaktu online;
niezależny smoke test `production` wykonano osobno w S3-4:
[wyniki odbioru](../../docs/notes/sprint-3-closeout.md).

### Użycie w pomiarach S3-2

Narzędzie pomiarowe wczytuje każdy pakiet przez `load_model_package` i mierzy
wywołania `predict_model_package` na tych samych wierszach. Czas predykcji obejmuje
normalizację, preprocessing modelu, obliczenie prawdopodobieństw i zastosowanie
zapisanego progu. Czas wczytywania mierzymy oddzielnie.

Przykład samego pomiaru czasu po wczytaniu modelu i przygotowaniu `rows`:

```python
from statistics import median
from time import perf_counter

for _ in range(3):
    predict_model_package(model, metadata, rows)

seconds = []
for _ in range(10):
    start = perf_counter()
    predict_model_package(model, metadata, rows)
    seconds.append(perf_counter() - start)

median_seconds = median(seconds)
package_bytes = sum(
    path.stat().st_size
    for path in Path(packages["baseline"]).rglob("*")
    if path.is_file()
)
```

Pełny benchmark S3-2 odpowiada za nowe procesy, pomiar pamięci, opis sprzętu,
liczby rekordów i powtórzeń oraz porównanie modeli. Zapisane metryki jakości są
w `metadata["metrics"]`; sekcja `test` występuje tylko po zakończonej ewaluacji testowej.
Nie przeliczamy F1 na pięciu przykładowych rekordach jako wyniku porównania modeli.


### Testy pakietu

Podstawowe testy działają bez W&B i bez treningu AutoGluona:

```powershell
pytest -q tests/test_model_package.py tests/test_model_package_inference.py tests/pipelines/test_experiment_runner.py
```

Sprawdzają rzeczywisty zapis/odczyt sklearn, zgodność klas i prawdopodobieństw po
przeniesieniu pakietu, odczyt w nowym procesie bez `fit`, walidację metadanych,
brakujące lub uszkodzone pliki oraz zgodność metadanych z zamrożonym eksperymentem.

Osobne sprawdzenie rzeczywistego AutoGluona wymaga zależności z
`requirements/preflight-s3.txt`:

```powershell
pytest -q tests/test_model_package_autogluon.py --run-autogluon-package
```

Ten test trenuje mały model RF na danych syntetycznych, zapisuje i przenosi pakiet,
przenosi oryginalny katalog treningowy pod inną nazwę i porównuje predykcje
w nowym procesie z zablokowanym `fit`. Bez przełącznika jest pomijany.
Testy pakietu nie potwierdzają publikacji ani pobrania rzeczywistego W&B `production`.

### Smoke test i przekazanie do API

`scripts/smoke_model.py` wykonuje pięć predykcji w osobnym interpreterze, wspólnymi
funkcjami `load_model_package` i `predict_model_package`. Usuwa target z wejścia,
blokuje `fit`, sprawdza schemat, liczbę i kolejność wierszy, klasy `no`/`yes`,
skończone prawdopodobieństwa w `[0, 1]` oraz zgodność klas z zapisanym progiem.

Tryb `--package` sprawdza lokalny pakiet bez W&B. Tryb `--artifact` zawsze pobiera
model do nowego katalogu tymczasowego i zapisuje identyfikator konkretnej wersji.
Błąd pobrania przerywa test. `--expected-ref` pozwala wymagać wersji zatwierdzonej
w ADR-2 i Model Card. Sam proces predykcji działa z wyłączonym W&B, bez klucza API.

```powershell
$release = Get-Content docs/notes/results/sprint-3-production.json -Raw | ConvertFrom-Json
$env:WANDB_MODE = "online"
wandb login
python scripts/smoke_model.py --artifact $release.requested_ref --expected-ref $release.artifact_ref --output "$env:TEMP/production-smoke.json"
```

Raport nie jest nadpisywany; dla kolejnej próby wybierz nową ścieżkę `--output`.
Szczegółowy [quickstart odbioru S3-4](../../docs/notes/sprint-3-smoke-test.md) opisuje
świeży checkout i środowisko, wyniki kontroli, Actions secret oraz workflow online.
API w S4 przyjmuje te same cechy bez targetu i używa tego samego loadera,
normalizacji i progu. Wynik pozostaje parą `prediction` / `probability_positive`,
w kolejności rekordów wejściowych. Utworzenie HTTP API jest osobnym etapem S4.
