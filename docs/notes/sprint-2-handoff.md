# S2-2/S2-3 — połączenie przygotowania danych z treningiem

Anton robi przygotowanie (#11), Arrow trening (#12). Obie części korzystają
z fundamentu S2-1. Registry już je składa — wystarczy zachować poniższe nazwy.

```text
raw_data + params:project → data_processing → prepared_data → data_science
```

## Pliki połączonego pipeline’u

| Plik | Zawartość |
| --- | --- |
| `pipelines/data_processing/nodes.py` w `src/asi_project/` | Nody korzystające z walidacji, normalizacji i filtrowania z S1 |
| `pipelines/data_processing/pipeline.py` | Niepusta fabryka `create_pipeline(**kwargs)` |
| `conf/base/catalog.yml` | Wejście `raw_data` dla pełnych danych |
| `conf/ci/catalog.yml` | Nadpisanie `raw_data` dla próbki |
| `tests/pipelines/` | Testy własnych nodów |

Catalog zawiera wejście `raw_data` oraz wyjścia treningu `trained_model` i `metrics`.
Przy scalaniu zachowujemy oba zestawy wpisów. `raw_data` definiujemy tylko raz
w danym środowisku. Wspólne nazwy datasetów zostają bez namespace; nazwy wewnętrznych
nodów są dowolne, byle nie kolidowały z `split_data`, `train_models`, `evaluate_models`.

Wpis do `conf/base/catalog.yml`:

```yaml
raw_data:
  type: pandas.CSVDataset
  filepath: data/01_raw/bank_marketing/bank-full.csv
  load_args:
    sep: ";"
```

Wpis do `conf/ci/catalog.yml`:

```yaml
raw_data:
  type: pandas.CSVDataset
  filepath: data/01_raw/sample.csv
  load_args:
    sep: ","
```

Separatory są różne. Nadpisanie CI zawiera cały wpis, żeby nie odziedziczyć `;`.
Zależności CSV, pickle i JSON są już w przypiętych requirements Sprintu 2.

## Parametry i kontrakt DataFrame’u

`params:project` zawiera `dataset_id`, `target`, `positive_class`, `split_strategy`,
`numeric_features`, `categorical_features`, `test_size` i `random_state`.
Wartości rozwiązuje kontekst Kedro — nie kopiujemy ich do YAML-a ani nie czytamy
`project.yaml` w nodach. Kontrakt pobieramy przez `load_registry()[project["dataset_id"]]`.
Grupy `training` i `reporting` zostają po stronie treningu.

`prepared_data` ma odpowiadać wynikowi `normalize_input` → `filter_training_rows` z S1:

- zachowuje target `y` jako `yes`/`no`, wszystkie ALLOWED FEATURES oraz źródłowy indeks
  i kolejność; nie tasujemy, nie sortujemy ani nie resetujemy indeksu;
- `pdays=-1` staje się brakiem, `unknown` zostaje kategorią, ujemne saldo jest poprawne;
- filtrowanie usuwa rekordy bez targetu; nie dodajemy innych reguł bez wspólnego ustalenia;
- nie ma tu splitu, kodowania targetu ani imputacji, skalowania czy encodingu wymagającego `fit`;
- kolumny wykluczone, np. `duration`, można zostawić albo usunąć — trening sam wybiera cechy.

W testach własnych nodów można użyć wspólnej asercji:

```python
from asi_project.pipelines.contracts import assert_prepared_data

assert_prepared_data(raw, prepared, contract)
```

Dataset może zostać w pamięci. Zapis pod `data/02_intermediate/` jest potrzebny
jedynie, jeśli chcemy uruchamiać `data_science` osobno; warto wtedy zachować typy
oraz indeks. `--pipeline data_science` nie uruchamia przygotowania automatycznie.

## Odbiór po połączeniu obu części

W `asi-ml` instalujemy `requirements/preflight-s2.txt` z `constraints.txt`, a potem:

```bash
pytest -q tests/pipelines
python scripts/check_pipeline_integration.py --env ci
kedro run --env ci
```

Po pobraniu pełnych danych powtarzamy check z `--env local` i robimy `kedro run`.
Checker sprawdza źródło CSV, kontrakt przygotowania, rzeczywisty pipeline domyślny,
wczytanie modelu i zgodność metryk. Przygotowanie uruchamia dwa razy: do kontroli
wyniku i w pełnym przebiegu. Trening uruchamia raz; wyniki trafiają do catalogu.

Przygotowanie S2-2 jest podłączone do treningu S2-3. Testowy stand-in służy do
sprawdzania błędów kontraktu; rzeczywiste fabryki i catalog obejmuje osobny test
integracyjny. Uruchomienie w GitHub CI i niezależny smoke test na świeżym klonie
pozostają w S2-4.
