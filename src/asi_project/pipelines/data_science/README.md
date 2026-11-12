# data_science — baseline Sprintu 2

Wejście: `prepared_data` po przygotowaniu Antona, z targetem `y` i zachowaną kolejnością.

1. `split_data` wybiera dozwolone cechy i robi split zgodny z kontraktem.
2. `train_models` dopasowuje sklearn Pipeline z LogisticRegression oraz dummy.
3. `evaluate_models` zwraca metryki obu modeli w schemacie ze Sprintu 1.

Cechy, split i seed pochodzą z `params:project`, limit iteracji z `params:training`,
a tryb danych z `params:reporting.data_mode`. Preprocessing wymagający `fit`
zostaje wewnątrz modelu. Nody nie czytają konfiguracji projektu ani ścieżek plików.

| Środowisko | Model | Metryki |
| --- | --- | --- |
| local/base | `data/06_models/baseline.pkl` | `data/08_reporting/metrics.json` |
| CI | `data/06_models/ci/baseline.pkl` | `data/08_reporting/ci/metrics.json` |

Catalog zapisuje cały model; dummy zostaje w pamięci. Zapisujemy accuracy,
precision, recall, F1 i confusion matrix. Referencyjny `metrics.json` nie jest nadpisywany.
Błędne parametry, rozjechane indeksy X/y, niezgodne cechy i brak zbieżności przerywają run.

```bash
pytest -q tests/pipelines/test_data_science.py
python scripts/check_data_science.py --data-mode sample
python scripts/check_data_science.py --data-mode full --compare-baseline
```

To sprawdzenie komponentu zapisuje standardowe artefakty catalogu. Pełne dane
muszą być wcześniej pobrane. Wspólny `kedro run` wykonuje przygotowanie i trening;
`kedro run --env ci` używa próbki. Uruchomienie tylko `data_science` wymaga
dostępnego `prepared_data` w catalogu.

[Wyniki lokalne](../../../../docs/notes/sprint-2-data-science.md) ·
[Instrukcja integracji dla Antona](../../../../docs/notes/sprint-2-handoff.md)
