# S2-3 — trening i ewaluacja baseline’u

Zadanie #12: Arrow; review: Anton. Gałąź bazuje na fundamencie S2-1.

Pipeline przyjmuje `prepared_data`, wybiera ALLOWED FEATURES, robi chronologiczny
split 80/20, trenuje LogisticRegression i dummy, a potem zapisuje model i metryki.
Korzysta z kodu ze Sprintu 1. Preprocessing wymagający `fit` zostaje w sklearn
Pipeline i jest dopasowywany wyłącznie na treningu.

Parametry pochodzą z Kedro: cechy, split i seed z `project`, limit iteracji
z `training.max_iter`, a tryb danych z `reporting.data_mode`. W metrykach zapisujemy
faktycznie użyte wartości. Pełne dane i CI mają osobne ścieżki wyników; referencyjny
`metrics.json` ze Sprintu 1 pozostaje bez zmian.

Walidacja odrzuca błędne ustawienia i literówki, rozjechane indeksy X/y oraz
niezgodne cechy w modelu i metadanych. Brak zbieżności przerywa trening z komunikatem
o zwiększeniu limitu iteracji — taki model nie jest zwracany do zapisu.

## Sprawdzone lokalnie — 10.11.2026

- **176 testów przechodzi**, pre-commit również.
- Pipeline działa na próbce i pełnych danych, a zapisany model daje się wczytać
  i odtwarza zapisane metryki.
- Cały rekord z pełnych danych jest identyczny ze Sprintem 1: 36 168 rekordów
  treningowych, 9043 testowych, F1 regresji logistycznej **0,2238**, F1 dummy **0**.

Do powtórzenia sprawdzenia komponentu:

```bash
python scripts/check_data_science.py --data-mode sample
python scripts/check_data_science.py --data-mode full --compare-baseline
```

Ten check przygotowuje wejście helperami S1, więc nie potwierdza jeszcze pełnej
integracji. Do odbioru pozostają #11, wspólny `kedro run` na pełnych danych i CI
oraz review Antona. Konkretne kroki: [handoff S2-2/S2-3](sprint-2-handoff.md).
Ścieżki wyników i nody: [README data_science](../../src/asi_project/pipelines/data_science/README.md).
Eksperymenty i wybór modelu należą do Sprintu 3.
