# Sprint 1 — integracja

EDA i przygotowanie danych z [PR #7](https://github.com/arrow501/ASI-s29379-s30253/pull/7)
oraz baseline z [PR #6](https://github.com/arrow501/ASI-s29379-s30253/pull/6)
są scalone. Oba PR-y mają akceptację drugiej osoby w zespole.

`pdays=-1` jest normalizowane do braku danych typu `Int64`; imputacja medianą
odbywa się dopiero wewnątrz pipeline. Informację o braku wcześniejszego kontaktu
zachowuje `previous == 0`, co sprawdziliśmy na pełnym zbiorze. `unknown` zostaje
kategorią, ujemne salda są poprawne. Dane są już uporządkowane; nie stwierdziliśmy
duplikatów ani brakujących targetów, więc nie dodajemy filtrowania.

## Wynik po integracji

Chronologiczny split daje 36 168 rekordów treningowych i 9043 testowe.
F1 LogisticRegression wynosi 0,223807, precision 0,516456, recall 0,142857,
accuracy 0,687051. F1 DummyClassifier wynosi 0. Niski recall pozostaje
ograniczeniem baseline. Pełne wyniki są w `metrics.json`.

## Poprawki zamykające integrację

Testy wykrywają teraz usunięcie reguły `pdays=-1 → brak` i zmianę kolejności
rekordów podczas filtrowania. Wcześniej obie celowo wadliwe wersje przechodziły
wszystkie 67 testów. Po uzupełnieniu mamy 71 testów. Sprawdzamy też trening
i pojedynczą predykcję z nową reprezentacją braków.

Względna ścieżka `ASI_METRICS_PATH` jest liczona od katalogu projektu.
Polecenia testu na próbce w README zapisują i sprawdzają ten sam plik.
EDA wykonano na pełnych danych, a baseline na pełnych danych i próbce CI.

29.10.2026 wykonano lokalny smoke test przez Codex na świeżym klonie gałęzi,
z istniejącym środowiskiem `asi-ml` zgodnym z `constraints.txt`. Pobranie danych,
notebook 02 w nowym kernelu i walidacja zakończyły się poprawnie; wygenerowane
`metrics.json` jest identyczne z commitowanym. Test próbki CI potwierdził zapis
w `.pytest_cache/metrics_ci.json` w katalogu głównym repozytorium.

## Pozostałe potwierdzenia przed zamknięciem sprintu

- [ ] Potwierdzenie niezależnego smoke testu przez Antona (osobę inną niż autor baseline):
  świeży klon → środowisko → pobranie danych → notebook 02 od zera → metryki.
- [ ] Zielone CI i akceptacja drugiej osoby dla PR zamykającego integrację.

Te punkty wymagają potwierdzenia zespołu. `check_dod.py --sprint 1 --strict`
sprawdza wyłącznie część automatyczną i sam nie oznacza zamknięcia sprintu.
