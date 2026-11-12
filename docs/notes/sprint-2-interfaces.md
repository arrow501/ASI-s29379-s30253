# Sprint 2 — konfiguracja i interfejsy Kedro

Za fundament odpowiada Arrow (S2-1, #10), review robi Anton.
Instrukcja połączenia obu pipeline’ów: [handoff S2-2/S2-3](sprint-2-handoff.md).

## Skąd biorą się ustawienia

`project.yaml` pozostaje źródłem list cech i parametrów splitu, a `datasets.yaml`
— reguł kontraktu. Kontekst Kedro udostępnia je nodom jako `params:project`.
Nie kopiujemy tych wartości do kolejnych plików konfiguracyjnych.

Kolejność nadpisań: ustawienia projektu → `conf/base` → wybrane środowisko → CLI.
Słowniki są łączone, listy zastępowane. Częściowe nadpisanie zachowuje pozostałe
pola. Niedozwolone cechy, błędne parametry i zmiany sprzeczne z kontraktem są
odrzucane przed uruchomieniem nodów. W S2-3 sprawdzamy też `training` i `reporting`.

Kontekst czyta ustawienia z jawnie wybranego katalogu projektu, niezależnie od
bieżącego katalogu terminala. Ten katalog ma pierwszeństwo przed `ASI_PROJECT_ROOT`.
Ścieżki plików należą do catalogu, nie do nodów. Domyślne środowisko to `local`;
`ci` wybiera próbkę i osobne ścieżki wyników.

## Połączenie pipeline’ów

| Pipeline | Wejście | Wyjście | Osoba odpowiedzialna |
| --- | --- | --- | --- |
| `data_processing` | `raw_data`, `params:project` | `prepared_data` | Anton |
| `data_science` | `prepared_data` i parametry treningu | `trained_model`, `metrics` | Arrow |

Registry składa obie fabryki w `__default__`. `prepared_data` zachowuje target
oraz indeks i kolejność źródłowych rekordów. Split i cały preprocessing wymagający
`fit` należą do `data_science`; szczegóły są w instrukcji integracji.

## Uruchamianie i sprawdzenia

```bash
python -m pip install -r requirements/preflight-s2.txt -c constraints.txt
kedro info
kedro registry list
pytest -q
```

`kedro run` uruchamia przygotowanie i trening na pełnych danych, a
`kedro run --env ci` korzysta z próbki. Setup i ścieżki wyników:
[Kedro quickstart](../../README.md#kedro-quickstart--sprint-2).

CI jest skonfigurowane dla Linuxa, Windows i macOS z Pythonem 3.11. Lokalna
weryfikacja na Linuxie nie zastępuje wyników pozostałych runnerów. Testy obejmują
wybór katalogu projektu, nadpisania local/CI i CLI oraz połączenie fabryk.
