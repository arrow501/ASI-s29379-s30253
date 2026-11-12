# Start Sprintu 2 — 29.10.2026

Sprint trwa od 29.10 do 12.11.2026. Termin oddania to stan `main` na początku zajęć
12.11. Integratorem jest Anton (sanderJa, s30253). Arrow (arrow501, s29379) recenzuje
integrację. Spotkanie kontrolne: 05.11.

Celem jest odtworzenie baseline'u jedną komendą `kedro run`, od surowych danych do
zapisanego modelu i metryk. Zakres, Definition of Done i podział czterech PR-ów są zapisane
w [SPRINT_2.md](../sprints/SPRINT_2.md#przydział-zadań-zespołu).

## Kolejność pracy

| PR | Owner | Warunek rozpoczęcia / odbioru |
| --- | --- | --- |
| S2-1: fundament Kedro | Arrow | Ustalenie interfejsów i konfiguracji przed pracą nad pipeline'ami |
| S2-2: przygotowanie danych | Anton | Po S2-1; własne testy nodów i review Arrow |
| S2-3: trening i ewaluacja | Arrow | Rozwój równolegle z S2-2; odbiór po integracji danych i review Antona |
| S2-4: integracja i dokumentacja | Anton | Po S2-2 i S2-3; CI, smoke test, dokumentacja i review Arrow |

Na spotkaniu 05.11 sprawdzamy stan obu pipeline'ów, zgodność interfejsów i blokery.
Przed zamknięciem Anton sprawdza pełny przebieg na świeżym klonie, zapis modelu i metryk,
porównanie ze Sprintem 1 oraz zielone CI na `main`. Obie osoby mają własne PR-y i robią
merytoryczne review pracy drugiej osoby.

## Stan na starcie

29.10 na środowisku Arrow `asi-ml` wykonano `kedro info`: polecenie zakończyło się poprawnie
i pokazało wersję 1.7.0, zgodną z `requirements/preflight-s2.txt`. Potwierdzenie środowiska
Antona pozostaje do uzupełnienia. To sprawdzenie instalacji, a nie działającego pipeline'u.

Istniejąca logika ze Sprintu 1 pozostaje podstawą implementacji. Zachowujemy podział
`time_ordered`, normalizację `pdays` i informację o wcześniejszym kontakcie w `previous`.
Nie dodajemy filtrowania danych bez uzasadnienia wynikającego z EDA. Model preprocessing
pozostaje w sklearn `Pipeline` i jest dopasowywany tylko na części treningowej.

Przed scaleniem wykonujemy `pytest`, `pre-commit run --all-files` oraz
`python scripts/check_dod.py --sprint 2 --strict`. Końcowy odbiór wymaga również
`kedro run` na pełnych danych i `kedro run --env ci` na próbce. Wyniki tych sprawdzeń
zostaną zapisane przy odpowiednich PR-ach; na starcie sprintu nie oznaczamy ich jako gotowe.

## Przygotowanie zmian

Gałąź `sprint-2-start` obejmuje plan i formalności startowe; implementacja fundamentu Kedro
należy do PR S2-1. Publikacja gałęzi, tworzenie PR i publikacja zadań na GitHubie wymagają
osobnej zgody użytkownika.
