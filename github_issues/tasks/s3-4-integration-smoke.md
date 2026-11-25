# S3-4 — CI, niezależny smoke test production i przekazanie do Sprintu 4

Owner: Anton (`sanderJa`, s30253) · Review: Arrow (`arrow501`, s29379)

## Cel i zakres

Potwierdzić, że wybrany model działa poza procesem treningowym i jest gotowy do użycia
w API. Integratorem sprintu jest Arrow; Anton odpowiada za tę implementację i odbiór techniczny.

- Przygotować smoke test: pobranie W&B Artifact `production` → wczytanie wspólnym
  loaderem → predykcja dla pięciu wierszy → walidacja uzgodnionego schematu wyniku.
- Uruchamiać test w nowym procesie, bez ponownego fit i bez korzystania z lokalnego
  obiektu modelu lub oryginalnego katalogu treningowego.
- Rozszerzyć istniejące CI: `WANDB_MODE=disabled`, AutoGluon wyłączony parametrem
  albo ograniczony do krótkiego przebiegu, rzeczywisty test zapisu/odczytu pakietu.
- Uzupełnić quickstart i krótką notatkę odbioru z komendami, wynikami i kontraktem dla S4.

## Kryteria odbioru

- [ ] Lokalny test pakietu w CI działa bez sekretów i pełnych eksperymentów online.
- [ ] Smoke test online pobrał rzeczywisty `production` i wykonał pięć predykcji
  w czystym procesie bez treningu; zapisano konkretną wersję artefaktu.
- [ ] Sprawdzono liczbę i kolejność wyników, klasy oraz schemat odpowiedzi;
  jeśli wynik obejmuje prawdopodobieństwa, również ich zakres i klasę pozytywną.
- [ ] Niezależny odbiór Antona na świeżym checkoutcie potwierdza instrukcję;
  mock ani lokalny round-trip nie zastępują testu pobrania `production`.
- [ ] `pytest`, `pre-commit run --all-files` i `kedro run --env ci` przechodzą.
- [ ] CI jest zielone na wspieranych systemach; przed zamknięciem sprintu także na `main`.
- [ ] README/notatka opisują środowisko, pobranie, loader, wymagane wejście i wynik
  dla S4 oraz rozróżniają odbiór online od testów CI.
- [ ] Arrow wykonał merytoryczne review i sprawdził kompletność DoD sprintu.

## Zależności i przekazanie

CI i lokalny smoke test można rozwijać na baseline po udostępnieniu loadera S3-3.
Końcowy odbiór online wymaga wyników S3-1, wyboru S3-2 i opublikowanego artefaktu S3-3.
Implementacja API pozostaje w Sprincie 4.
