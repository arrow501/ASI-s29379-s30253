# S3-1 — Tracking W&B i eksperymenty AutoGluona w Kedro

Owner: Arrow (`arrow501`, s29379) · Review: Anton (`sanderJa`, s30253)

## Cel i zakres

Rozszerzyć istniejący `data_science` o reprodukowalne eksperymenty bez zależności od
notebooków. Pierwszy run W&B odtwarza baseline S2 bez zmiany jego parametrów i splitu.
Następnie uruchomić minimum dwie sensowne konfiguracje AutoGluona z jawnym limitem czasu.

- Logować parametry, metryki, Git SHA i fingerprint faktycznie użytych danych.
- Dodać konfigurację eksperymentów i przełączniki potrzebne do lekkiego przebiegu CI.
- Uzgodnić z Antonem i opisać dane runu oraz wynik eksperymentu potrzebny do pakowania
  i porównania; format pakietu rozwija S3-3.
- Zachować zewnętrzny chronologiczny test 20%. Przed eksperymentami opisać sposób
  walidacji wewnątrz części treningowej; test nie służy do fit, tuningu ani doboru progu.
- Zachować jedno źródło ustawień projektu i preprocessing dopasowywany tylko na treningu.

## Kryteria odbioru

- [ ] Baseline jest pierwszym runem W&B i odpowiada wynikom S2 w opisanej tolerancji.
- [ ] Minimum dwa warianty AutoGluona wykonano na pełnych danych przez pipeline Kedro.
- [ ] Każdy run ma parametry, limit czasu, metryki, SHA i fingerprint danych.
- [ ] Opisano wspólny podział danych, walidację, seed i klasę pozytywną; żaden wariant
  nie wykorzystuje testu podczas dopasowania.
- [ ] Tryb `WANDB_MODE=disabled` działa bez konta i sekretów, a AutoGluon można wyłączyć
  w lekkim przebiegu CI.
- [ ] Testy sprawdzają rozdzielenie danych i treść logowanych informacji, nie tylko
  samo wywołanie klienta W&B. Zapisano komendy uruchomienia i linki do rzeczywistych runów.
- [ ] Anton wykonał merytoryczne review.

## Zależności i przekazanie

Równolegle z S3-3 po uzgodnieniu kontraktów. Przekazać do S3-2 modele, konfiguracje,
identyfikatory runów i wyniki dla tych samych rekordów testowych.
Wizualizacje marimo są już wykonanym dodatkiem, poza zakresem tego zadania.
