# ADR-2: Wybór modelu do Sprintu 4

- **Status:** propozycja — oczekuje wspólnej decyzji i pomiarów paczki z #20
- **Sprint:** S3 / #19
- **Autor:** Arrow (`arrow501`, s29379)
- **Recenzent:** Anton (`sanderJa`, s30253); review jeszcze niewykonane

## Kontekst

Potrzebujemy modelu ładowanego bez treningu w przyszłym API. Porównaliśmy regresję
logistyczną i cztery konfiguracje AutoGluona na tym samym końcowym teście 9043 rekordów.
Jakość i koszty natywnych artefaktów są w [raporcie #19](../notes/sprint-3-model-comparison.md).
Test był już oglądany; nie uznajemy najwyższego testowego F1 za regułę nowej selekcji.

## Rozważone warianty

| Wariant | Wybór przed testem | F1 na teście | Natywny rozmiar |
| --- | --- | ---: | ---: |
| Baseline, chronologiczne okna i refit 80% | zwycięzca: średnie F1 okien 0,1707 | 0,5210 | 4,56 KiB |
| GBM small, ten sam protokół | średnie F1 okien 0,1688 | 0,4923 | ok. 3,15 MiB |
| Tree ensemble, holdout 64/16/20 | zwycięzca validation F1 0,3208 | 0,1737 | ok. 216 MiB |
| Tree ensemble, losowy bagging 80/20 | zwycięzca OOF | 0,4976 | ok. 313 MiB |

Pozostałe modele i pełne czasy/RSS są w tabelach raportu. Holdout GBM small osiąga
0,5390 na teście, ale nie wygrał selekcji validation. Same wyniki testu nie upoważniają
do zmiany protokołu. Losowy bagging jest użytecznym porównaniem, lecz miesza okresy
wewnątrz development i słabiej odpowiada predykcji kolejnych kontaktów.

## Propozycja decyzji

Do pakowania rekomendujemy `baseline` z runu
`20261123T232543-chronological-3d2c20c4`: model wybrany przez chronologiczne okna,
refit na 36 168 rekordach i próg **0,07166958577186548** zamrożony przed testem.
Różnica walidacyjna względem GBM small jest mała. Za baseline przemawia także
mały artefakt i prosty sklearn Pipeline; nie wymaga AutoGluona do inferencji.

Reguła predykcji to `P(y=1) >= decision_threshold`. Natywne `predict()` baseline
stosuje 0,5 i daje tu inne wyniki (F1 0,2238), dlatego próg musi być częścią metadanych
pakietu i kontraktu wspólnego loadera.

Docelowa postać to paczka z #20: model wraz z preprocessingiem, konfiguracją cech,
klasą pozytywną, progiem, wersjami i pochodzeniem. Format oraz wspólny zapis/odczyt
implementuje Anton. Niniejszy ADR nie ustala za niego layoutu plików.
Po wspólnym zatwierdzeniu paczka otrzyma alias `production`; publikacji jeszcze nie ma.

## Konsekwencje i warunki zatwierdzenia

Baseline ma recall 0,7630, ale precision 0,3955: 3330 false positives i 677 false
negatives na teście. Użycie progu maksymalizującego F1 nie jest optymalizacją kosztu
kampanii; tych kosztów nie oszacowaliśmy. Wzrost udziału pozytywów w teście ogranicza
możliwość przenoszenia wyniku na inne okresy.

Przed przyjęciem decyzji potrzebne są review Antona, zgoda obu osób i pomiary przez
wspólny loader na faktycznych paczkach (baseline + minimum dwa AutoGluony).
Rozmiary katalogów natywnych i RSS backendów nie opisują jeszcze paczki wdrożeniowej.
Następnie uzupełniamy W&B, wersję artefaktu i finalną [Model Card](../model_card.md).

## Historia

| Data | Zmiana |
| --- | --- |
| 2026-11-24 | Lokalny benchmark i propozycja do wspólnego review; bez zatwierdzenia production |
