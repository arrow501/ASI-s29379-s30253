# S3-3 — Wspólny pakiet modelu, metadata.json i W&B Artifact

Owner: Anton (`sanderJa`, s30253) · Review: Arrow (`arrow501`, s29379)

## Cel i zakres

Uzupełnić scaffold `src/asi_project/model_package.py`: jeden sposób zapisu i odczytu
używany przez trening, pomiary, smoke test oraz przyszłe API.

- Doprecyzować z Arrow kontrakt w `src/asi_project/README.md`: katalog modelu,
  `metadata.json`, wejście predykcji i schemat wyniku. Uwzględnić sklearn i AutoGluon.
- Metadane zawierają cechy, target, klasę pozytywną, metryki, Git SHA, fingerprint
  danych i wersje bibliotek; loader sprawdza wymagane pola i kompletność pakietu.
- Zapewnić przenośność pakietu: ładowanie nie zależy od oryginalnego katalogu treningu.
- Dodać zapis/pobranie W&B Artifact. Po wspólnym wyborze S3-2 opublikować wybrany
  pakiet z aliasem `production` i zapisać identyfikator konkretnej wersji.

## Kryteria odbioru

- [ ] Trening i odbiorcy używają wspólnych funkcji; nie powstają równoległe loadery.
- [ ] Pakiet zawiera cały model i komplet wymaganych metadanych.
- [ ] Testy wykonują rzeczywisty zapis/odczyt w katalogu tymczasowym i porównują
  predykcje przed i po zapisie, także po przeniesieniu pakietu.
- [ ] Sprawdzono błędne/brakujące metadane i niekompletny pakiet.
- [ ] Potwierdzono zapis/odczyt sklearn i AutoGluona; cięższe sprawdzenie AutoGluona
  ma osobną komendę, jeśli nie należy do podstawowego CI.
- [ ] Wybrany pakiet jest rzeczywistym W&B Artifact z aliasem `production`;
  identyfikator wersji i pochodzenie odpowiadają decyzji S3-2.
- [ ] Dokumentacja zawiera komendy zapisu, pobrania i wczytania bez treningu.
- [ ] Arrow wykonał merytoryczne review.

## Zależności i przekazanie

Rozwój równolegle z S3-1, początkowo na baseline S2. Kontrakt loadera udostępnić
wcześnie do S3-2 i S3-4. Końcowa publikacja `production` wymaga wspólnej decyzji S3-2.
Testy lokalne ani mock klienta W&B nie potwierdzają publikacji artefaktu.
