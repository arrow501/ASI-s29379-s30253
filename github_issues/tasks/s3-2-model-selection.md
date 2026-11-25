# S3-2 — Porównanie modeli, wybór production, ADR-2 i Model Card

Owner: Arrow (`arrow501`, s29379) · Review: Anton (`sanderJa`, s30253)

## Cel i zakres

Porównać baseline oraz minimum dwa warianty AutoGluona z S3-1 i przygotować wspólną
decyzję o modelu przekazywanym do Sprintu 4.

- Przygotować odtwarzalny pomiar F1, pozostałych metryk i macierzy pomyłek,
  czasu predykcji, rozmiaru pakietu oraz pamięci po wczytaniu.
- Używać wspólnych funkcji pakietu z S3-3. Opisać sprzęt, liczbę rekordów,
  rozgrzewkę/powtórzenia oraz sposób pomiaru pamięci, z oddzieleniem kosztu procesu.
- Zestawić wyniki i linki do W&B w krótkim raporcie w `docs/notes/`.
- Przygotować ADR-2 w `docs/adr/` oraz uzupełnić `docs/model_card.md` dla faktycznie
  wybranego modelu: przeznaczenie, dane, cechy, target/klasa pozytywna, metryki,
  pochodzenie i ograniczenia.

## Kryteria odbioru

- [ ] Tabela obejmuje baseline i minimum dwie konfiguracje AutoGluona na tym samym teście.
- [ ] Każdy wiersz zawiera F1, czas predykcji, rozmiar pakietu i pamięć po wczytaniu,
  z jednostkami i opisaną metodą pomiaru.
- [ ] Pomiary pamięci i ładowania wykonano w nowych procesach, bez pozostałości treningu;
  kod pomiarowy ma adekwatne sprawdzenia obliczeń i kompletności raportu.
- [ ] Dane raportu można powiązać z runem, SHA, fingerprintem oraz konfiguracją.
- [ ] Obie osoby zatwierdziły wybór modelu na podstawie jakości i kosztów użycia.
- [ ] ADR-2 zawiera rozważane warianty, konkretne liczby, decyzję i uzasadnienie.
- [ ] Model Card odpowiada wybranemu modelowi; nie zawiera niepotwierdzonych wyników.
- [ ] Raport można odtworzyć bez marimo. Anton wykonał merytoryczne review.

## Zależności i przekazanie

Wymaga wyników S3-1 i funkcji pakietu S3-3. Metodę pomiarów można przygotować wcześniej
na baseline. Przekazać Antonowi identyfikator wybranego modelu i finalne metadane do
publikacji `production`, następnie do odbioru S3-4.
