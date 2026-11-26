# ADR-2: Wybór modelu do Sprintu 4

- **Status:** przyjęty 26.11.2026 — wybór zaakceptowany przez Arrow i Antona
- **Sprint:** S3 / #19
- **Autor:** Arrow (`arrow501`, s29379)
- **Recenzent:** Anton (`sanderJa`, s30253)

## Kontekst

API w Sprincie 4 potrzebuje modelu, który można wczytać i użyć bez treningu.
Porównaliśmy regresję logistyczną i cztery konfiguracje AutoGluona w trzech
protokołach: holdout 64/16/20, losowy bagging 80/20 oraz okna chronologiczne.
Oprócz jakości uwzględniamy czas predykcji, rozmiar pakietu i pamięć procesu.
Pełne wyniki i metodę pomiaru opisuje [raport #19](../notes/sprint-3-model-comparison.md).

## Decyzja

Wybieramy regresję logistyczną `baseline` z runu
`20261123T232543-chronological-3d2c20c4`
([run W&B](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/b4d90586db1e9551327d)).
Model wybrano według średniego F1 trzech okien chronologicznych, a następnie
ponownie dopasowano na pierwszych 80% danych (36 168 rekordów).
Próg `0.07166958577186548` dobrano na predykcjach walidacyjnych i zamrożono przed testem.

Do API przekazujemy pakiet z #20: sklearn Pipeline z dopasowanym preprocessingiem
oraz metadane opisujące cechy, etykiety, próg, wersje i pochodzenie modelu.
API korzysta ze wspólnego loadera i predykcji, z regułą:

```text
P(y=yes) >= 0.07166958577186548 → yes
```

Nie korzystamy z domyślnego `predict()` estymatora: stosuje próg 0,5, który dla
tego modelu daje F1 0,2238 zamiast 0,5210. Próg jest częścią wybranego modelu.

## Uzasadnienie i alternatywy

Chronologiczna walidacja odpowiada przewidywaniu wyników późniejszych kontaktów:
każde okno jest oceniane po treningu wyłącznie na wcześniejszych rekordach.
Baseline uzyskał średnie F1 okien **0,1707**, a GBM small **0,1688**.
Ta różnica jest mała; sama nie dowodzi przewagi jakościowej. Za baseline przemawiają
również mały pakiet, niższy koszt pojedynczej predykcji i brak zależności od AutoGluona
podczas inferencji.

| Model, okna chronologiczne | F1 testu | Pakiet [B] | 1 rekord [ms] | RSS po rozgrzewce [MiB] |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0,5210 | 6406 | 2,83 | 185,8 |
| GBM small | 0,4923 | 142 284 | 4,82 | 213,4 |

GBM small jest szybszy dla całego batcha 9043 rekordów: **17,33 ms** wobec
**23,72 ms** baseline’u. Wybrany model lepiej odpowiada jednak obsłudze pojedynczych
zapytań w planowanym API. Czasy są lokalnymi medianami, bez HTTP; RSS obejmuje
cały proces z bibliotekami, a nie sam model.
[Pomiary pięciu pakietów](../notes/results/sprint-3-package-benchmark.md) wykonano
przez loader #20 (`7bfd06f`); wszystkie zachowały prawdopodobieństwa oryginałów.

Rozważyliśmy też inne protokoły. W holdout GBM small osiągnął testowe F1 0,5390,
ale selekcję walidacyjną wygrał tree ensemble (testowe F1 0,1737).
Tree ensemble wybrany przez losowy bagging uzyskał na teście 0,4976.
Bagging miesza okresy wewnątrz development, dlatego słabiej odpowiada temu zastosowaniu.
Nie zmieniamy protokołu ani progu na podstawie najwyższego wyniku obejrzanego testu.

## Konsekwencje

Na teście baseline ma precision **0,3955** i recall **0,7630**:
3330 false positives oraz 677 false negatives. Próg maksymalizujący F1 nie uwzględnia
kosztu zbędnych kontaktów ani utraconych szans; tych kosztów nie oszacowaliśmy.

Udział pozytywów rośnie z **6,73%** w development do **31,58%** w teście,
a wyniki kolejnych okien są zmienne. Test był wcześniej oglądany w eksploracjach;
wyniki traktujemy jako benchmark, nie niezależny audyt generalizacji.
Ograniczenia i kontrakt wejścia opisuje [Model Card](../model_card.md).

Wybrany pakiet opublikowano jako `bank-marketing-model:v0` z aliasem `production`.
Pobranie po aliasie, digest i pięć predykcji zostały sprawdzone. Pełny identyfikator
jest w [Model Card](../model_card.md) i [potwierdzeniu](../notes/results/sprint-3-production.json).
Po merge #28 niezależny smoke online w GitHub Actions pobrał `production:v0`
i wykonał pięć predykcji bez treningu. Instalacja w nowym środowisku conda również
przeszła. Dowody: [odbiór S3](../notes/sprint-3-closeout.md).

## Historia

| Data | Zmiana |
| --- | --- |
| 2026-11-24 | Propozycja wyboru po lokalnym porównaniu modeli |
| 2026-11-26 | Wspólna akceptacja baseline’u i potwierdzenie pomiarów pakietów przez loader #20 |
| 2026-11-26 | Publikacja wybranego pakietu jako `bank-marketing-model:v0`, alias `production`; potwierdzone pobranie i predykcja |
