# ADR-1: Gdzie odbywa się przetwarzanie danych

- **Status:** przyjęty
- **Sprint:** S2
- **Autor:** Anton Sudak (`sanderJa`, s30253)
- **Recenzent:** Arrow (`arrow501`, s29379)

## Kontekst

Pipeline Kedro odtwarza baseline Bank Marketing ze Sprintu 1. Potrzebujemy wspólnych reguł dla treningu i przyszłego API oraz chronologicznego podziału danych, który pozwala uczciwie ocenić model na późniejszych rekordach.

## Rozważone opcje

1. Wykonać normalizację, filtrowanie i dopasowanie preprocessingu w `data_processing` przed podziałem danych.
2. Rozdzielić normalizację pojedynczych rekordów, filtrowanie treningowe i preprocessing dopasowywany przez model.

## Decyzja

Wybieramy opcję 2.

- **Input normalization** jest wspólna dla treningu i przyszłego API. `normalize_input()` ujednolica pojedynczy rekord bez targetu i statystyk całego zbioru. W treningu wywołuje ją `data_processing`.
- **Training-only filtering** działa wyłącznie podczas przygotowania danych treningowych. `filter_training_rows()` usuwa rekordy bez targetu `y`, bo bez etykiety nie da się ich wykorzystać do treningu ani ewaluacji.
- **Model preprocessing** jest częścią `sklearn.pipeline.Pipeline`. Imputacja, skalowanie i kodowanie są dopasowywane po podziale na danych treningowych. Zapisany model stosuje te same transformacje przy predykcji.

| Wielkość | Wartość | Skąd pochodzi |
| --- | ---: | --- |
| Rekordy z `pdays=-1` | 36 954 z 45 211 | Zliczenie w pełnym pliku `bank-full.csv` |
| Rekordy treningowe | 36 168 | `split.n_train` w [raporcie bazowym](../../metrics.json) |
| Rekordy testowe | 9 043 | `split.n_test` w [raporcie bazowym](../../metrics.json) |

Normalizacja zachowuje rekordy z `pdays=-1`, które stanowią większość zbioru. Wydzielenie 9 043 rekordów do testu pozwala ocenić model na danych, które nie wpłynęły na dopasowanie preprocessingu.

## Odrzucone alternatywy

Odrzuciliśmy dopasowanie preprocessingu przed splitem. Mediana do imputacji obliczona na pełnych danych uwzględniałaby rekordy testowe, co prowadziłoby do wycieku danych.

## Konsekwencje

`prepared_data` zachowuje target i kolejność rekordów. API będzie wywoływać wspólną normalizację przed predykcją. Zmiana transformacji wymagających dopasowania oznacza ponowny trening i zapis modelu.

## Historia

| Data | Zmiana |
| --- | --- |
| 2026-11-12 | Utworzenie ADR-1 |
