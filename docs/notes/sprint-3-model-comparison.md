# Sprint 3 — porównanie modeli (#19)

Owner: Arrow; review: Anton. Stan na 24.11.2026: lokalne porównanie gotowe,
wybór `production` wymaga wspólnego zatwierdzenia i pomiarów paczki z #20.

## Wynik i rekomendacja

Proponujemy przekazać do pakowania `baseline` z runu
`20261123T232543-chronological-3d2c20c4`. To regresja logistyczna z preprocessingiem,
refitowana na pierwszych 80% danych. Została wybrana przed testem według średniego
F1 trzech okien chronologicznych: **0,1707** wobec **0,1688** dla `gbm_small`.
Różnica jest niewielka; nie traktujemy jej jako dowodu istotnej przewagi jakościowej.

Na końcowych 9043 rekordach baseline ma **F1 0,5210**, precision **0,3955** i recall
**0,7630**. Jego natywny plik zajmuje **4,56 KiB**. Holdout `gbm_small` ma wyższe
testowe F1 (**0,5390**), lecz nie wygrał własnej selekcji validation. Nie wybieramy
protokołu ani nowego progu na podstawie najwyższego wyniku obejrzanego testu.
Rekomendację i jej warunki opisuje [ADR-2](../adr/0002-wybor-modelu.md).

[Pełne tabele](results/sprint-3-native-benchmark.md) obejmują baseline i cztery
konfiguracje AutoGluona w każdym z trzech protokołów, a w holdout także DummyClassifier
(łącznie 16 pomiarów). [Raport JSON](results/sprint-3-native-benchmark.json) zachowuje
wszystkie powtórzenia, metryki, macierze pomyłek i dane źródłowe. Nie wymaga marimo.

## Co mierzymy

Każdy model jest wczytywany w osobnym nowym interpreterze, kolejno. Proces nadrzędny
nie ładuje modeli. Wejście pochodzi z `raw_data` w catalogu Kedro; fingerprint musi
zgadzać się z runem. Kod odtwarza przygotowanie i końcowy split z zapisanego configu,
a następnie sprawdza identyczność cech, targetu, indeksu i kolejności testu między runami.
Wszystkie progi pochodzą z `selection.json`; benchmark niczego nie trenuje ani nie stroi.

- **Jakość:** accuracy, precision, recall, F1 dla `yes = 1`, macierz TN/FP/FN/TP.
  Po ponownym wczytaniu wyniki muszą odpowiadać `test_metrics.json`. Dodatkowo
  niezależna arytmetyka macierzy sprawdza metryki i liczbę rekordów.
- **Predykcja:** 2 rozgrzewki i 5 pomiarów `predict_proba` + zastosowanie progu;
  osobno batch 9043 rekordów i pierwszy pojedynczy rekord testu. Tabele podają medianę,
  JSON także minimum, maksimum i surowe czasy. Pojedynczy rekord nie opisuje rozkładu
  opóźnień całego API; nie mierzymy HTTP, normalizacji ani walidacji wejścia.
- **Ładowanie:** jeden pomiar na model, razem z importami jego backendu, po przygotowaniu
  danych w procesie pomiarowym. Bez startu interpretera. Hashowanie plików przed
  pomiarem rozgrzewa cache systemu: to nie jest pomiar zimnego dysku.
- **Pamięć:** RSS procesu przez `psutil`, w bajtach: przed loaderem, po loaderze i po
  rozgrzewce batcha. Podajemy całość oraz różnicę względem procesu z wczytanymi danymi
  i wspólnymi bibliotekami. Różnica obejmuje backend i model, nie samą strukturę modelu.
  RSS po rozgrzewce uwzględnia leniwe ładowanie AutoGluona. To snapshoty, nie peak RSS
  ani pomiar pamięci procesów potomnych; różnica może być ujemna.
- **Rozmiar:** suma logicznych bajtów wszystkich plików wskazanego natywnego artefaktu,
  także zależności, metadanych i logów w tym katalogu. Bez kompresji, bibliotek i danych.
  Nie jest to rozmiar docelowej paczki; AutoGluon może przechowywać niewykorzystywane
  już modele treningowe. Digest zawartości jest sprawdzany przed i po pomiarze.

Ustawiamy limity BLAS/OpenMP na 8 wątków przez środowisko i `threadpoolctl`.
Parametry równoległości zapisane wewnątrz estymatorów pozostają bez zmian; nie jest to
izolacja do 8 rdzeni ani pomiar na gwarantowanym sprzęcie wdrożeniowym. CPU, RAM, system,
wersje bibliotek i ustawienia znajdują się w JSON. Pomiar wykonano na Linux x86_64,
AMD Ryzen 9 7900X (24 logiczne CPU), z 31,1 GB RAM widocznymi dla systemu.
MiB oznacza 2²⁰ bajtów.

Dla rekomendowanego baseline mediana batcha wyniosła **15,58 ms**, pojedynczego
rekordu **1,68 ms**, a RSS po rozgrzewce **193,8 MiB**. Chronologiczny GBM small
osiągnął odpowiednio **9,45 ms**, **3,78 ms** i **220,5 MiB**: jest szybszy dla
dużego batcha, wolniejszy dla pojedynczego rekordu. Koszt zależy więc od sposobu użycia.

## Odtworzenie

```bash
conda activate asi-ml
python -m pip install -r requirements/benchmark-s3.txt -c constraints.txt
# Dla modeli AutoGluona potrzebne także requirements/preflight-s3.txt.
# Najpierw utwórz runy komendami z sprint-3-local-experiments.md.
KEDRO_DISABLE_TELEMETRY=1 WANDB_MODE=disabled python -m asi_project.benchmark \
  --run data/08_reporting/experiments/20261123T232543-holdout-6bf251c7 \
  --run data/08_reporting/experiments/20261123T232543-bagged-c7e1b345 \
  --run data/08_reporting/experiments/20261123T232543-chronological-3d2c20c4 \
  --output /tmp/s3-native-benchmark.json
python -m asi_project.benchmark --render /tmp/s3-native-benchmark.json \
  --output /tmp/s3-native-benchmark.md
```

Podstaw własne identyfikatory runów; `selection.json` wskazuje lokalne ścieżki modeli.
Wyników ani modeli nie pobieramy automatycznie. CLI odmawia nadpisania istniejącego
pliku. `--env ci` pozwala mierzyć baseline na próbce; nazwy runów i output muszą
wskazywać jej osobne wyniki. Na Windows ustaw zmienne środowiskowe w PowerShell,
a ścieżki `/tmp/` zastąp ścieżkami lokalnymi.

Raport jest zapisywany dopiero po kompletnym poprawnym porównaniu. Brak kandydata,
błąd procesu, timeout lub rozbieżność metryk przerywają pomiar. Testy obejmują rzeczywisty
zapis/odczyt modelu i osobny interpreter, progi, uszkodzony artefakt, zmienione dane,
niepełny raport, arytmetykę pamięci i czasu oraz konwersję jednostek.

## Pochodzenie i ograniczenia

Runy treningowe, config, SHA, dirty flag i fingerprint są w `sources` raportu JSON.
Fingerprint wejścia to `2a81c082b2a35c24ae36ba337231bd88e4a417d0a5e02346169078fac5a92253`
(`loaded-data-csv-v1`). SHA treningu w tych runach to
`c66e7f81e3e8bb1aa3630e165b7f24c768250508`, z `git_dirty=true`:
sam ten commit nie odtwarza całego ówczesnego kodu. Raport benchmarku zapisuje także
SHA/dirty i hash pliku kodu pomiarowego, hash selekcji oraz hash każdego artefaktu.

Test był wcześniej oglądany w eksploracjach i porównaniach #18. Stanowi wspólny
benchmark, nie niezależny audyt generalizacji. Udział pozytywów rośnie: 6,73% w pierwszych
80% wobec 31,58% w teście. Wyniki F1 zależą od tej zmiany i dobranego progu. Walidacja
chronologiczna także ma duży rozrzut: F1 baseline w oknach to 0,1367 / 0,0977 / 0,2778.
Losowy bagging miesza okresy development; jego OOF nie mierzy prognozy późniejszego okresu.

## Powiązanie z W&B

Tracking działa; w #18 zweryfikowano przez API metryki, konfigurację, provenance
oraz raporty eksperymentów. [Konfiguracja i dowody](sprint-3-wandb.md) opisują też
tymczasowy dostęp i możliwe opóźnienia lub błędy po stronie usługi.

Poniższe runy są publikacją tych samych lokalnych źródeł, których używa benchmark:

| Protokół | Model | Run W&B |
| --- | --- | --- |
| bagged | `baseline` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/c2fb5ef969486935cd00) |
| bagged | `catboost` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/4474be9f1a3faff7e0f1) |
| bagged | `gbm_large` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/3302a90ddafe0d90454c) |
| bagged | `gbm_small` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/5536c33748089b85ce07) |
| bagged | `tree_ensemble` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/99cc2500407db9cca170) |
| chronological | `baseline` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/b4d90586db1e9551327d) |
| chronological | `catboost` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/3191cb2e35a0947d4ae4) |
| chronological | `gbm_large` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/0bc30bd94211b4991361) |
| chronological | `gbm_small` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/55008bdeb4eb197026bb) |
| chronological | `tree_ensemble` | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/fa90ce6c2500d6447cb3) |

Holdout w benchmarku pochodzi z `20261123T232543-holdout-6bf251c7`.
Opublikowany holdout `20261124T161322-holdout-b410b77e` to osobny, ponowny trening;
nie przypisujemy jego URL starszym artefaktom ani pomiarom kosztów. Komplet 17 runów
jest w [rejestrze weryfikacji](sprint-3-wandb-runs.json).

Zapisane `wandb_url=null` w historycznym JSON benchmarku oznacza stan w chwili
pomiaru, przed publikacją. Zachowujemy ten raport bez zmian; powiązania powyżej
uzupełniają jego pochodzenie. W&B zawiera raporty typu `experiment-report`;
wersja paczki modelu i alias `production` pozostają częścią #20.

## Sprawdzenia

Po połączeniu ze zmianami W&B z #18 przeszło **412 testów**, w tym **32 dla benchmarku**,
oraz pre-commit dla
zmienionych plików i kontrola diffu. Wszystkie 16 świeżych procesów odtworzyło zapisane
metryki testu. Końcowe pomiary czasów wykonano po testach, bez równoległego uruchamiania
pytest. CI na Windows/macOS nie zostało jeszcze uruchomione dla tej gałęzi.

## Do zamknięcia #19

- Anton: wspólny loader i format paczki z #20, następnie niezależne review porównania.
  Powtórzyć koszty na rzeczywistych paczkach baseline i co najmniej dwóch AutoGluonów,
  sprawdzając identyczność predykcji i zachowanie progu. Obecny adapter natywny pozostaje
  narzędziem eksperymentów, nie zastępuje implementacji #20.
- Arrow i Anton: zatwierdzić albo zmienić rekomendację w ADR-2; dopiero wtedy oznaczyć
  Model Card jako finalną i przekazać identyfikator modelu do `production`.
- Po #20: dopisać konkretną wersję artefaktu modelu i alias `production`.
  Linki do runów eksperymentów są już dostępne powyżej.
