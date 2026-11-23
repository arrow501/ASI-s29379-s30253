# Sprint 3 — eksperymenty lokalne (#18)

Owner: Arrow; review: Anton. Pipeline `experiments` działa bez W&B, konta i sieci.
Domyślny pipeline S2 i jego referencyjne `metrics.json` zachowują split 80/20.

## Aktywne warianty porównania

W każdym wariancie wspólny test to ostatnie 20% danych. 70/15/15 wycofano z aktywnego
porównania; stare wyniki pozostają archiwalne.

| Wariant | Dobór modelu i progu | Finalny trening |
| --- | --- | --- |
| 64/16/20 | chronologiczne validation 16% | pierwsze 64% |
| 80/20 bagged | losowe, stratyfikowane OOF w pierwszych 80% | bagging na pierwszych 80%; baseline refit na 80% |
| crossfold chronologiczny | trzy rozszerzające się foldy wewnątrz pierwszych 80% | baseline na 80%; AutoGluon refit ostatniego foldu na 80% |

Crossfold ma nominalne okna: train 0–32% → validation 32–48%, train 0–48% → validation
48–64%, train 0–64% → validation 64–80%. Dokładne granice pozycyjne i indeksy są
w `folds.json`. Preprocessing i early stopping korzystają tylko z odpowiedniego foldu;
progi pochodzą ze złączonych predykcji kolejnych okien. Konfigurację wybieramy według
średniego F1 okien przy tym progu; raportujemy również każde okno, minimum i odchylenie.
Wyniki validation po doborze progu pozostają wynikami selekcji, nie niezależnym testem.

Bagging ma pięć losowych foldów, jeden zestaw baggingu i brak stackingu. RF/XT mogą
użyć natywnego OOB zamiast pięciu kopii. AutoGluon współdzieli generator cech wewnątrz
development; ten OOF miesza okresy i nie zastępuje oceny chronologicznej.
Liczby foldów pochodzą z `conf/base/parameters_validation_protocols.yml`.

Pełne przebiegi potwierdzono dla obu nowych protokołów. Na wspólnym teście 9043 rekordów
model wybrany przez bagging (`tree_ensemble`) ma F1 0,4976; wybrany przez crossfold
baseline ma F1 0,5210. Dla porównania wybór z pojedynczego 64/16/20 ma F1 0,1737.
Aktywna tabela wszystkich modeli: `data/08_reporting/experiments/active-comparison.md`,
źródła i progi: `active-comparison.json` obok. Wszystkie trzy warianty używają tego
samego fingerprintu i końcowego testu 20%; ten test był już wcześniej oglądany.
Po dodaniu protokołów przeszło 278 testów, lint, format, `git diff --check`, `pip check`
oraz oba nowe pipeline'y na próbce CI bez AutoGluona.

```bash
# Bagging, wybór na OOF i końcowy test:
KEDRO_DISABLE_TELEMETRY=1 kedro run --pipelines experiments_bagged_test

# Trzy foldy chronologiczne, refit na development i końcowy test:
KEDRO_DISABLE_TELEMETRY=1 kedro run --pipelines experiments_crossfold_test
```

Bez sufiksu `_test` oba polecenia kończą się na walidacji i zapisie modeli. Dodanie
`--env ci` wykonuje je na próbce, tylko dla baseline'u (bez SDK AutoGluona i W&B).
Każdy `time_limit` obowiązuje pojedynczy `fit`, więc crossfold może zużyć trzy takie
budżety na konfigurację. Końcowy `refit_full` AutoGluona jest poza limitem `fit`;
odtwarza liczbę iteracji wybranego modelu. Żaden refit nie dostaje końcowego testu.

## Podział i wybór konfiguracji

Źródło parametrów splitu: `project.yaml`. Najpierw odcinamy ostatnie 20% danych jako test,
potem ostatnie 20% pozostałej części jako validation. Liczności zaokrąglamy w górę
dla testu i validation: na pełnych danych **28 934 train / 7234 validation / 9043 test**.
Indeks i kolejność wejścia pozostają zachowane; nie sortujemy po indeksie ani nie tasujemy.

Nody treningowe dostają tylko train i validation. Pipeline `experiments` nie ocenia testu.
`experiments_test` dodaje końcową ocenę wszystkich zamkniętych konfiguracji na ostatnich 20%.
Najpierw zapisuje `selection.json`: wybór na podstawie F1 validation, z alfabetycznym
rozstrzyganiem remisów. Wynik testowy nie zmienia wyboru ani progu.

`threshold_mode: validation_f1` dobiera próg osobno dla każdego modelu na validation
(maksymalne F1, przy remisie wyższy próg). Dummy zachowuje próg 0,5. Raport pokazuje
również wynik przy 0,5. Nie wykonujemy refit na 80%.

Sklearn preprocessing jest dopasowywany na train. Dla AutoGluona jawnie dopasowujemy
`AutoMLPipelineFeatureGenerator` na train przed `fit`, a validation przekazujemy jako
`tuning_data`. To istotne: domyślny generator AutoGluona analizuje train razem z tuning_data.
W wariancie 64/16/20 wyłączamy bagging, stacking i dynamic stacking. Weighted ensemble może dobierać wagi
na validation, więc jego wynik nie jest niezależną oceną na teście.
AutoGluon wykorzystuje `stopping_metric: log_loss` także do wewnętrznego wyboru modeli
i ensemble; porównanie konfiguracji projektu nadal opiera się na F1 validation.
Kolumny numeryczne konwertujemy bez dopasowania do `float64`, ponieważ nullable pandas
`Int64` (`pdays`) nie jest poprawnie rozpoznawane przez inferencję typów AutoGluona.

## Konfiguracje i komendy

Instalacja zgodnie z przypięciami projektu:

```bash
conda activate asi-ml
python -m pip install -r requirements/preflight-s3.txt -c constraints.txt
python -m pip check
```

`conf/base/parameters_experiments.yml` definiuje cztery konfiguracje: `gbm_small`,
`gbm_large`, `catboost`, `tree_ensemble` (GBM/CAT/RF/XT). Każda ma limit `fit` 120 s,
8 CPU i 0 GPU. Limit AutoGluona jest budżetem treningu, nie twardym limitem całego
procesu: preprocessing, ensemble, zapis i sprawdzenie odczytu mogą dodać czas.
Baseline LogisticRegression i dummy są uruchamiane przed kandydatami na tym samym train.

```bash
# Lekki przebieg na próbce, bez SDK AutoGluona i W&B:
KEDRO_DISABLE_TELEMETRY=1 kedro run --pipelines experiments --env ci

# Rzeczywiste cztery konfiguracje AutoGluona na próbce:
KEDRO_DISABLE_TELEMETRY=1 kedro run --pipelines experiments --env ci --params experiments.autogluon_enabled=true

# Ten sam zestaw na pełnych danych:
KEDRO_DISABLE_TELEMETRY=1 kedro run --pipelines experiments

# Pełny przebieg z wyborem na validation i końcowym testem:
KEDRO_DISABLE_TELEMETRY=1 kedro run --pipelines experiments_test
```

Na PowerShell ustawić wcześniej `$env:KEDRO_DISABLE_TELEMETRY="1"`, a następnie wykonać
samo `kedro run ...`. Pełne dane muszą już istnieć zgodnie z catalogiem.

## Wynik i przekazanie do Antona

Catalog wskazuje `experiment_store`: pełne dane w `data/08_reporting/experiments/`,
próbka w `data/08_reporting/ci/experiments/`. Każdy przebieg dostaje nowy katalog
z czasem UTC i losowym identyfikatorem; poprzednie wyniki pozostają zachowane.

| Plik | Zawartość |
| --- | --- |
| `config.json` | efektywne parametry projektu, treningu i wszystkich kandydatów |
| `provenance.json` | SHA, dirty flag, fingerprint wejścia i wersje bibliotek |
| `split.json` | liczności i przedziały pozycji; prawy kraniec jest wyłączny |
| `metrics.json` | status, metryki validation, wstępny czas predykcji i ścieżki modeli |
| `selection.json` | wybór na validation zapisany przed oceną testu (tylko `experiments_test`) |
| `test_metrics.json` | końcowe metryki testowe przy zamrożonych progach |
| `baseline.pkl` | natywny zapis sklearn na potrzeby eksperymentu |
| `dummy.pkl` | natywny zapis modelu odniesienia |
| katalog nazwy kandydata | natywny zapis AutoGluona, w tym jego logi |

`prediction_seconds` to pojedynczy pomiar pomocniczy, jeszcze nie benchmark z #19.
Zapis/odczyt każdego modelu jest sprawdzany przez porównanie prawdopodobieństw validation.
Wyniki częściowe są zapisywane po każdym kandydacie; błąd kończy komendę błędem i statusem
`failed`, zamiast udawać udany zestaw eksperymentów.

Fingerprint (`loaded-data-csv-v1`) identyfikuje wczytane `raw_data`, wraz z indeksem,
nagłówkami i kolejnością. Jest inny niż hash bajtów oryginalnego pliku z downloadera.
Dirty flag oznacza, że sam SHA nie odtwarza jeszcze lokalnego stanu kodu.

Anton (#20) otrzymuje natywny model, efektywną konfigurację, provenance i metryki.
Protokoły CV dodają `folds.json` oraz `<model>_oof.json` z indeksami, etykietami i
prawdopodobieństwami do odtworzenia selekcji. Końcowy model crossfold jest zapisany
w katalogu ostatniego foldu jako wariant `_FULL`; ścieżkę wskazuje `model_path`.
Te pliki są wynikami eksperymentu; docelowy pakiet i wspólny loader pozostają jego zadaniem.
Nie publikujemy modelu ani nie przypisujemy aliasu `production` w tej ścieżce.

## Diagnoza F1=0 i lokalna weryfikacja

Pierwszy przebieg wszystkich modeli dawał F1=0 przy progu 0,5. Niezależne przeliczenie
potwierdziło metryki: modele przewidywały same zera. Train ma 1441 pozytywów z 28 934
(5,0%), validation 992 z 7234 (13,7%); validation zawiera też miesiące `apr` i `mar`
niewidziane w train. Dodatkowo F1 przy 0,5 było płaskim sygnałem dla early stopping
i wewnętrznego wyboru modeli. Sama zmiana na log-loss nie usunęła zerowego F1.

Po wprowadzeniu progów validation, probabilistycznej metryki wewnętrznej i poprawie
typu `pdays`, pełny przebieg `20261123T225040-363c1efb` zakończył się następująco:

| Konfiguracja | Próg z validation | F1 validation | F1 test |
| --- | ---: | ---: | ---: |
| baseline | 0,05364 | 0,2997 | 0,5064 |
| gbm_small | 0,04300 | 0,2667 | 0,5390 |
| gbm_large | 0,04602 | 0,2654 | 0,5220 |
| catboost | 0,18402 | 0,3188 | 0,1510 |
| tree_ensemble | 0,17940 | 0,3208 | 0,1737 |
| dummy | 0,5 (stały) | 0 | 0 |

Wybór na validation wskazał `tree_ensemble` przed oceną testu. Jego słabszy wynik
testowy jest ograniczeniem generalizacji w czasie; nie zmieniamy zwycięzcy na podstawie
testu. To wynik eksperymentu, nie decyzja o modelu produkcyjnym z #19.
Baseline S3 był trenowany na 64%, więc nie jest identycznym eksperymentem co baseline S2 na 80%.

Sprawdzenia: 265 testów `pytest`, `ruff check src tests scripts`, `git diff --check`
i `pip check` przeszły. Wykonano rzeczywiste przebiegi na próbce i pełnych danych,
z zapisem/odczytem modeli; końcowy pełny przebieg obejmował etap testowy. Testy regresyjne
sprawdzają znany optymalny próg, remisy, brak wpływu etykiet testowych na wybór/progi,
chronologiczne granice, rzeczywisty zapis/odczyt i dopasowanie generatora tylko na train.

Dokumentacja API: [AutoGluon `fit`](https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.fit.html).
