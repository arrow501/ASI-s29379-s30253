# Sprint 3 — tracking W&B (#18)

Owner: Arrow; review: Anton. W&B jest opcjonalnym odbiorcą gotowych wyników lokalnych.
Nie bierze udziału w splicie, treningu, wyborze modelu ani progu. Domyślny tryb
`disabled` nie importuje SDK i nie wymaga konta; domyślny `kedro run` nadal odtwarza S2.

## Tymczasowy dostęp i dostępność usługi

Obecnie korzystamy z tymczasowego dostępu przekazanego przez prowadzącą. To obejście
zgłaszanych problemów z zakładaniem nowych kont po zmianach właścicielskich W&B,
a nie docelowe konto zespołu. Nie zakładamy bezterminowej ważności tych poświadczeń
ani dostępności obecnego projektu. Po uzyskaniu własnego dostępu podmienimy logowanie,
`WANDB_ENTITY` i `WANDB_PROJECT` oraz ewentualny sekret GitHub Actions. Obecne linki
zachowują znaczenie jako historyczny zapis odbioru; lokalne raporty pozostają dostępne.

Po stronie W&B mogą wystąpić błędy usługi, timeouty oraz opóźnienia wysyłki lub
indeksowania metryk i artefaktów. `completed` w `wandb.json` oznacza, że SDK zakończyło
zapis bez zgłoszenia wyjątku; nie gwarantuje natychmiastowej widoczności wszystkiego
w UI/API. Przy braku metryk sprawdzamy lokalny wynik, log synchronizacji i ponawiamy
odczyt po chwili. Błędy autoryzacji mogą też wynikać z wygaśnięcia tymczasowego dostępu.
Nie przypisujemy automatycznie każdego błędu dostawcy — sprawdzamy również konfigurację.

Awaria W&B nie unieważnia poprawnego treningu. W razie niedostępności usługi można
kontynuować z `WANDB_MODE=disabled` albo zapisać run SDK w `offline` i wysłać go później.
Nie zmieniamy modelu, progu ani wyników tylko dlatego, że interfejs W&B ich jeszcze nie pokazuje.

## Konfiguracja i logowanie

W `conf/base/parameters_tracking.yml` są tylko tryb, entity i projekt:
`wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253`.
`WANDB_MODE`, `WANDB_ENTITY` i `WANDB_PROJECT` nadpisują te wartości. Inne pola,
w tym klucz API w parametrach, są odrzucane. Online/offline wymaga jawnego entity
oraz projektu, żeby nie zapisywać runów w przypadkowym prywatnym projekcie.

```bash
conda activate asi-ml
python -m pip install -r requirements/preflight-s3.txt -c constraints.txt
wandb login
```

Klucz wpisujemy interaktywnie; W&B zapisuje go lokalnie poza repo (`~/.netrc`).
Można też podać `WANDB_API_KEY` w środowisku procesu. Nie umieszczamy wartości klucza
w komendzie, kodzie, konfiguracji Kedro, notebooku ani opisie PR.
Sekret GitHub Actions nie jest dostępny do odczytu lokalnie. Zwykłe CI ma
`WANDB_MODE=disabled`, nie dostaje klucza i niczego nie publikuje.

## Pipeline i pierwsza referencja

Pierwszy run projektu odtwarza S2: chronologiczne 80/20, seed 42, bez validation
ani strojenia, domyślny próg 0,5. Sklearn Pipeline i parametry pozostają te same.
Run zawiera baseline oraz DummyClassifier, żeby dało się porównać obie referencje.

```bash
# Składnia env działa w Bash i fish.
env WANDB_MODE=online KEDRO_DISABLE_TELEMETRY=1 kedro run --pipelines baseline_tracked

# Każdy eksperyment: lokalne raporty → freeze → test → publikacja W&B.
env WANDB_MODE=online KEDRO_DISABLE_TELEMETRY=1 kedro run --pipelines experiments_test
env WANDB_MODE=online KEDRO_DISABLE_TELEMETRY=1 kedro run --pipelines experiments_bagged_test
env WANDB_MODE=online KEDRO_DISABLE_TELEMETRY=1 kedro run --pipelines experiments_crossfold_test
```

PowerShell: ustaw `$env:WANDB_MODE="online"` i `$env:KEDRO_DISABLE_TELEMETRY="1"`,
następnie uruchom samo `kedro run ...`. Na próbce użyj `--env ci`; AutoGluon jest tam
wyłączony, a raport ma `data_mode=sample`. Nie jest to pomiar jakości na pełnych danych.
Bez `_test` publikujemy tylko wyniki validation, nawet gdy katalog zawiera starszy test.

Każdy kandydat eksperymentu ma osobny run W&B, grupowany według lokalnego ID.
Wysyłamy efektywny config (cechy, split, seed, target i klasa pozytywna, konfiguracje
AutoGluona, budżet fit i CPU), SHA/dirty, fingerprint, wersje, metryki validation,
zamrożony próg i opcjonalnie metryki testowe. Limit `time_limit_seconds` dotyczy `fit`
AutoGluona; baseline nie ma sztucznego limitu. Dokładne zasady refit i early stopping:
[sprint-3-local-experiments.md](sprint-3-local-experiments.md).

Artefakt typu `experiment-report` zawiera JSON-y konfiguracji, pochodzenia, splitu/foldów,
metryk i decyzji; przy `_test` także `test_metrics.json`. Nie wysyłamy danych wejściowych,
modeli ani kodu przez adapter trackingu. SDK może dodawać własny artefakt `wandb-events`.
Pakiet modelu publikujemy osobno jako artefakt typu `model`:
[komendy #20](../../src/asi_project/README.md#komendy-obsługi-pakietu).

## Offline, awarie i ponowienie

`WANDB_MODE=offline` używa tego samego adaptera i zapisuje binarne runy SDK lokalnie
w `<run_dir>/wandb/`. Konto ani dostęp do sieci nie są wymagane. Późniejsza synchronizacja:
`wandb sync <ścieżka-do-offline-run>`. Nie mylić z `disabled`, który całkowicie pomija SDK.

Błąd W&B przerywa node trackingowy. Raporty i modele lokalne pozostają gotowe, a
`wandb.json` ma status `failed`, typ wyjątku i dotychczasowe linki. Nie zmieniamy statusu
udanego treningu na nieudany i nie uruchamiamy automatycznie kolejnego treningu.
Wysłanie istniejących wyników eksperymentu (przykładowa ścieżka, podstaw własny run):

```bash
env WANDB_MODE=online python -m asi_project.tracking \
  --run data/08_reporting/experiments/20261123T232543-holdout-6bf251c7 \
  --include-test
```

Replay CLI obsługuje eksperymenty holdout/bagged/chronological. Nie obsługuje katalogów
`s2_baseline`; ponowne `baseline_tracked` wykonuje tani baseline i tworzy nowy run referencyjny.
Komenda odczytuje konfigurację i provenance z runu, nie zastępuje ich bieżącym SHA.
Sprawdza komplet kandydatów, zamrożoną selekcję, zgodność progów i finalnego testu
oraz skończoność metryk. ID W&B jest deterministyczne dla lokalnego runu i kandydata;
ponowienie online używa `resume=allow`, może dopisać historię i wersję raportu, ale
nie tworzy nowego ID. Po offline nie należy równolegle wysyłać tych samych wyników
dwoma sposobami. `wandb.json` zapisuje wybrany tryb, entity/projekt i linki po zakończeniu SDK.

## Odbiór

Testy sprawdzają treść payloadów i prawdziwe JSON-y, nie tylko wywołanie klienta:
metryki i macierz pomyłek, progi, fingerprint, SHA, parametry i brak klucza w konfiguracji.
Obejmują uszkodzone źródła, błędy SDK, ponowienie z tym samym ID i zachowanie lokalnych
wyników. Przy zainstalowanym SDK wykonujemy też rzeczywisty zapis offline.
W sandboxie test SDK wymaga zgody na lokalny socket i katalog cache; bez SDK ten
jeden test jest pomijany, natomiast testy `disabled` i payloadów nadal działają.

## Rzeczywiste uruchomienia — 24.11.2026

Odczyt z API potwierdził **17 zakończonych runów**: pierwszą referencję S2, 6 kandydatów
nowego holdout i 10 kandydatów z istniejących runów bagging/chronological. Wszystkie
metryki, progi, config i provenance zgadzają się z lokalnymi JSON-ami. Manifesty
artefaktów są kompletne. Raporty S2 i GBM small z każdego protokołu pobrano i porównano
z lokalną zawartością JSON. Szczegóły: [rejestr sprawdzeń](sprint-3-wandb-runs.json).

W&B może udostępnić część summary z opóźnieniem po zakończeniu runu. Sprawdzenie API
wykonano ponownie po indeksowaniu; nie zmieniano brakujących chwilowo metryk lokalnych.

Referencja S2 ma identyczny wynik jak zatwierdzony `metrics.json`: F1 **0,2238069116840373**,
TN/FP/FN/TP **5805 / 382 / 2448 / 408**. Nowy holdout odtworzył dotychczasowe F1:
baseline 0,5064; GBM small 0,5390; GBM large 0,5220; CatBoost 0,1510; ensemble 0,1737.
Wyboru `tree_ensemble` i progów nie zmieniano po teście.

Świeży holdout: `20261124T161322-holdout-b410b77e`. Replay zachowuje źródła
`20261123T232543-bagged-c7e1b345` i `20261123T232543-chronological-3d2c20c4`;
nie jest nowym treningiem ani zmianą oryginalnego SHA. Wszystkie te runy mają
`git_dirty=true` w provenance; sam zapisany SHA nie odtwarza niezacommitowanego kodu.

| Protokół | Kandydat | Run W&B |
| --- | --- | --- |
| S2 | s2_baseline | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/db312c065316ac07e52a) |
| holdout | baseline | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/206aff6fc467a9667f79) |
| holdout | catboost | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/410e5fb5b84363305cb0) |
| holdout | dummy | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/7ccc579043d9413f93bc) |
| holdout | gbm_large | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/4a3e554d38e7995f75d4) |
| holdout | gbm_small | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/c355b76ada19c9d2fd47) |
| holdout | tree_ensemble | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/a6e7f424834222038bca) |
| bagged | baseline | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/c2fb5ef969486935cd00) |
| bagged | catboost | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/4474be9f1a3faff7e0f1) |
| bagged | gbm_large | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/3302a90ddafe0d90454c) |
| bagged | gbm_small | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/5536c33748089b85ce07) |
| bagged | tree_ensemble | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/99cc2500407db9cca170) |
| chronological | baseline | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/b4d90586db1e9551327d) |
| chronological | catboost | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/3191cb2e35a0947d4ae4) |
| chronological | gbm_large | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/0bc30bd94211b4991361) |
| chronological | gbm_small | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/55008bdeb4eb197026bb) |
| chronological | tree_ensemble | [run](https://wandb.ai/wnukdominika-polish-japanese-academy-of-information-tec/ASI-s29379-s30253/runs/fa90ce6c2500d6447cb3) |

Sprawdzenia lokalne i w czystym checkoutcie `b6ecf55`: **380 testów, 0 failures**,
w tym 20 nowych dla trackingu; pre-commit i kontrola diffu przeszły. To wynik odbioru #18.
Po scaleniu wykonano także [nowy trening online z pięcioma konfiguracjami](sprint-3-online-integration.md)
oraz [smoke production na main i w nowym środowisku](sprint-3-closeout.md).
