# Sprint 3 — start W&B

Owner trackingu: Arrow; review: Anton (#18). W&B: `wandb==0.30.0`, Python 3.11,
środowisko `asi-ml`. To fundament trackingu baseline'u; eksperymenty AutoGluona
i pakiet W&B Artifact pozostają kolejnymi pracami.

## Konto i projekt zespołu

Obie osoby potrzebują własnego konta i dostępu do tego samego projektu.
`entity` to identyfikator właściciela/zespołu z adresu URL, a nie nazwa wyświetlana konta.
Projekt wskazujemy jawnie: `https://wandb.ai/<entity>/<project>`.
Nie zakładamy, że poprawny login oznacza uprawnienia do projektu drugiej osoby.

```bash
conda activate asi-ml
python -m pip install wandb==0.30.0 -c constraints.txt
python -m pip check
wandb login
```

Każdy loguje się we własnym terminalu. Klucz API nie trafia do repozytorium ani rozmowy.
Nie jest potrzebny do testów CI. Pipeline nie odczytuje automatycznie pliku `.env`.

## Uruchomienia

Zwykły `kedro run` nadal działa bez W&B. Tracking jest osobnym pipeline'em,
korzystającym z tego samego przygotowania, splitu, treningu i ewaluacji S2.

Sprawdzenie połączenia pipeline'u bez SDK i bez sieci:

```bash
kedro run --pipeline baseline_tracking --env ci
```

Rzeczywisty lokalny zapis W&B na próbce, bez logowania:

```bash
kedro run --pipeline baseline_tracking --env ci --params tracking.mode=offline
```

`offline` zapisuje run w ignorowanym katalogu `wandb/`; nie potwierdza dostępu do serwera.
Nie synchronizujemy próbnych runów jako baseline'u pełnych danych.

Po utworzeniu wspólnego projektu i zalogowaniu, baseline na pełnych danych:

```bash
kedro run --pipeline baseline_tracking --params tracking.mode=online,tracking.entity=TEAM_SLUG,tracking.project=PROJECT_SLUG
```

Zastąpić `TEAM_SLUG` i `PROJECT_SLUG` wartościami z adresu projektu. Parametry można
zapisać w ignorowanym `conf/local/parameters.yml`. `WANDB_MODE` ma pierwszeństwo nad
`tracking.mode`, więc `WANDB_MODE=disabled` wyłącza tracking również przy konfiguracji online.
Przed odbiorem online sprawdzić, czy zmienna nie pozostała ustawiona po lokalnych testach.

## Kontrakt runu

- Config: efektywne parametry projektu i treningu, split/liczebności, `data_mode`,
  Git SHA, informacja o niezacommitowanych zmianach i wersje Python/pandas/sklearn/W&B.
- `data_fingerprint`: SHA-256 reprezentacji CSV **wczytanego `raw_data`**, z nagłówkami,
  indeksem, kolejnością rekordów i końcami linii LF (`loaded-data-csv-v1`).
  To inny fingerprint niż SHA oryginalnego pliku w `fingerprint.json`; identyfikuje
  faktyczne wejście catalogu, w tym próbkę CI. Nie polegamy na potencjalnie starym sidecarze.
- Metryki: `dummy/*` i `logistic_regression/*`, w tym F1 i liczności macierzy pomyłek.
- Nie wysyłamy tabeli rekordów, modelu ani snapshotu kodu. Pakiet modelu należy do #20.

## Odbiór i diagnoza

Online jest odebrane dopiero, gdy po zakończeniu polecenia run jest widoczny pod właściwym
`entity/project`, ma config i metryki zgodne z `data/08_reporting/metrics.json`, a Anton
potwierdzi dostęp. Baseline powinien odtworzyć `metrics.json` z S1/S2.

- Brak SDK: instalować w `asi-ml`, z którego uruchamiane jest Kedro.
- Brak klucza / zły login: wykonać `wandb login` lub `wandb login --relogin` we własnym terminalu.
- Odmowa dostępu / brak projektu: sprawdzić URL, slug zespołu i członkostwo obu osób.
- Timeout: sprawdzić sieć/proxy i dostęp do W&B; pipeline zgłasza błąd, nie zmienia
  automatycznie online na offline. Nie traktować lokalnych plików jako dowodu publikacji.
- `disabled` działa bez importowania SDK, `offline` wymaga SDK, `online` dodatkowo
  wymaga konta, sieci i jawnego wskazania projektu.

Zachowanie trybów i logowania: [SDK W&B](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_init.py),
[weryfikacja loginu](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_login.py).
