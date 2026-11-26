# S3-4 — odbiór pakietu production i przekazanie do Sprintu 4

Owner: Anton (`sanderJa`, s30253). Review: Arrow (`arrow501`, s29379).

Wybrany model to chronologiczny baseline opisany w [ADR-2](../adr/0002-wybor-modelu.md)
i [Model Card](../model_card.md). Źródłem zatwierdzonego aliasu i konkretnej wersji
jest [potwierdzenie publikacji](results/sprint-3-production.json).

## Co sprawdza smoke test

```text
W&B production → pobranie do pustego katalogu → nowy interpreter
→ wspólny loader → 5 rekordów sample.csv bez targetu
→ wspólna predykcja → kontrola kontraktu → raport JSON
```

Skrypt sprawdza liczbę i kolejność wierszy, indeks, dokładne kolumny wyniku,
etykiety `no`/`yes`, klasę pozytywną, skończone prawdopodobieństwa w `[0, 1]`
oraz zastosowanie progu z metadanych. Odczyt i predykcja nie mogą wywołać `fit`.
Nowy proces działa z `WANDB_MODE=disabled`, zablokowanym SDK W&B i bez przekazywania
`WANDB_API_KEY`. Model jest wczytywany wyłącznie ze wskazanego pakietu.

Tryb online korzysta z `download_model_package`. Zapisuje konkretną wersję,
pochodzenie modelu, identyfikator procesu, pięć wyników, SHA checkoutu i hash skryptu.
Błąd pobrania, inna wersja niż `--expected-ref` albo błędna predykcja kończą
komendę niezerowym kodem; nie powstaje raport sukcesu ani lokalny model zastępczy.
Katalog pobrania jest tymczasowy i jest sprzątany po próbie.

## Niezależny odbiór lokalny — PowerShell

Wymagane są Git, conda i dostęp do opublikowanego projektu W&B. Klucza nie zapisuj
w repozytorium. Poniższa procedura tworzy nowy checkout i środowisko, bez kopiowania
modeli, pełnych danych ani obiektów z procesu treningowego. Korzysta z próbki w repo.

Przed scaleniem S3-4 ustaw `$ref` na opublikowaną gałąź PR, np. `feat/21-ci-smoke`.
Po scaleniu użyj `main`. Jeśli środowisko `asi-s3-smoke` już istnieje, wybierz nową
nazwę w obu poleceniach conda.

```powershell
$ref = "main"
$smokeRoot = Join-Path $env:TEMP ("asi-s3-smoke-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
git clone --single-branch --branch $ref https://github.com/arrow501/ASI-s29379-s30253.git "$smokeRoot/repo"
if ($LASTEXITCODE -ne 0) { throw "Nie udalo sie sklonowac galezi." }
Set-Location "$smokeRoot/repo"

conda env create -n asi-s3-smoke -f environment.yml
if ($LASTEXITCODE -ne 0) { throw "Nie udalo sie utworzyc srodowiska." }
conda activate asi-s3-smoke
python -m pip install --no-deps -e .
python -m pip install -r requirements/preflight-s2.txt -r requirements/preflight-s3.txt -r requirements/benchmark-s3.txt -c constraints.txt
if ($LASTEXITCODE -ne 0) { throw "Instalacja zaleznosci nie powiodla sie." }

$env:KEDRO_DISABLE_TELEMETRY = "1"
$env:WANDB_MODE = "online"
wandb login
if ($LASTEXITCODE -ne 0) { throw "Logowanie W&B nie powiodlo sie." }
$release = Get-Content docs/notes/results/sprint-3-production.json -Raw | ConvertFrom-Json
$report = Join-Path $smokeRoot "production-smoke.json"
python scripts/smoke_model.py --artifact $release.requested_ref --expected-ref $release.artifact_ref --output $report
if ($LASTEXITCODE -ne 0) { throw "Smoke test production nie przeszedl." }
Get-Content $report

$env:WANDB_MODE = "disabled"
pytest -q
pre-commit run --all-files --show-diff-on-failure
kedro run --env ci
```

Każda kontrola powinna zakończyć się kodem 0. Gdy pre-commit poprawi pliki,
przejrzyj zmiany i uruchom kontrolę ponownie. Raport smoke zachowaj jako dowód
odbioru, z rzeczywistą datą, SHA sprawdzanej wersji i pełnym `artifact_ref`.
Samo istnienie skryptu lub test z atrapą SDK nie oznacza ukończonego odbioru online.

## CI bez sekretów

`.github/workflows/ci.yml` działa na Linuxie, Windowsie i macOS.
Zachowuje globalne `WANDB_MODE=disabled`. Oprócz pytest i domyślnego pipeline'u
uruchamia `experiments_test` na próbce z `experiments.autogluon_enabled=false`.
Baseline i Dummy są rzeczywiście trenowane, zapisywane do pakietów i odczytywane.
Następnie `smoke_model.py --package` sprawdza nowy pakiet baseline'u w nowym procesie.
Nie wymaga to klucza W&B ani pełnych danych. Test AutoGluona ma osobny przełącznik
`pytest -q tests/test_model_package_autogluon.py --run-autogluon-package`.

## Osobny workflow online

`.github/workflows/model-smoke.yml` ma job `production-smoke`. Uruchamia się po
pushu do `main` oraz ręcznie przez **Actions → Production model smoke → Run workflow**.
Wykonuje świeży checkout, instaluje środowisko i pobiera rzeczywisty alias W&B.
Wersja musi odpowiadać `artifact_ref` z zatwierdzonego potwierdzenia publikacji.

Przed pierwszym uruchomieniem dodaj w repozytorium:

**Settings → Secrets and variables → Actions → New repository secret**

- nazwa: `WANDB_API_KEY`;
- wartość: klucz konta mającego dostęp do wskazanego artefaktu.

Brak sekretu albo niedostępność W&B oznaczają błąd joba. Zwykły workflow CI nadal
sprawdza lokalny pakiet bez sieci i sekretów. Dostęp W&B jest tymczasowy, zgodnie
z [notatką o trackingu](sprint-3-wandb.md).

Raport jest dostępny w artefakcie Actions **production-model-smoke** przez 14 dni.
Przenieś zweryfikowany JSON do `docs/notes/results/` i dopisz link do przebiegu
oraz wynik niezależnego odbioru. Zielony zwykły CI nie zastępuje zielonego joba online.

## Przekazanie do S4

API korzysta z `load_model_package` i `predict_model_package`. Wejściem jest
DataFrame z cechami z metadanych, bez wymaganego targetu. Wynik zawiera
`prediction` i `probability_positive`, w kolejności wejścia. Klasa pozytywna to
`yes`; próg jest odczytywany z pakietu, a nie ustawiany na domyślne `0.5`.
Kontrakt szczegółowy jest w [`src/asi_project/README.md`](../../src/asi_project/README.md).

## Stan odbioru

- Próba deweloperska online 26.11.2026: pobrano rzeczywisty `production` wskazujący
  na `bank-marketing-model:v0`. Nowy interpreter wykonał pięć predykcji baseline'u
  bez targetu i bez `fit`. [Raport JSON](results/sprint-3-smoke-development.json)
  zawiera model, wersję, pochodzenie, wyniki i hash użytego skryptu.
  Próba korzystała z istniejącego środowiska conda i kopii roboczej bazującej na
  `cf8a929` z niezacommitowaną implementacją S3-4 (`git_dirty=true`). To potwierdzenie
  działania kodu online; końcowy odbiór instrukcji wymaga opublikowanej wersji S3-4.
- Testy lokalne: **529 passed, 1 skipped**, w tym rzeczywisty AutoGluon. Pominięto
  wyłącznie test tworzenia symlinków, niedostępny w tym Windows.
- Wszystkie hooki pre-commit przeszły dla plików repozytorium i nowych plików S3-4.
  `kedro run --env ci`, `experiments_test` z wyłączonym AutoGluonem i lokalny smoke
  nowego pakietu baseline'u zakończyły się kodem 0.
- Świeży checkout po publikacji gałęzi S3-4: do potwierdzenia przez Antona powyższą procedurą.
- Review Arrow i zielone workflow na końcowym `main`: do potwierdzenia po publikacji/scaleniu.
