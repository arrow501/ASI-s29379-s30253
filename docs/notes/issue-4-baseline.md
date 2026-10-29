# Issue #4: notatki do baseline w Sprincie 1

Issue: <https://github.com/arrow501/ASI-s29379-s30253/issues/4>

## Zakres

Uzupełniamy cechy w `project.yaml`, budujemy pipeline z regresją logistyczną,
kończymy `notebooks/02_baseline.ipynb` i zapisujemy metryki z pełnych danych
w `metrics.json`. Hook pre-commit ma sprawdzać konfigurację względem kontraktu
po zmianie któregoś z plików YAML. Kontrakt prowadzącej w
`src/asi_project/datasets.yaml` zostaje bez zmian.

## Ustalenia do pierwszej wersji

- Cechy numeryczne: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`.
- Cechy kategoryczne: `job`, `marital`, `education`, `default`, `housing`, `loan`,
  `contact`, `month`, `poutcome`.
- Pomijamy `duration`, zgodnie z kontraktem dotyczącym momentu predykcji.
- Ostatnie 20% rekordów stanowi zbiór testowy. Zachowujemy kolejność danych.
- W pipeline uzupełniamy braki medianą i skalujemy cechy numeryczne. Dla cech
  kategorycznych używamy najczęstszej wartości i kodowania one-hot.
  Kategorie spoza treningu są obsługiwane przez `handle_unknown="ignore"`.
- Zaczynamy od regresji logistycznej bez strojenia. Porównujemy ją z modelem
  przewidującym najczęstszą klasę na tym samym podziale. Główną metryką jest F1.

Po EDA (issue #3) traktujemy `day` jako liczbę, `unknown` jako osobną kategorię,
a `pdays=-1` zamieniamy w normalizacji na brak danych typu `Int64`.
Imputacja medianą pozostaje wewnątrz pipeline. Informację o braku wcześniejszego
kontaktu zachowuje `previous`: na pełnych danych `previous == 0` odpowiada
dokładnie pierwotnemu `pdays == -1`. Metryki przeliczyliśmy po połączeniu zmian.

## Stan prac i sprawdzenia

EDA i baseline są scalone przez PR #7 i #6, oba po review drugiej osoby.
Na pełnych danych mamy 36 168 rekordów treningowych i 9043 testowe.
Po uwzględnieniu normalizacji z EDA F1 wynosi 0,223807, a recall 0,142857.
DummyClassifier ma F1 równe 0. Niski recall pozostaje ograniczeniem baseline.

Aktualny stan integracji i pozostałe potwierdzenia opisuje
[podsumowanie Sprintu 1](sprint-1-wrap-up.md). Model Card i ADR pozostają
zadaniami kolejnych sprintów.

## Poprawki po review

Dodaliśmy testy braków `pd.NA`, `None` i `np.nan` w kolumnach typu object,
string i category. Sprawdzają trening, predykcję i zachowanie oryginalnych
ramek danych. Pipeline ujednolica te braki przed imputacją.

Loader konfiguracji sprawdza teraz typy przed konwersją. Odrzuca między innymi
ułamkowy lub logiczny seed oraz mapę podaną zamiast listy cech. Nowe testy
najpierw odtworzyły błędy, a po poprawkach przechodzą.

Po usunięciu wyników notebooka przez hook przeszło wszystkich 67 testów,
bez pomijania. Ścisłe automatyczne sprawdzenia Sprintu 1 również przeszły.
Ponowne wykonanie notebooka na pełnych danych dało identyczne metryki.

## Integracja Sprintu 1

Dodaliśmy testy wykrywające brak zamiany `pdays=-1` na brak danych oraz zmianę
kolejności rekordów w filtrowaniu. Łącznie przechodzi 71 testów. Metryki z pełnych
danych zostały odświeżone. Notebook 02 rozwiązuje względny `ASI_METRICS_PATH`
względem katalogu projektu, więc zapis i walidacja próbki CI używają tego samego pliku.
