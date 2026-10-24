# Architektura systemu

Ten dokument opisuje architekturę systemu zespołu. W Sprincie 1 powstaje diagram v0, który
będzie aktualizowany w kolejnych sprintach.

## Architektura docelowa (scaffold)

Tak wygląda system po sześciu sprintach — na poziomie warstw. To mapa odpowiedzialności,
a nie diagram do skopiowania: diagram v0 (niżej) rysujecie sami.

Relacje między warstwami (opis tekstowy, nie diagram):

- training: dane → pipeline Kedro → artefakt modelu
- serving: UI → API → model (artefakt wczytany przez API)
- state: API → persistence
- deployment: API + UI → Cloud Run

| Warstwa | Odpowiedzialność | Sprint | Miejsce w repozytorium |
|---|---|---|---|
| Dane i kontrakt | źródło danych, kontrakt, input normalization, training-only filtering | S1 | `data/01_raw/`, `src/asi_project/datasets.yaml`, `src/asi_project/data.py` |
| Pipeline | odtwarzalne przetwarzanie i trening od surowych danych | S2 | `conf/`, `src/asi_project/pipelines/` |
| Eksperymenty i artefakt | porównanie runów, wybór modelu, pakiet modelu ładowalny poza treningiem | S3 | W&B, pakiet modelu, `docs/model_card.md` |
| API | kontrakt usługi, walidacja na granicy, model wczytywany raz | S4 | `src/asi_project/api/`, `docker/` |
| Persistence | zapis predykcji poza procesem aplikacji (tryby `sql` / `disabled`) | S4 → S5 → S6 | repository w API, Postgres w compose |
| UI | formularz jako klient API | S5 | `ui/`, `docker/`, `docker-compose.yml` |
| Chmura | wdrożenie, sekret, logi, release | S6 | `deploy/`, Cloud Run, Secret Manager, Cloud Logging |

## Diagram v0 (Sprint 1)

Narysujcie, jak — według waszej obecnej wiedzy — będzie wyglądał system na koniec projektu.
Diagram v0 nie musi być poprawny w szczegółach; ma pokazać wasze rozumienie przepływu.
Sama lista warstw z sekcji wyżej nie wystarcza — diagram pokazuje wasz system i wasze decyzje.

Na diagramie zaznaczcie co najmniej:

- skąd pochodzą dane i gdzie żyje kontrakt danych;
- gdzie działa każdy z trzech typów przetwarzania: input normalization, training-only
  filtering, model preprocessing;
- gdzie powstaje model i jak trafia tam, gdzie wykonuje się predykcja;
- kto lub co wysyła żądanie predykcji i co dostaje w odpowiedzi;
- gdzie przechowywany jest stan (np. zapis predykcji);
- które elementy już istnieją po Sprincie 1, a które są planowane.

Diagram umieśćcie w bloku `mermaid` w sekcji poniżej (GitHub renderuje Mermaid w Markdown)
albo jako obraz w `docs/` z linkiem w tej sekcji. Usuńcie znaczniki TODO, gdy diagram będzie
gotowy — sprawdza to `python scripts/check_dod.py`.

<!-- DIAGRAM-V0:START -->
```mermaid
flowchart TB
    subgraph DATA["Dane i kontrakt — S1"]
        UCI["UCI Bank Marketing (bank-full)"] --> RAW["bank-full.csv<br/>lokalnie, poza Gitem"]
        CONTRACT["datasets.yaml + project.yaml<br/>target y, ALLOWED/EXCLUDED, time_ordered"]
    end

    subgraph TRAINING["Przygotowanie danych i baseline — S1"]
        NORM["normalize_input()<br/>input normalization"]
        FILTER["filter_training_rows()<br/>tylko dane treningowe"]
        SPLIT["Podział time_ordered<br/>train / test"]
        PREP["Model preprocessing w sklearn Pipeline<br/>imputacja, skalowanie, kodowanie"]
        MODELS["DummyClassifier + LogisticRegression<br/>porównanie F1, predict_proba()"]
        METRICS["metrics.json<br/>metryki i confusion matrix"]
    end

    RAW --> NORM --> FILTER --> SPLIT --> PREP --> MODELS --> METRICS
    CONTRACT -.-> NORM
    CONTRACT -.-> SPLIT
    CONTRACT -.-> PREP

    ARTEFACT["Pakiet modelu — plan S3"]
    MODELS --> ARTEFACT

    subgraph SERVING["Predykcja i stan — plan S4–S5"]
        UI["Streamlit UI<br/>fikcyjne dane klienta"]
        API["API<br/>walidacja + normalize_input()"]
        DB["Persistence<br/>zapis predykcji"]
        UI -->|"żądanie predykcji"| API
        API -->|"P(y=yes) i predykcja"| UI
        API --> DB
    end

    ARTEFACT -->|"model wczytany raz"| API
    CLOUD["Cloud Run, secret i logi — plan S6"]
    API -.-> CLOUD
    UI -.-> CLOUD

    classDef s1 fill:#e8f5e9,stroke:#2e7d32,color:#111;
    classDef planned fill:#f5f5f5,stroke:#616161,stroke-dasharray:5 5,color:#111;
    class UCI,RAW,CONTRACT,NORM,FILTER,SPLIT,PREP,MODELS,METRICS s1;
    class ARTEFACT,UI,API,DB,CLOUD planned;
```

Źródłem danych jest UCI Bank Marketing, a reguły targetu, dozwolonych cech i podziału przechowują kontrakty w repozytorium. Input normalization jest wspólne dla treningu i przyszłego API, training-only filtering dotyczy wyłącznie treningu, a preprocessing wymagający `fit()` należy do sklearn `Pipeline`. W Sprincie 1 porównamy `DummyClassifier` i `LogisticRegression` na podziale `time_ordered`, zapisując metryki dla klasy `yes`. Zielone elementy będą istnieć po Sprincie 1, a szare są planowane: pakiet modelu trafi do API, które zwróci do UI wartość `P(y=yes)` i zapisze predykcję w warstwie persistence. Docelowo API i UI zostaną wdrożone w Cloud Run z osobnym sekretem i logami.
<!-- DIAGRAM-V0:END -->

## Historia zmian architektury

| Sprint | Zmiana | Powiązany ADR |
|---|---|---|
| S1 | diagram v0 | — |
