"""Sprawdza konfigurację projektu bez wczytywania danych."""

import sys

import yaml

from asi_project.config import get_contract, load_project_config
from asi_project.data import feature_list_problems


def main() -> int:
    try:
        config = load_project_config()
        contract = get_contract(config)
        problems = feature_list_problems(contract, config.model_features)
        overlap = set(config.numeric_features) & set(config.categorical_features)
        if overlap:
            problems.append(f"Cechy jednocześnie numeryczne i kategoryczne: {sorted(overlap)}.")
        if not 0 < config.test_size < 1:
            problems.append("split.test_size musi być pomiędzy 0 a 1 (bez końców przedziału).")
        if not 0 <= config.random_state < 2**32:
            problems.append("split.random_state musi być w zakresie 0..2**32-1.")
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as error:
        print(f"project.yaml: {error}")
        return 1
    if problems:
        print("\n".join(problems))
        return 1
    print("project.yaml: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
