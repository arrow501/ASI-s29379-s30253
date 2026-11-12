"""Sprawdza konfigurację projektu bez wczytywania danych."""

import sys

import yaml

from asi_project.config import get_contract, load_project_config
from asi_project.config_validation import project_parameters, validate_project_parameters


def main() -> int:
    try:
        config = load_project_config()
        contract = get_contract(config)
        # Keep CLI checks and Kedro's resolved-parameter checks on the same rules,
        # including settings supplied through environment or runtime overrides.
        validate_project_parameters(project_parameters(config, contract), contract)
    except (OSError, ValueError, TypeError, AttributeError, yaml.YAMLError) as error:
        print(f"project.yaml: {error}")
        return 1
    print("project.yaml: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
