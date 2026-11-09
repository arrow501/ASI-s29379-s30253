"""Kedro configuration; project.yaml remains the source of team decisions."""

from kedro.config import OmegaConfigLoader

from asi_project.context import ValidatedKedroContext

CONTEXT_CLASS = ValidatedKedroContext
CONFIG_LOADER_CLASS = OmegaConfigLoader
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # A partial environment override must retain the other shared project fields.
    "merge_strategy": {"parameters": "soft"},
}
