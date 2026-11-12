"""Validate effective Kedro settings, including environment and CLI overrides."""

from typing import Any

from kedro.framework.context import KedroContext
from omegaconf import OmegaConf

from asi_project.config import get_contract, load_project_config
from asi_project.config_validation import project_parameters, validate_project_parameters


class ValidatedKedroContext(KedroContext):
    def __init__(self, project_path, config_loader, **kwargs):
        super().__init__(project_path=project_path, config_loader=config_loader, **kwargs)
        # Anchor defaults to the selected session, never to CWD or another checkout.
        config = load_project_config(self.project_path)
        contract = get_contract(config)
        defaults = {"project": project_parameters(config, contract)}
        # The loader has already merged base, environment and runtime overrides.
        # Supply defaults before Kedro performs its own node-parameter validation.
        params = OmegaConf.to_container(
            OmegaConf.merge(defaults, config_loader["parameters"]), resolve=True
        )
        validate_project_parameters(params.get("project"), contract)
        config_loader["parameters"] = params

    @property
    def params(self) -> dict[str, Any]:
        params = super().params
        contract = get_contract(load_project_config(self.project_path))
        validate_project_parameters(params.get("project"), contract)
        return params
