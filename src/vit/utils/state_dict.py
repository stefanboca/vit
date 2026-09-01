from typing import cast

from torch import nn


class StateDictExcludeModulesMixin:
    def _exclude_modules_from_state_dict(self, *module_paths: str) -> None:
        """Exclude modules from saved state dicts and strict-load requirements."""
        self._state_dict_excluded_module_paths = tuple(
            path.strip(".") for path in module_paths
        )

        module = cast(nn.Module, self)
        module.register_load_state_dict_pre_hook(self._capture_load_state_dict_prefix)
        module.register_load_state_dict_post_hook(self._remove_excluded_missing_keys)
        module._register_state_dict_hook(self._remove_excluded_state_dict_keys)

    def _capture_load_state_dict_prefix(
        self,
        _module,
        _state_dict,
        prefix,
        _local_metadata,
        _strict,
        _missing_keys,
        _unexpected_keys,
        _error_msgs,
    ) -> None:
        self._load_state_dict_prefix = prefix

    def _remove_excluded_missing_keys(self, _module, incompatible_keys) -> None:
        prefixes = self._excluded_state_dict_prefixes(self._load_state_dict_prefix)
        incompatible_keys.missing_keys[:] = [
            key
            for key in incompatible_keys.missing_keys
            if not key.startswith(prefixes)
        ]

    def _remove_excluded_state_dict_keys(
        self, _module, state_dict, prefix, _local_metadata
    ) -> None:
        prefixes = self._excluded_state_dict_prefixes(prefix)
        for key in list(state_dict):
            if key.startswith(prefixes):
                del state_dict[key]

    def _excluded_state_dict_prefixes(self, prefix: str) -> tuple[str, ...]:
        return tuple(
            f"{prefix}{path}." for path in self._state_dict_excluded_module_paths
        )
