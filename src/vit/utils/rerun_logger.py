import json
import os
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar, override
from uuid import uuid4

import rerun as rr
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger
from torch import Tensor, nn

# TODO: @rank_zero_only as needed


class RerunLogger(Logger):
    def __init__(
        self,
        root_dir: str | Path,
        name: str | None = "lightning_logs",
        version: int | str | None = None,
        prefix: str = "",
        sub_dir: str | Path | None = None,
    ):
        super().__init__()

        self._root_dir = os.fspath(root_dir)
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._sub_dir = None if sub_dir is None else os.fspath(sub_dir)

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def version(self) -> int | str:
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    @override
    def root_dir(self) -> str:
        return self._root_dir

    @property
    @override
    def log_dir(self) -> str:
        version = (
            self.version if isinstance(self.version, str) else f"version_{self.version}"
        )
        log_dir = os.path.join(self.root_dir, self.name, version)
        if self.sub_dir is not None:
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    @override
    def group_separator(self) -> str:
        return "/"

    @property
    def save_dir(self) -> str:
        return os.path.join(self.root_dir, self.name)

    @property
    def sub_dir(self) -> str | None:
        return self._sub_dir

    @property
    def experiment(self) -> rr.RecordingStream:
        # TODO: need to create directory?
        _recording = getattr(self, "_recording", None)
        if _recording is not None:
            return _recording

        os.makedirs(self.log_dir, exist_ok=True)
        self._recording = rr.RecordingStream("lightning", recording_id=uuid4())
        self._recording.set_log_tick_enabled(True)  # TODO: is this a good idea?
        # TODO: error if log_dir / log.rrd already exists.
        self._recording.set_sinks(
            rr.GrpcServerSink(), rr.FileSink(os.path.join(self.log_dir, "log.rrd"))
        )
        return self._recording

    @override
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        metrics = metrics.copy()
        epoch = metrics.pop("epoch", None)

        if epoch is not None:
            self.experiment.set_time("epoch", sequence=int(epoch))
        if step is not None:
            self.experiment.set_time("step", sequence=step)

        try:
            metrics = self._add_prefix(metrics)
            for k, v in metrics.items():
                if isinstance(v, Tensor):
                    v = v.item()
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        self.experiment.log(
                            f"{k}{self.group_separator}{vk}", rr.Scalers(vv)
                        )
                else:
                    self.experiment.log(k, rr.Scalars(v))
        finally:
            self.experiment.disable_timeline("epoch")
            self.experiment.disable_timeline("step")

    @override
    def log_hyperparams(
        self, params: dict[str, Any] | Namespace, *args: Any, **kwargs: Any
    ) -> None:
        # TODO: flatten instead of serializing to json
        values = vars(params) if isinstance(params, Namespace) else params
        document = json.dumps(values, indent=2, sort_keys=True, default=str)
        self.experiment.log(
            "hyperparams"
            if not self._prefix
            else f"{self._prefix}{self.group_separator}hyperparams",
            rr.TextDocument(document),
            static=True,  # TODO: is it safe to assume this?
        )

    @override
    def log_graph(self, model: nn.Module, input_array: Tensor | None = None) -> None:
        pass

    @override
    def save(self) -> None:
        self.experiment.flush()

    @override
    def finalize(self, status: str) -> None:
        recording: rr.RecordingStream | None = getattr(self, "_recording", None)
        if recording is not None:
            recording.flush()
            recording.disconnect()
            self._recording = None

    def _get_next_version(self) -> int:
        try:
            listdir_info = os.listdir(self.save_dir)
        except OSError:
            return 0

        existing_versions = []
        for listing in listdir_info:
            bn = os.path.basename(listing)
            if os.path.isdir(listing) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    def _add_prefix[T](self, metrics: dict[str, T]) -> dict[str, T]:
        if not self._prefix:
            return metrics
        return {
            f"{self._prefix}{self.group_separator}{k}": v for k, v in metrics.items()
        }
