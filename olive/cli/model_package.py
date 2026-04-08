# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path

from olive.cli.base import BaseOliveCLICommand, add_logging_options, add_telemetry_options
from olive.common.utils import hardlink_copy_dir
from olive.telemetry import action

logger = logging.getLogger(__name__)


@action
class ModelPackageCommand(BaseOliveCLICommand):
    """Merge multiple single-target context binary outputs into a multi-target package with manifest.json."""

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "model-package",
            help="Merge multiple context binary outputs into a multi-target package with manifest.json",
        )

        sub_parser.add_argument(
            "-s",
            "--source",
            type=str,
            action="append",
            required=True,
            help=("Source context binary output directory. Can be specified multiple times. "),
        )

        sub_parser.add_argument(
            "-o",
            "--output_path",
            type=str,
            required=True,
            help="Output directory for the merged multi-target package.",
        )

        sub_parser.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="Model name for the manifest. If not set, derived from the output directory name.",
        )

        add_logging_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=ModelPackageCommand)

    def run(self):
        sources = self._parse_sources()
        output_dir = Path(self.args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = self.args.model_name or output_dir.name

        # Copy config files (genai_config.json, chat_template) to configs/
        config_file_names = self._copy_config_files(sources, output_dir)

        # Create component model directory under models/
        component_dir = output_dir / "models" / model_name
        component_dir.mkdir(parents=True, exist_ok=True)

        model_variants = {}
        for target_name, source_path in sources:
            model_config = self._read_model_config(source_path)
            model_attrs = model_config.get("config", {}).get("model_attributes") or {}

            # Copy source directory into component_dir/{target_name}/
            target_dir = component_dir / target_name
            hardlink_copy_dir(source_path, target_dir)

            constraints = {}
            for key in ("ep", "device", "architecture", "ep_compatibility_info"):
                if model_attrs.get(key) is not None:
                    constraints[key] = model_attrs[key]

            # Fall back to ONNX model metadata for ep_compatibility_info
            if "ep_compatibility_info" not in constraints:
                ep_compat = self._extract_ep_compatibility_from_onnx(source_path, constraints.get("ep", ""))
                if ep_compat:
                    constraints["ep_compatibility_info"] = ep_compat

            model_variants[target_name] = {
                "file": model_config.get("config", {}).get("model_path", f"{target_name}/"),
                "constraints": constraints,
            }

        # Write metadata.json in component directory
        metadata = {"name": model_name, "model_variants": model_variants}
        with open(component_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Remove config files from variant directories (they belong in configs/)
        for name in config_file_names:
            for p in component_dir.rglob(name):
                p.unlink()

        # Write manifest.json at package root
        manifest = {
            "name": model_name,
            "component_models": {
                model_name: {"model_variants": model_variants},
            },
        }
        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Merged {len(sources)} targets into {output_dir}")
        print(f"Manifest written to {manifest_path}")

    def _parse_sources(self) -> list[tuple[str, Path]]:
        sources = []
        for source in self.args.source:
            path = Path(source)
            if not path.is_dir():
                raise ValueError(f"Source path does not exist or is not a directory: {path}")

            if not (path / "model_config.json").exists():
                raise ValueError(
                    f"No model_config.json found in {path}. "
                    "Source must be an Olive output directory with model_config.json."
                )

            sources.append((path.name, path))

        if len(sources) < 2:
            raise ValueError("At least two --source directories are required to merge.")

        return sources

    @staticmethod
    def _read_model_config(source_path: Path) -> dict:
        """Read and return model_config.json from a source directory."""
        config_path = source_path / "model_config.json"
        with open(config_path) as f:
            return json.load(f)

    @staticmethod
    def _copy_config_files(sources, output_dir):
        """Copy non-model files (genai_config, tokenizer, chat_template, etc.) to configs/.

        Collects files listed in ``additional_files`` from the first source's
        model_config.json and copies them to ``configs/`` at the package root.
        Returns the set of copied file names so they can be removed from variant dirs.
        """
        config_files: dict[str, Path] = {}

        for _, source_path in sources:
            model_config = ModelPackageCommand._read_model_config(source_path)
            additional_files = model_config.get("config", {}).get("model_attributes", {}).get("additional_files", [])
            for fp in additional_files:
                p = Path(fp)
                # The file may be in the original cache dir; also check source_path
                if not p.is_file():
                    p = source_path / p.name
                if p.is_file() and p.name not in config_files:
                    config_files[p.name] = p
            if config_files:
                break

        if not config_files:
            return set()

        configs_dir = output_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        for name, src_path in config_files.items():
            shutil.copy2(str(src_path), str(configs_dir / name))

        return set(config_files.keys())

    @staticmethod
    def _extract_ep_compatibility_from_onnx(source_path: Path, ep: str = "") -> "str | None":
        """Extract ep_compatibility_info from ONNX model files in *source_path*.

        Looks for metadata keys prefixed with ``ep_compatibility_info.`` in the
        first ``.onnx`` file found in the directory.
        """
        onnx_files = sorted(source_path.glob("*.onnx"))
        if not onnx_files:
            return None

        try:
            import onnx

            onnx_model = onnx.load(str(onnx_files[0]), load_external_data=False)
            prefix = "ep_compatibility_info."
            ep_compat_map = {
                entry.key[len(prefix) :]: entry.value
                for entry in onnx_model.metadata_props
                if entry.key.startswith(prefix)
            }
        except Exception:
            logger.debug("Could not read ONNX metadata from %s", onnx_files[0], exc_info=True)
            return None

        if not ep_compat_map:
            return None
        if ep and ep in ep_compat_map:
            return ep_compat_map[ep]
        if len(ep_compat_map) == 1:
            return next(iter(ep_compat_map.values()))
        return None
