# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.handler.multi_target import MultiTargetModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class ModelPackager(Pass):
    """Generate an ORT model package with manifest.json and per-component metadata.json.

    This pass takes a MultiTargetModelHandler (produced by EPContextBinaryGenerator with
    a list of provider_options) and generates a model package following the ORT spec:

    - manifest.json at package root with component_models and model_variants
    - metadata.json per component model directory with variant descriptors
    - configs/ directory for genai_config.json and chat_template files

    Variant constraints include:
    - ep (required): execution provider name
    - device (optional): target device type (cpu, gpu, npu)
    - architecture (optional): hardware architecture hint
    - ep_compatibility_info (optional): EP-specific compatibility string
    """

    _accepts_composite_model = True
    _accepts_multi_target_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "model_name": PassConfigParam(
                type_=str,
                default_value=None,
                description="Model name for the manifest. If not set, derived from the output directory name.",
            ),
        }

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        return False

    def _run_for_config(
        self,
        model: MultiTargetModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> MultiTargetModelHandler:
        assert isinstance(model, MultiTargetModelHandler), "ModelPackager requires a MultiTargetModelHandler as input."

        output_dir = Path(output_model_path).with_suffix("")
        output_dir.mkdir(parents=True, exist_ok=True)

        model_name = config.model_name or output_dir.name

        # Copy config files (genai_config.json, chat_template) to configs/
        config_file_names = self._copy_config_files(model, output_dir)

        # Build model_variants dict and copy files into models/<model_name>/
        component_dir = output_dir / "models" / model_name
        component_dir.mkdir(parents=True, exist_ok=True)

        model_variants = {}
        for target_name, target_model in model.get_target_models():
            target_attrs = target_model.model_attributes or {}

            self._copy_target_model(target_name, target_model, component_dir)

            file_path = self._get_relative_model_path(target_name, target_model)

            constraints = {"ep": self.accelerator_spec.execution_provider}
            device = target_attrs.get("device")
            if device:
                constraints["device"] = device
            architecture = target_attrs.get("architecture")
            if architecture:
                constraints["architecture"] = architecture
            ep_compat = target_attrs.get("ep_compatibility_info")
            if not ep_compat:
                ep_compat = self._extract_ep_compatibility_from_onnx(
                    target_model, self.accelerator_spec.execution_provider
                )
            if ep_compat:
                constraints["ep_compatibility_info"] = ep_compat

            model_variants[target_name] = {"file": file_path, "constraints": constraints}

        # Copy base model (pre-context-binary) into base/ subdirectory
        base_model_path = (model.model_attributes or {}).get("base_model_path")
        if base_model_path:
            self._copy_base_model(Path(base_model_path), component_dir, config_file_names)

        # Remove config files from variant directories (they belong in configs/)
        self._remove_config_files(component_dir, config_file_names)

        # Write metadata.json in the component directory
        metadata = {"name": model_name, "model_variants": model_variants}
        metadata_path = component_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Generated metadata at %s", metadata_path)

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
        logger.info("Generated manifest at %s", manifest_path)

        # Update model_attributes
        new_model_attributes = model.model_attributes or {}
        new_model_attributes = {**new_model_attributes, "manifest_path": str(manifest_path)}
        new_model_attributes.pop("additional_files", None)
        new_model_attributes.pop("base_model_path", None)

        return MultiTargetModelHandler(
            [target_model for _, target_model in model.get_target_models()],
            [target_name for target_name, _ in model.get_target_models()],
            model_path=output_dir,
            model_attributes=new_model_attributes,
        )

    @staticmethod
    def _copy_target_model(
        target_name: str,
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
        output_dir: Path,
    ) -> None:
        dest_dir = output_dir / target_name
        if dest_dir.exists():
            return

        if isinstance(target_model, CompositeModelHandler):
            src_dir = Path(target_model.model_path)
        else:
            src_dir = Path(target_model.model_path).parent

        if src_dir.is_dir():
            shutil.copytree(str(src_dir), str(dest_dir))
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(target_model.model_path), str(dest_dir))

    @staticmethod
    def _get_relative_model_path(
        target_name: str,
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
    ) -> str:
        if isinstance(target_model, ONNXModelHandler):
            return f"{target_name}/{Path(target_model.model_path).name}"
        # For CompositeModelHandler or other types, use the directory
        return f"{target_name}/"

    @classmethod
    def _copy_config_files(cls, model, output_dir):
        """Copy non-model files (genai_config, tokenizer, chat_template, etc.) to configs/.

        Collects files and directories from target model ``additional_files`` and copies
        them to the ``configs/`` directory at the package root.  Returns the set of copied
        entry names so they can be removed from the variant directories later.
        """
        config_entries = cls._collect_config_files(model)
        if not config_entries:
            return set()

        configs_dir = output_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)

        for name, src_path in config_entries.items():
            dest = configs_dir / name
            if src_path.is_dir():
                if not dest.exists():
                    shutil.copytree(str(src_path), str(dest))
            else:
                shutil.copy2(str(src_path), str(dest))
            logger.info("Copied %s to %s", name, configs_dir)

        return set(config_entries.keys())

    @classmethod
    def _collect_config_files(cls, model):
        """Find config files from target model additional_files or model directories."""
        config_files: dict[str, Path] = {}

        # Collect from each target model's additional_files
        for _, target_model in model.get_target_models():
            for fp in (target_model.model_attributes or {}).get("additional_files", []):
                p = Path(fp)
                if (p.is_file() or p.is_dir()) and p.name not in config_files:
                    config_files[p.name] = p
            if config_files:
                break

        # Fall back to parent model's additional_files
        if not config_files:
            for fp in (model.model_attributes or {}).get("additional_files", []):
                p = Path(fp)
                if (p.is_file() or p.is_dir()) and p.name not in config_files:
                    config_files[p.name] = p

        return config_files

    @staticmethod
    def _get_model_dir(target_model):
        """Get the directory containing the target model."""
        if isinstance(target_model, CompositeModelHandler):
            return Path(target_model.model_path)
        p = Path(target_model.model_path)
        return p.parent if p.is_file() else p

    @staticmethod
    def _remove_config_files(component_dir, config_file_names):
        """Remove config files and directories from variant subdirectories.

        Skips the ``base/`` directory since base model files are copied separately.
        """
        for name in config_file_names:
            for p in component_dir.rglob(name):
                # Don't remove from base/ — base model is handled by _copy_base_model
                if "base" in p.relative_to(component_dir).parts:
                    continue
                if p.is_dir():
                    shutil.rmtree(str(p))
                    logger.debug("Removed duplicate config directory %s from variant directory", p)
                else:
                    p.unlink()
                    logger.debug("Removed duplicate config file %s from variant directory", p)

    @staticmethod
    def _copy_base_model(base_model_path, component_dir, config_file_names):
        """Copy the pre-optimized base model to the ``base/`` subdirectory.

        Only model files are copied — config files that belong in ``configs/`` are
        skipped.  Recognised model suffixes: ``.onnx``, ``.data``, ``.xml``, ``.bin``.
        """
        base_dir = component_dir / "base"
        if base_dir.exists():
            return

        base_model_path = Path(base_model_path)
        if not base_model_path.is_dir():
            logger.warning("Base model path %s not found, skipping base model copy", base_model_path)
            return

        base_dir.mkdir(parents=True, exist_ok=True)
        model_suffixes = {".onnx", ".data", ".xml", ".bin"}
        for f in sorted(base_model_path.iterdir()):
            if f.is_file() and f.name not in config_file_names and f.suffix in model_suffixes:
                shutil.copy2(str(f), str(base_dir / f.name))
                logger.info("Copied base model file %s to %s", f.name, base_dir)

    @staticmethod
    def _extract_ep_compatibility_from_onnx(
        target_model: Union[ONNXModelHandler, CompositeModelHandler],
        ep: str = "",
    ) -> Optional[str]:
        """Extract ep_compatibility_info from ONNX model custom metadata.

        Looks for metadata keys prefixed with ``ep_compatibility_info.`` in the
        ONNX model file.  If *ep* is given, the entry matching that EP name is
        preferred.  When only a single entry exists it is returned regardless of
        the EP name.
        """
        model_path = None
        if isinstance(target_model, ONNXModelHandler):
            model_path = Path(target_model.model_path)
        elif isinstance(target_model, CompositeModelHandler):
            for component in target_model.model_components:
                if isinstance(component, ONNXModelHandler):
                    model_path = Path(component.model_path)
                    break

        if model_path is None or not model_path.is_file():
            return None

        try:
            import onnx

            onnx_model = onnx.load(str(model_path), load_external_data=False)
            prefix = "ep_compatibility_info."
            ep_compat_map = {
                entry.key[len(prefix) :]: entry.value
                for entry in onnx_model.metadata_props
                if entry.key.startswith(prefix)
            }
        except Exception:
            logger.debug("Could not read ONNX metadata from %s", model_path, exc_info=True)
            return None

        if not ep_compat_map:
            return None
        if ep and ep in ep_compat_map:
            return ep_compat_map[ep]
        if len(ep_compat_map) == 1:
            return next(iter(ep_compat_map.values()))
        return None
