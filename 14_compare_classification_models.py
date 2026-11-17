from __future__ import annotations

"""Train and compare MedViTV2, MedMamba, EVA-X, CXFormer-S, and our cascade.

The script keeps the patient-level split in ``data/manifest.csv`` fixed for all
five methods.  It reports the first experiment's five prespecified metrics on
clean images and on five degradations at three severity levels.

Official source snapshots used by the loader (verified 2026-10-27):
  MedViTV2  https://github.com/Omid-Nejati/MedViTV2
  MedMamba  https://github.com/YubiaoYue/MedMamba
  EVA-X     https://github.com/hustvl/EVA-X
  CXFormer  https://github.com/m42-health/CXformer

Typical paper run (all workflow options below are the defaults):
  python 14_compare_classification_models.py

Quick pipeline check:
  python 14_compare_classification_models.py --stage evaluate \
      --models ours --conditions clean,noise:heavy --max-samples 8

MedViTV2 needs ``natten``.  MedMamba uses the official ``mamba_ssm`` CUDA
extension when it is available, otherwise this script uses a PyTorch 2.11
CUDA associative-scan implementation with a custom backward pass.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
import time
import types
import urllib.request
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import project_config as project
from cxr_cascade.calibration import TemperatureCalibrator, fit_temperature, sigmoid
from cxr_cascade.config import AppConfig, save_config_module
from cxr_cascade.data import (
    assert_no_patient_leakage,
    build_presplit_manifest,
    read_manifest,
    write_manifest,
)
from cxr_cascade.degradations import LEVELS, apply_degradation
from cxr_cascade.inference import CascadeInference, ModelRunner
from cxr_cascade.metrics import auroc
from cxr_cascade.prediction import collect_logits
from cxr_cascade.quality import QualityModel, raw_quality_metrics
from cxr_cascade.training import TrainConfig, train_model
from cxr_cascade.tuning import tune_router


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = PROJECT_ROOT.parent / ".reference_repos"
DEFAULT_OUTPUT_ROOT = project.OUTPUT_ROOT / "classification_comparison"

MODEL_ORDER = ("ours", "medvitv2", "medmamba", "eva_x", "cxformer_s")
DEGRADATIONS = tuple(LEVELS)
SEVERITIES = ("light", "medium", "heavy")

REPOSITORIES = {
    "medvitv2": {
        "url": "https://github.com/IMPACT-L/MedViTV2.git",
        "directory": "MedViTV2",
        "commit": "66f46fdf7aef48639de5255a64a9284e581ebef5",
        "marker": "MedViT.py",
    },
    "medmamba": {
        "url": "https://github.com/YubiaoYue/MedMamba.git",
        "directory": "MedMamba",
        "commit": "104beb3d641d67cef6f09b94ac299083d91bc417",
        "marker": "MedMamba.py",
    },
    "eva_x": {
        "url": "https://github.com/hustvl/EVA-X.git",
        "directory": "EVA-X",
        "commit": "eedd642c85339b25682ceb574b644cb9afe88e62",
        "marker": "eva_x.py",
    },
}

EVA_X_TINY_URL = (
    "https://huggingface.co/MapleF/eva_x/resolve/main/"
    "eva_x_tiny_patch16_merged520k_mim.pt"
)

CXFORMER_SOURCE_COMMIT = "da740db15a6d16cc06a26b82bd78527f2616dd7e"
CXFORMER_SMALL_REVISION = "21777c302a43197b704c6c92591d9897efaaac3b"
CXFORMER_SMALL_URL = (
    "https://huggingface.co/m42-health/CXformer-small/resolve/"
    f"{CXFORMER_SMALL_REVISION}/model.safetensors"
)
CXFORMER_SMALL_MIRROR_URL = (
    "https://hf-mirror.com/m42-health/CXformer-small/resolve/"
    f"{CXFORMER_SMALL_REVISION}/model.safetensors"
)
CXFORMER_SMALL_SHA256 = "a300b6e23ca32b361c2209ac114d77ba13dfeb151cd4a337212d13416df273a2"


@dataclass(frozen=True)
class Condition:
    name: str
    degradation: str | None = None
    severity: str | None = None


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    learning_rate: float
    weight_decay: float
    amp_safe: bool = True
    backbone_learning_rate: float | None = None
    histogram_equalization: bool = False
    center_crop: bool = False


BASELINE_SPECS = {
    "medvitv2": BaselineSpec(
        "medvitv2", 224, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 3e-4, 0.05, False
    ),
    "medmamba": BaselineSpec(
        "medmamba", 224, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 1e-4, 1e-4, False
    ),
    "eva_x": BaselineSpec(
        "eva_x",
        224,
        (0.49185243, 0.49185243, 0.49185243),
        (0.28509309, 0.28509309, 0.28509309),
        1e-4,
        0.05,
        False,
    ),
    "cxformer_s": BaselineSpec(
        "cxformer_s",
        518,
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
        5e-5,
        0.0,
        False,
        backbone_learning_rate=5e-7,
        histogram_equalization=True,
        center_crop=True,
    ),
}


def model_source_commit(name: str) -> str:
    if name == "cxformer_s":
        return f"github:{CXFORMER_SOURCE_COMMIT};hf:{CXFORMER_SMALL_REVISION}"
    return REPOSITORIES[name]["commit"]


def require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch and torchvision are required. Install the project with "
            "`python -m pip install -e .` before running this script."
        ) from exc
    return torch, DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch, _, _ = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def stable_seed(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def parse_models(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(MODEL_ORDER)
    aliases = {
        "eva-x": "eva_x",
        "medvit-v2": "medvitv2",
        "medvit2": "medvitv2",
        "cxformer": "cxformer_s",
        "cxformer-s": "cxformer_s",
    }
    models = [
        aliases.get(item.strip().lower(), item.strip().lower())
        for item in value.split(",")
        if item.strip()
    ]
    unknown = sorted(set(models) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; choose from {MODEL_ORDER}")
    return list(dict.fromkeys(models))


def parse_conditions(value: str) -> list[Condition]:
    if value.strip().lower() == "all":
        return [Condition("clean")] + [
            Condition(f"{name}:{level}", name, level)
            for name in DEGRADATIONS
            for level in SEVERITIES
        ]
    result: list[Condition] = []
    for raw in value.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token == "clean":
            result.append(Condition("clean"))
            continue
        if ":" in token:
            name, level = token.split(":", 1)
            if name not in DEGRADATIONS or level not in SEVERITIES:
                raise ValueError(f"Invalid condition: {token}")
            result.append(Condition(token, name, level))
            continue
        if token not in DEGRADATIONS:
            raise ValueError(f"Invalid condition: {token}")
        result.extend(Condition(f"{token}:{level}", token, level) for level in SEVERITIES)
    if not result:
        raise ValueError("At least one evaluation condition is required")
    return list(dict.fromkeys(result))


def resolve_device(requested: str):
    torch, _, _ = require_torch()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def source_directory(source_root: Path, model_name: str) -> Path:
    return source_root / REPOSITORIES[model_name]["directory"]


def fetch_sources(source_root: Path) -> None:
    source_root.mkdir(parents=True, exist_ok=True)
    for name, record in REPOSITORIES.items():
        target = source_directory(source_root, name)
        marker = target / record["marker"]
        if marker.exists():
            print(f"[SOURCE] {name}: using {target}", flush=True)
            continue
        print(f"[SOURCE] cloning {name} from {record['url']}", flush=True)
        subprocess.run(
            ["git", "clone", "--filter=blob:none", record["url"], str(target)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(target), "checkout", record["commit"]],
            check=True,
        )
        if not marker.exists():
            raise RuntimeError(f"Official source marker is missing after clone: {marker}")


def verify_source(source_root: Path, model_name: str) -> Path:
    record = REPOSITORIES[model_name]
    root = source_directory(source_root, model_name)
    marker = root / record["marker"]
    if not marker.exists():
        raise FileNotFoundError(
            f"Missing official {model_name} source at {marker}. "
            "Run once with --fetch-sources."
        )
    return root


def import_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create an import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def unwrap_state_dict(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint must contain a state-dict-like mapping")
    current = payload
    for key in ("model_ema", "model", "module", "state_dict"):
        value = current.get(key)
        if isinstance(value, dict):
            current = value
    cleaned = {}
    for key, value in current.items():
        name = str(key)
        for prefix in ("module.", "backbone."):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        cleaned[name] = value
    return cleaned


def load_compatible_weights(model, state_dict: dict[str, object], label: str) -> None:
    own = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in own and hasattr(value, "shape") and tuple(value.shape) == tuple(own[key].shape)
    }
    skipped = sorted(set(state_dict) - set(compatible))
    message = model.load_state_dict(compatible, strict=False)
    print(
        f"[PRETRAIN] {label}: loaded={len(compatible)} skipped={len(skipped)} "
        f"missing={len(message.missing_keys)} unexpected={len(message.unexpected_keys)}",
        flush=True,
    )


def _affine_scan_combine(left, right):
    """Compose two elementwise affine maps for an associative prefix scan."""

    left_a, left_b = left
    right_a, right_b = right
    return right_a * left_a, right_a * left_b + right_b


_AFFINE_SCAN_AUTOGRAD = None


def _affine_scan_states(transition, drive):
    """Evaluate x[t] = transition[t] * x[t-1] + drive[t] on CUDA.

    ``torch.associative_scan`` is fast on CUDA but intentionally has no
    autograd implementation.  The small custom Function below supplies the
    exact reverse recurrence, so MedMamba remains fully trainable without a
    separately compiled mamba_ssm wheel.
    """

    torch, _, _ = require_torch()
    from torch._higher_order_ops.associative_scan import associative_scan

    global _AFFINE_SCAN_AUTOGRAD
    if _AFFINE_SCAN_AUTOGRAD is None:

        class _AffineScanAutograd(torch.autograd.Function):
            @staticmethod
            def forward(ctx, scan_transition, scan_drive):
                _, states = associative_scan(
                    _affine_scan_combine,
                    (scan_transition, scan_drive),
                    dim=-1,
                    combine_mode="pointwise",
                )
                ctx.save_for_backward(scan_transition, states)
                return states

            @staticmethod
            def backward(ctx, grad_states):
                scan_transition, states = ctx.saved_tensors
                # r[t] = grad_states[t] + transition[t+1] * r[t+1]
                reverse_transition = torch.cat(
                    (
                        scan_transition[..., 1:],
                        torch.zeros_like(scan_transition[..., :1]),
                    ),
                    dim=-1,
                )
                _, grad_drive = associative_scan(
                    _affine_scan_combine,
                    (reverse_transition, grad_states.contiguous()),
                    dim=-1,
                    reverse=True,
                    combine_mode="pointwise",
                )
                previous_states = torch.cat(
                    (torch.zeros_like(states[..., :1]), states[..., :-1]), dim=-1
                )
                grad_transition = grad_drive * previous_states
                return grad_transition, grad_drive

        _AFFINE_SCAN_AUTOGRAD = _AffineScanAutograd

    return _AFFINE_SCAN_AUTOGRAD.apply(transition, drive)


def selective_scan_torch_reference(
    u,
    delta,
    A,
    B,
    C,
    D=None,
    z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
    **_kwargs,
):
    """Differentiable PyTorch implementation matching selective_scan_fn.

    CUDA uses a parallel associative scan.  CPU retains the simple sequential
    reference path, which is intended only for debugging.
    """

    torch, _, _ = require_torch()
    input_dtype = u.dtype
    u = u.float()
    delta = delta.float()
    A = A.float()
    B = B.float()
    C = C.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float().view(1, -1, 1)
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
    batch, channels, length = u.shape
    state_size = A.shape[-1]
    groups = B.shape[1] if B.ndim == 4 else 1
    channels_per_group = channels // groups

    if u.is_cuda:
        outputs = []
        last_states = []
        for group in range(groups):
            start = group * channels_per_group
            stop = start + channels_per_group
            group_u = u[:, start:stop]
            group_delta = delta[:, start:stop]
            group_a = A[start:stop]
            transition = torch.exp(
                group_delta.unsqueeze(-2) * group_a.unsqueeze(0).unsqueeze(-1)
            )
            if B.ndim == 4:
                group_b = B[:, group].unsqueeze(1)
                group_c = C[:, group].unsqueeze(1)
            else:
                group_b = B.unsqueeze(1)
                group_c = C.unsqueeze(1)
            drive = (
                group_delta.unsqueeze(-2)
                * group_b
                * group_u.unsqueeze(-2)
            )
            states = _affine_scan_states(transition, drive)
            outputs.append(torch.sum(states * group_c, dim=-2))
            if return_last_state:
                last_states.append(states[..., -1])
        output = torch.cat(outputs, dim=1)
        if D is not None:
            output = output + u * D.float().view(1, -1, 1)
        if z is not None:
            output = output * torch.nn.functional.silu(z.float())
        output = output.to(input_dtype)
        if return_last_state:
            return output, torch.cat(last_states, dim=1)
        return output

    state = u.new_zeros((batch, channels, state_size))
    outputs = []
    for index in range(length):
        delta_i = delta[:, :, index]
        transition = torch.exp(delta_i.unsqueeze(-1) * A.unsqueeze(0))
        if B.ndim == 4:
            b_i = B[:, :, :, index].repeat_interleave(channels_per_group, dim=1)
        else:
            b_i = B[:, :, index].unsqueeze(1).expand(-1, channels, -1)
        state = transition * state + delta_i.unsqueeze(-1) * b_i * u[:, :, index].unsqueeze(-1)
        if C.ndim == 4:
            c_i = C[:, :, :, index].repeat_interleave(channels_per_group, dim=1)
        else:
            c_i = C[:, :, index].unsqueeze(1).expand(-1, channels, -1)
        outputs.append(torch.sum(state * c_i, dim=-1))
    output = torch.stack(outputs, dim=-1)
    if D is not None:
        output = output + u * D.float().view(1, -1, 1)
    if z is not None:
        output = output * torch.nn.functional.silu(z.float())
    output = output.to(input_dtype)
    return (output, state) if return_last_state else output


def medmamba_native_scan_available() -> bool:
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _scan

        return _scan is not None
    except Exception:
        return False


def build_medvitv2(source_root: Path, pretrained: Path | None):
    root = verify_source(source_root, "medvitv2")
    sys.path.insert(0, str(root))
    try:
        module = import_file("comparison_medvitv2", root / "MedViT.py")
        model = module.MedViT_tiny(num_classes=1)
    finally:
        if sys.path and sys.path[0] == str(root):
            sys.path.pop(0)
    if pretrained is not None:
        torch, _, _ = require_torch()
        load_compatible_weights(
            model, unwrap_state_dict(torch.load(pretrained, map_location="cpu", weights_only=False)), "MedViTV2"
        )
    return model


def build_medmamba(source_root: Path, pretrained: Path | None, allow_slow_fallback: bool):
    root = verify_source(source_root, "medmamba")
    source_path = root / "MedMamba.py"
    source = source_path.read_text(encoding="utf-8")
    demo_marker = "\nmedmamba_t = VSSM("
    if demo_marker in source:
        source = source.split(demo_marker, 1)[0]
    module = types.ModuleType("comparison_medmamba")
    module.__file__ = str(source_path)
    sys.modules[module.__name__] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    native_scan = "selective_scan_fn" in module.__dict__
    if not native_scan:
        module.selective_scan_fn = selective_scan_torch_reference
        print(
            "[SCAN] MedMamba: mamba_ssm is unavailable; using the PyTorch 2.11 "
            "CUDA associative-scan backend.",
            flush=True,
        )
    model = module.VSSM(depths=[2, 2, 4, 2], dims=[96, 192, 384, 768], num_classes=1)
    model._comparison_scan_backend = "mamba_ssm" if native_scan else "torch_associative"
    if pretrained is not None:
        torch, _, _ = require_torch()
        load_compatible_weights(
            model, unwrap_state_dict(torch.load(pretrained, map_location="cpu", weights_only=False)), "MedMamba"
        )
    return model


def build_eva_x(
    source_root: Path,
    pretrained: Path | None,
    allow_random_init: bool,
    warn_random_init: bool = True,
):
    root = verify_source(source_root, "eva_x")
    module = import_file("comparison_eva_x", root / "eva_x.py")
    model = module.EVA_X(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(14, 14),
        num_classes=1,
    )
    if pretrained is None:
        if not allow_random_init:
            raise RuntimeError(
                "EVA-X must use its chest-X-ray self-supervised checkpoint for a valid EVA-X "
                "comparison. Supply --eva-x-pretrained or --download-pretrained."
            )
        if warn_random_init:
            print(
                "[WARNING] EVA-X is randomly initialized; do not report this as pretrained EVA-X.",
                flush=True,
            )
        return model
    torch, _, _ = require_torch()
    checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
    filtered = module.checkpoint_filter_fn(checkpoint, model)
    load_compatible_weights(model, filtered, "EVA-X-Ti")
    return model


def load_cxformer_s_weights(backbone, checkpoint: Path) -> None:
    """Map the official Hugging Face DINOv2-with-registers weights to timm."""

    torch, _, _ = require_torch()
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "CXFormer-S needs safetensors, which is expected in the existing project environment."
        ) from exc

    raw_state = load_file(str(checkpoint), device="cpu")
    hf_state = {
        (key[len("model.") :] if key.startswith("model.") else key): value
        for key, value in raw_state.items()
    }
    target = backbone.state_dict()
    mapped = {}
    direct = {
        "embeddings.register_tokens": "reg_token",
        "embeddings.patch_embeddings.projection.weight": "patch_embed.proj.weight",
        "embeddings.patch_embeddings.projection.bias": "patch_embed.proj.bias",
        "layernorm.weight": "norm.weight",
        "layernorm.bias": "norm.bias",
    }
    for source_key, target_key in direct.items():
        value = hf_state.get(source_key)
        if value is not None and target_key in target and value.shape == target[target_key].shape:
            mapped[target_key] = value

    # Hugging Face stores one positional vector for CLS; timm folds it into
    # cls_token and keeps pos_embed for patch tokens only.
    cls_token = hf_state.get("embeddings.cls_token")
    position_embeddings = hf_state.get("embeddings.position_embeddings")
    if cls_token is not None and position_embeddings is not None:
        converted_cls = cls_token + position_embeddings[:, :1]
        converted_position = position_embeddings[:, 1:]
        if converted_cls.shape == target["cls_token"].shape:
            mapped["cls_token"] = converted_cls
        if converted_position.shape == target["pos_embed"].shape:
            mapped["pos_embed"] = converted_position

    block_fields = {
        "norm1.weight": "norm1.weight",
        "norm1.bias": "norm1.bias",
        "norm2.weight": "norm2.weight",
        "norm2.bias": "norm2.bias",
        "attention.output.dense.weight": "attn.proj.weight",
        "attention.output.dense.bias": "attn.proj.bias",
        "layer_scale1.lambda1": "ls1.gamma",
        "layer_scale2.lambda1": "ls2.gamma",
        "mlp.fc1.weight": "mlp.fc1.weight",
        "mlp.fc1.bias": "mlp.fc1.bias",
        "mlp.fc2.weight": "mlp.fc2.weight",
        "mlp.fc2.bias": "mlp.fc2.bias",
    }
    for layer in range(12):
        source_prefix = f"encoder.layer.{layer}."
        target_prefix = f"blocks.{layer}."
        for source_suffix, target_suffix in block_fields.items():
            source_key = source_prefix + source_suffix
            target_key = target_prefix + target_suffix
            value = hf_state.get(source_key)
            if value is not None and target_key in target and value.shape == target[target_key].shape:
                mapped[target_key] = value

        attention_prefix = source_prefix + "attention.attention."
        for suffix in ("weight", "bias"):
            parts = [
                hf_state.get(attention_prefix + projection + "." + suffix)
                for projection in ("query", "key", "value")
            ]
            target_key = target_prefix + "attn.qkv." + suffix
            if all(part is not None for part in parts):
                merged = torch.cat(parts, dim=0)
                if target_key in target and merged.shape == target[target_key].shape:
                    mapped[target_key] = merged

    if len(mapped) < 170:
        raise RuntimeError(
            f"CXFormer-S checkpoint mapping loaded only {len(mapped)} tensors; "
            "the checkpoint does not match the official small model."
        )
    message = backbone.load_state_dict(mapped, strict=False)
    print(
        f"[PRETRAIN] CXFormer-S: loaded={len(mapped)} "
        f"missing={len(message.missing_keys)} unexpected={len(message.unexpected_keys)}",
        flush=True,
    )


def build_cxformer_s(pretrained: Path | None, allow_random_init: bool):
    """Build CXFormer-S with timm, without adding Transformers to the environment."""

    torch, _, _ = require_torch()
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("CXFormer-S requires the project's existing timm installation") from exc

    backbone = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        pretrained=False,
        img_size=518,
        num_classes=0,
    )
    if pretrained is not None:
        load_cxformer_s_weights(backbone, pretrained)
    elif not allow_random_init:
        raise RuntimeError(
            "CXFormer-S must use its official chest-X-ray pretrained checkpoint. "
            "Keep --download-pretrained enabled or supply --cxformer-s-pretrained."
        )

    class CXFormerSmallClassifier(torch.nn.Module):
        def __init__(self, feature_extractor):
            super().__init__()
            self.backbone = feature_extractor
            # Official downstream recipe: four CLS tokens plus final-layer mean pool.
            self.classifier = torch.nn.Linear(384 * 5, 1)
            torch.nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
            torch.nn.init.zeros_(self.classifier.bias)

        def forward(self, images):
            intermediates = self.backbone.forward_intermediates(
                images,
                indices=4,
                return_prefix_tokens=True,
                norm=False,
                output_fmt="NLC",
                intermediates_only=True,
            )
            cls_features = [prefix[:, 0] for spatial, prefix in intermediates]
            final_spatial, final_prefix = intermediates[-1]
            pooled = torch.cat((final_prefix[:, 1:], final_spatial), dim=1).mean(dim=1)
            features = torch.cat((*cls_features, pooled), dim=-1)
            return self.classifier(features)

    return CXFormerSmallClassifier(backbone)


def build_baseline_model(
    name: str,
    source_root: Path,
    pretrained: Path | None,
    allow_random_init: bool,
    allow_slow_medmamba_fallback: bool,
    warn_random_init: bool = True,
):
    if name == "medvitv2":
        return build_medvitv2(source_root, pretrained)
    if name == "medmamba":
        return build_medmamba(source_root, pretrained, allow_slow_medmamba_fallback)
    if name == "eva_x":
        return build_eva_x(
            source_root, pretrained, allow_random_init, warn_random_init=warn_random_init
        )
    if name == "cxformer_s":
        return build_cxformer_s(pretrained, allow_random_init)
    raise ValueError(name)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: Path, expected_sha256: str | None = None) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        if expected_sha256 is None or file_sha256(target) == expected_sha256:
            return target
        print(f"[DOWNLOAD] removing incomplete or invalid file: {target}", flush=True)
        target.unlink()
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(f"[DOWNLOAD] {url} -> {target}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        try:
            # WSL networking can be unavailable even when Windows networking works.
            windows_target = subprocess.check_output(
                ["wslpath", "-w", str(temporary)], text=True
            ).strip()
            print("[DOWNLOAD] WSL network unavailable; retrying through Windows", flush=True)
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "$ProgressPreference='SilentlyContinue'; "
                    "Invoke-WebRequest -UseBasicParsing -Uri $args[0] -OutFile $args[1]",
                    url,
                    windows_target,
                ],
                check=True,
            )
        except Exception as windows_exc:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(
                f"Unable to download the official pretrained checkpoint. Download it manually from "
                f"{url} and save it as {target}"
            ) from windows_exc
    if expected_sha256 is not None and file_sha256(temporary) != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file failed SHA256 verification: {url}")
    temporary.replace(target)
    return target


def resolve_pretrained_paths(args) -> dict[str, Path | None]:
    result = {
        "medvitv2": Path(args.medvitv2_pretrained).resolve() if args.medvitv2_pretrained else None,
        "medmamba": Path(args.medmamba_pretrained).resolve() if args.medmamba_pretrained else None,
        "eva_x": Path(args.eva_x_pretrained).resolve() if args.eva_x_pretrained else None,
        "cxformer_s": (
            Path(args.cxformer_s_pretrained).resolve()
            if args.cxformer_s_pretrained
            else None
        ),
    }
    if args.download_pretrained and result["eva_x"] is None:
        result["eva_x"] = download_file(
            EVA_X_TINY_URL, args.output_root / "pretrained" / "eva_x_tiny_patch16_merged520k_mim.pt"
        )
    if args.download_pretrained and result["cxformer_s"] is None:
        cxformer_target = (
            args.output_root / "pretrained" / "cxformer_small_21777c3.safetensors"
        )
        try:
            result["cxformer_s"] = download_file(
                CXFORMER_SMALL_URL, cxformer_target, CXFORMER_SMALL_SHA256
            )
        except RuntimeError:
            print(
                "[DOWNLOAD] official CXFormer-S route failed; retrying the "
                "SHA256-verified Hugging Face mirror",
                flush=True,
            )
            result["cxformer_s"] = download_file(
                CXFORMER_SMALL_MIRROR_URL,
                cxformer_target,
                CXFORMER_SMALL_SHA256,
            )
    for name, path in result.items():
        if path is not None and not path.exists():
            raise FileNotFoundError(f"{name} pretrained checkpoint does not exist: {path}")
    return result


def _border_median(image: Image.Image) -> tuple[int, int, int]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    value = int(np.median(border))
    return value, value, value


def prepare_baseline_image(
    image: Image.Image,
    spec: BaselineSpec,
    training: bool,
    augment_robustness: bool,
    rng: np.random.Generator,
):
    torch, _, _ = require_torch()
    gray = image.convert("L")
    if training and augment_robustness and rng.random() < 0.70:
        name = str(rng.choice(DEGRADATIONS))
        level = str(rng.choice(SEVERITIES))
        gray = apply_degradation(gray, name, level, rng)
        if rng.random() < 0.25:
            gray = apply_degradation(
                gray, str(rng.choice(DEGRADATIONS)), str(rng.choice(SEVERITIES)), rng
            )
    if spec.histogram_equalization:
        gray = ImageOps.equalize(gray)
    rgb = gray.convert("RGB")
    if spec.center_crop:
        rgb = ImageOps.fit(
            rgb,
            (spec.input_size, spec.input_size),
            method=Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
    else:
        rgb = ImageOps.pad(
            rgb,
            (spec.input_size, spec.input_size),
            method=Image.Resampling.BICUBIC,
            color=_border_median(rgb),
            centering=(0.5, 0.5),
        )
    if training and rng.random() < 0.5:
        rgb = ImageOps.mirror(rgb)
    array = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray(spec.mean, dtype=np.float32)[:, None, None]
    std = np.asarray(spec.std, dtype=np.float32)[:, None, None]
    return torch.from_numpy((array - mean) / std)


def make_baseline_dataset_class():
    _, _, Dataset = require_torch()

    class BaselineDataset(Dataset):
        def __init__(
            self,
            rows: Sequence[dict[str, str]],
            root: Path,
            spec: BaselineSpec,
            seed: int,
            training: bool = False,
            augment_robustness: bool = False,
            condition: Condition | None = None,
        ):
            self.rows = list(rows)
            self.root = root
            self.spec = spec
            self.seed = seed
            self.training = training
            self.augment_robustness = augment_robustness
            self.condition = condition or Condition("clean")

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int):
            row = self.rows[index]
            sample_id = row.get("sample_id") or f"{row['patient_id']}:{row['image_path']}"
            if self.training:
                # The DataLoader seeds NumPy per worker. Drawing a fresh child
                # seed gives reproducible but epoch-varying degradation samples.
                rng = np.random.default_rng(np.random.randint(0, 2**32, dtype=np.uint32))
            else:
                rng = np.random.default_rng(
                    stable_seed(self.seed, sample_id, self.condition.name, "eval")
                )
            with Image.open(self.root / row["image_path"]) as source:
                image = source.convert("L")
            if self.condition.degradation is not None:
                image = apply_degradation(
                    image, self.condition.degradation, self.condition.severity, rng
                )
            tensor = prepare_baseline_image(
                image, self.spec, self.training, self.augment_robustness, rng
            )
            return tensor, int(row["label"]), str(sample_id)

    return BaselineDataset


def worker_seed(worker_id: int) -> None:
    torch, _, _ = require_torch()
    seed = (int(torch.initial_seed()) + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def rows_for_split(
    rows: Sequence[dict[str, str]], split: str, max_samples: int | None = None
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["split"] == split]
    if max_samples is None or max_samples >= len(selected):
        return selected
    # Keep smoke tests useful by interleaving classes instead of taking the
    # first manifest rows, which are often ordered by class on disk.
    by_label: dict[str, list[dict[str, str]]] = {}
    for row in selected:
        by_label.setdefault(str(row["label"]), []).append(row)
    result: list[dict[str, str]] = []
    index = 0
    labels = sorted(by_label)
    while len(result) < max_samples:
        progressed = False
        for label in labels:
            if index < len(by_label[label]) and len(result) < max_samples:
                result.append(by_label[label][index])
                progressed = True
        if not progressed:
            break
        index += 1
    return result


def average_precision(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    sorted_y = y[order]
    sorted_p = p[order]
    true_positives = np.cumsum(sorted_y)
    false_positives = np.cumsum(1 - sorted_y)
    distinct = np.where(np.diff(sorted_p))[0]
    threshold_indices = np.r_[distinct, len(sorted_p) - 1]
    precision = true_positives[threshold_indices] / (
        true_positives[threshold_indices] + false_positives[threshold_indices]
    )
    recall = true_positives[threshold_indices] / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def classification_metrics(
    labels: Iterable[int], probabilities: Iterable[float], threshold: float
) -> dict[str, float]:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.ndim != 1 or y.shape != p.shape or y.size == 0:
        raise ValueError("Labels and probabilities must be non-empty aligned vectors")
    pred = (p >= threshold).astype(np.int64)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    return {
        "auroc": float(auroc(y, p)),
        "auprc": average_precision(y, p),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "n": int(len(y)),
    }


def autocast_context(torch, device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, enabled=True)


def predict_logits(model, loader, device, amp: bool) -> tuple[list[int], list[float], float]:
    torch, _, _ = require_torch()
    model.eval()
    labels: list[int] = []
    logits: list[float] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for images, targets, _sample_ids in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(torch, device, amp and device.type == "cuda"):
                output = model(images).reshape(-1)
            logits.extend(output.detach().float().cpu().tolist())
            labels.extend(np.asarray(targets).astype(int).tolist())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    latency_ms = 1000.0 * elapsed / max(len(labels), 1)
    return labels, logits, latency_ms


def save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    torch, _, _ = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_baseline(name: str, args, pretrained: Path | None) -> Path:
    torch, DataLoader, _ = require_torch()
    spec = BASELINE_SPECS[name]
    rows = read_manifest(project.MANIFEST_PATH)
    assert_no_patient_leakage(rows)
    train_rows = rows_for_split(rows, "train", args.max_train_samples)
    validation_rows = rows_for_split(rows, "validation", args.max_validation_samples)
    Dataset = make_baseline_dataset_class()
    train_dataset = Dataset(
        train_rows,
        project.DATASET_ROOT,
        spec,
        args.seed,
        training=True,
        augment_robustness=not args.no_robust_training,
    )
    validation_dataset = Dataset(
        validation_rows, project.DATASET_ROOT, spec, args.seed, training=False
    )
    train_batch_size = args.batch_size
    validation_batch_size = args.eval_batch_size
    if name == "medmamba" and not medmamba_native_scan_available():
        # The PyTorch scan is fully GPU accelerated but materializes scan
        # states.  Batch four peaks below 5 GB on an 8 GB RTX 5060 Laptop GPU.
        train_batch_size = min(args.batch_size, 4)
        validation_batch_size = min(args.eval_batch_size, 4)
        print(
            f"[BATCH] medmamba PyTorch scan: train={train_batch_size} "
            f"validation={validation_batch_size} (8 GB safe)",
            flush=True,
        )
    if name == "cxformer_s":
        train_batch_size = min(args.batch_size, 4)
        validation_batch_size = min(args.eval_batch_size, 8)
        print(
            f"[BATCH] cxformer_s 518px: train={train_batch_size} "
            f"validation={validation_batch_size} (8 GB safe)",
            flush=True,
        )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=worker_seed if args.num_workers else None,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=validation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        worker_init_fn=worker_seed if args.num_workers else None,
    )
    device = resolve_device(args.device)
    seed_everything(args.seed)
    model = build_baseline_model(
        name,
        args.source_root,
        pretrained,
        args.allow_random_init,
        args.allow_slow_medmamba_fallback,
    ).to(device)
    if name == "cxformer_s" and hasattr(model.backbone, "set_grad_checkpointing"):
        model.backbone.set_grad_checkpointing(True)
    labels = np.asarray([int(row["label"]) for row in train_rows], dtype=np.int64)
    positives = max(int(labels.sum()), 1)
    pos_weight = torch.tensor([(len(labels) - positives) / positives], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "cxformer_s":
        optimizer = torch.optim.AdamW(
            [
                {"params": model.classifier.parameters(), "lr": spec.learning_rate},
                {
                    "params": model.backbone.parameters(),
                    "lr": spec.backbone_learning_rate,
                },
            ],
            weight_decay=spec.weight_decay,
        )
        print(
            f"[OPTIMIZER] cxformer_s classifier_lr={spec.learning_rate:g} "
            f"backbone_lr={spec.backbone_learning_rate:g}",
            flush=True,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = bool(args.amp and device.type == "cuda" and spec.amp_safe)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoint = baseline_checkpoint_path(name, args)
    last_checkpoint = checkpoint.with_name(f"{checkpoint.stem}.last{checkpoint.suffix}")
    best_auc = -math.inf
    best_epoch = 0
    stale = 0
    start_epoch = 1
    resume_path = last_checkpoint if last_checkpoint.exists() else checkpoint
    if resume_path.exists() and not args.force_retrain:
        resume = torch.load(resume_path, map_location="cpu", weights_only=False)
        if not bool(resume.get("training_complete", False)):
            model.load_state_dict(resume["state_dict"], strict=True)
            completed = int(resume.get("epochs_completed", resume.get("epoch", 0)))
            start_epoch = completed + 1
            best_auc = float(
                resume.get("best_validation_auroc", resume.get("validation_auroc", -math.inf))
            )
            best_epoch = int(resume.get("best_epoch", resume.get("epoch", 0)))
            stale = int(resume.get("stale", 0))
            if "optimizer_state_dict" in resume:
                optimizer.load_state_dict(resume["optimizer_state_dict"])
            if "scheduler_state_dict" in resume:
                scheduler.load_state_dict(resume["scheduler_state_dict"])
            if scaler.is_enabled() and resume.get("scaler_state_dict"):
                scaler.load_state_dict(resume["scaler_state_dict"])
            generator.manual_seed(args.seed + completed)
            print(
                f"[RESUME] {name}: loaded epoch {completed}; continuing at epoch {start_epoch}",
                flush=True,
            )
    print(
        f"\n[TRAIN START] {name} | device={device} | train={len(train_dataset)} | "
        f"validation={len(validation_dataset)} | epochs={args.epochs} | "
        f"amp={'on' if use_amp else 'off'} | grad_clip=1.0",
        flush=True,
    )
    completed_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        epoch_started = time.perf_counter()
        total_batches = len(train_loader)
        print(
            f"[EPOCH START] model={name} epoch={epoch:02d}/{args.epochs:02d} "
            f"batches={total_batches}",
            flush=True,
        )
        for batch_index, (images, targets, _sample_ids) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(torch, device, scaler.is_enabled()):
                logits = model(images).reshape(-1)
                loss = criterion(logits, targets)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError(
                    f"{name} produced a non-finite training loss at epoch {epoch}; "
                    "the last finite checkpoint was preserved"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not bool(torch.isfinite(grad_norm).item()):
                raise FloatingPointError(
                    f"{name} produced non-finite gradients at epoch {epoch}; "
                    "the last finite checkpoint was preserved"
                )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * len(targets)
            total_samples += len(targets)
            if (
                batch_index == 1
                or batch_index % args.progress_every_batches == 0
                or batch_index == total_batches
            ):
                elapsed = time.perf_counter() - epoch_started
                batches_per_second = batch_index / max(elapsed, 1e-9)
                remaining_seconds = (total_batches - batch_index) / max(
                    batches_per_second, 1e-9
                )
                print(
                    f"[TRAIN PROGRESS] model={name} epoch={epoch:02d}/{args.epochs:02d} "
                    f"batch={batch_index}/{total_batches} "
                    f"progress={100.0 * batch_index / total_batches:5.1f}% "
                    f"loss={total_loss / max(total_samples, 1):.6f} "
                    f"elapsed={elapsed / 60.0:.1f}min "
                    f"eta={remaining_seconds / 60.0:.1f}min",
                    flush=True,
                )
        print(
            f"[VALIDATION START] model={name} epoch={epoch:02d}/{args.epochs:02d} "
            f"batches={len(validation_loader)}",
            flush=True,
        )
        val_labels, val_logits, _ = predict_logits(model, validation_loader, device, use_amp)
        val_logits_array = np.asarray(val_logits, dtype=np.float64)
        if not np.isfinite(val_logits_array).all():
            count = int((~np.isfinite(val_logits_array)).sum())
            raise FloatingPointError(
                f"{name} produced {count} non-finite validation logits at epoch {epoch}; "
                "the last finite checkpoint was preserved"
            )
        val_auc = auroc(val_labels, sigmoid(val_logits_array))
        improved = val_auc > best_auc
        if improved:
            best_auc = float(val_auc)
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                checkpoint,
                {
                    "model_name": name,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                    "epochs_completed": epoch,
                    "epochs_requested": args.epochs,
                    "validation_auroc": best_auc,
                    "best_validation_auroc": best_auc,
                    "best_epoch": best_epoch,
                    "training_complete": False,
                    "spec": asdict(spec),
                    "source_commit": model_source_commit(name),
                    "pretrained_checkpoint": str(pretrained) if pretrained else None,
                    "robust_training": not args.no_robust_training,
                    "seed": args.seed,
                    "train_samples": len(train_dataset),
                    "validation_samples": len(validation_dataset),
                },
            )
        else:
            stale += 1
        scheduler.step()
        completed_epoch = epoch
        save_checkpoint(
            last_checkpoint,
            {
                "model_name": name,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
                "epoch": epoch,
                "epochs_completed": epoch,
                "epochs_requested": args.epochs,
                "validation_auroc": float(val_auc),
                "best_validation_auroc": float(best_auc),
                "best_epoch": int(best_epoch),
                "stale": int(stale),
                "training_complete": False,
                "spec": asdict(spec),
                "source_commit": model_source_commit(name),
                "pretrained_checkpoint": str(pretrained) if pretrained else None,
                "robust_training": not args.no_robust_training,
                "seed": args.seed,
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
            },
        )
        print(
            f"[EPOCH {epoch:02d}/{args.epochs:02d}] model={name} "
            f"loss={total_loss / max(total_samples, 1):.6f} "
            f"val_auroc={val_auc:.6f} best={best_auc:.6f}@{best_epoch} "
            f"early_stop={stale}/{args.patience} seconds={time.perf_counter() - epoch_started:.1f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    if not checkpoint.exists():
        raise RuntimeError(f"{name} training ended without a finite best checkpoint")
    best_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    best_payload["training_complete"] = True
    best_payload["epochs_completed"] = int(completed_epoch)
    best_payload["epochs_requested"] = int(args.epochs)
    best_payload["stopped_early"] = bool(completed_epoch < args.epochs)
    best_payload["best_validation_auroc"] = float(best_auc)
    best_payload["best_epoch"] = int(best_epoch)
    best_payload["spec"] = asdict(spec)
    best_payload["source_commit"] = model_source_commit(name)
    best_payload["pretrained_checkpoint"] = str(pretrained) if pretrained else None
    best_payload["robust_training"] = not args.no_robust_training
    best_payload["seed"] = int(args.seed)
    best_payload["train_samples"] = len(train_dataset)
    best_payload["validation_samples"] = len(validation_dataset)
    save_checkpoint(checkpoint, best_payload)
    print(f"[TRAIN END] {name}: {checkpoint}", flush=True)
    return checkpoint


def ensure_manifest_and_quality_model(args) -> None:
    """Create only missing prerequisites; never repartition an existing manifest."""

    if not project.MANIFEST_PATH.exists():
        if not project.DATASET_ROOT.exists():
            raise FileNotFoundError(f"Chest-X-ray dataset root is missing: {project.DATASET_ROOT}")
        print("[PREPARE] patient-level manifest is missing; creating it now", flush=True)
        rows = build_presplit_manifest(
            project.DATASET_ROOT,
            None,
            repartition_patients=True,
            seed=args.seed,
        )
        sample_counts = Counter(str(row["sample_id"]) for row in rows)
        duplicates = [key for key, count in sample_counts.items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate sample identifiers detected: {duplicates[:5]}")
        write_manifest(rows, project.MANIFEST_PATH)
    else:
        rows = read_manifest(project.MANIFEST_PATH)
        assert_no_patient_leakage(rows)
        print(f"[PREPARE] reusing fixed manifest: {project.MANIFEST_PATH}", flush=True)

    if project.QUALITY_MODEL_PATH.exists():
        print(f"[PREPARE] reusing quality model: {project.QUALITY_MODEL_PATH}", flush=True)
        return

    print("[PREPARE] quality model is missing; fitting it from train/calibration only", flush=True)
    grouped: dict[str, list[dict[str, float]]] = {"train": [], "calibration": []}
    for row in rows:
        split = str(row["split"])
        if split not in grouped:
            continue
        with Image.open(project.DATASET_ROOT / row["image_path"]) as image:
            grouped[split].append(
                raw_quality_metrics(
                    image,
                    original_shape=(image.height, image.width),
                    analysis_max_side=project.APP_CONFIG.quality.analysis_max_side,
                )
            )
    if not grouped["train"] or not grouped["calibration"]:
        raise ValueError("Manifest must contain non-empty train and calibration splits")
    quality_model = QualityModel.fit(
        grouped["train"], grouped["calibration"], project.APP_CONFIG.quality
    )
    project.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    quality_model.save(project.QUALITY_MODEL_PATH)
    print(f"[PREPARE] saved quality model: {project.QUALITY_MODEL_PATH}", flush=True)


def ours_artifacts_complete() -> bool:
    required = (
        project.LIGHT_CHECKPOINT,
        project.HEAVY_CHECKPOINT,
        project.QUALITY_MODEL_PATH,
        project.CALIBRATOR_PATH,
        project.FROZEN_CONFIG_PATH,
    )
    return all(Path(path).exists() for path in required)


def baseline_checkpoint_path(name: str, args) -> Path:
    return args.output_root / "checkpoints" / f"{name}.pt"


def baseline_checkpoint_complete(name: str, args) -> bool:
    path = baseline_checkpoint_path(name, args)
    if not path.exists():
        return False
    try:
        torch, _, _ = require_torch()
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = read_manifest(project.MANIFEST_PATH)
        expected_train = len(rows_for_split(rows, "train", args.max_train_samples))
        expected_validation = len(
            rows_for_split(rows, "validation", args.max_validation_samples)
        )
        return bool(payload.get("training_complete", False)) and all(
            (
                int(payload.get("epochs_requested", -1)) == int(args.epochs),
                bool(payload.get("robust_training", True)) == (not args.no_robust_training),
                int(payload.get("seed", -1)) == int(args.seed),
                int(payload.get("train_samples", -1)) == expected_train,
                int(payload.get("validation_samples", -1)) == expected_validation,
                str(payload.get("source_commit", "")) == model_source_commit(name),
            )
        )
    except Exception:
        return False


def train_ours(args) -> None:
    quality_model = QualityModel.load(project.QUALITY_MODEL_PATH)
    print("\n[TRAIN START] ours: MobileNetV2 + ResNet50 + calibration + router", flush=True)
    train_model(
        project.MANIFEST_PATH,
        project.DATASET_ROOT,
        quality_model,
        project.APP_CONFIG.wavelet,
        TrainConfig(
            model_name="mobilenet_v2",
            batch_size=project.LIGHT_BATCH_SIZE,
            epochs=args.epochs,
            learning_rate=project.LIGHT_LEARNING_RATE,
            seed=args.seed,
            preprocessing_variant="adaptive",
        ),
        project.LIGHT_CHECKPOINT,
    )
    train_model(
        project.MANIFEST_PATH,
        project.DATASET_ROOT,
        quality_model,
        project.APP_CONFIG.wavelet,
        TrainConfig(
            model_name="resnet50",
            batch_size=project.HEAVY_BATCH_SIZE,
            epochs=args.epochs,
            learning_rate=project.HEAVY_LEARNING_RATE,
            seed=args.seed,
            preprocessing_variant="adaptive",
        ),
        project.HEAVY_CHECKPOINT,
    )
    light = ModelRunner(project.LIGHT_CHECKPOINT, args.device)
    heavy = ModelRunner(project.HEAVY_CHECKPOINT, args.device)
    collect_logits(
        project.MANIFEST_PATH,
        project.DATASET_ROOT,
        "calibration",
        light,
        heavy,
        quality_model,
        project.APP_CONFIG.wavelet,
        project.CALIBRATION_LOGITS_PATH,
    )
    frame = pd.read_csv(project.CALIBRATION_LOGITS_PATH)
    calibrator = TemperatureCalibrator(
        "quality_conditional", minimum_group_size=project.MINIMUM_CALIBRATION_GROUP_SIZE
    )
    for model_name in ("light", "heavy"):
        calibrator.fit(
            model_name,
            frame[f"{model_name}_logit"],
            frame["label"],
            frame["quality_grade"],
        )
        frame[f"{model_name}_probability"] = [
            calibrator.transform(model_name, logit, grade)
            for logit, grade in zip(
                frame[f"{model_name}_logit"], frame["quality_grade"]
            )
        ]
    calibrator.save(project.CALIBRATOR_PATH)
    router, metrics = tune_router(
        frame,
        base=project.APP_CONFIG.router,
        min_sensitivity=project.MINIMUM_ROUTER_SENSITIVITY,
    )
    save_config_module(
        project.FROZEN_CONFIG_PATH,
        AppConfig(
            quality=project.APP_CONFIG.quality,
            wavelet=project.APP_CONFIG.wavelet,
            router=router,
        ),
    )
    print(f"[TRAIN END] ours: router calibration metrics={metrics}", flush=True)


def load_baseline_checkpoint(name: str, args):
    torch, _, _ = require_torch()
    path = baseline_checkpoint_path(name, args)
    if not path.exists():
        raise FileNotFoundError(f"Missing trained comparison checkpoint: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = build_baseline_model(
        name,
        args.source_root,
        pretrained=None,
        allow_random_init=True,
        allow_slow_medmamba_fallback=args.allow_slow_medmamba_fallback,
        warn_random_init=False,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, payload


def parameter_count(model) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def result_row(
    model: str,
    condition: str,
    labels: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
    params: int,
    latency_ms: float,
    temperature: float | None,
    heavy_call_rate: float | None = None,
) -> dict[str, object]:
    metrics = classification_metrics(labels, probabilities, threshold)
    return {
        "model": model,
        "condition": condition,
        **metrics,
        "threshold": float(threshold),
        "temperature": temperature,
        "parameters_m": params / 1_000_000,
        "latency_ms_per_image": float(latency_ms),
        "heavy_call_rate": heavy_call_rate,
    }


def evaluate_baseline(name: str, args, conditions: Sequence[Condition]) -> list[dict[str, object]]:
    torch, DataLoader, _ = require_torch()
    device = resolve_device(args.device)
    model, _payload = load_baseline_checkpoint(name, args)
    model.to(device).eval()
    use_amp = bool(args.amp and device.type == "cuda" and BASELINE_SPECS[name].amp_safe)
    params = parameter_count(model)
    rows = read_manifest(project.MANIFEST_PATH)
    calibration_rows = rows_for_split(rows, "calibration", args.max_samples)
    test_rows = rows_for_split(rows, "test", args.max_samples)
    Dataset = make_baseline_dataset_class()
    calibration_dataset = Dataset(
        calibration_rows,
        project.DATASET_ROOT,
        BASELINE_SPECS[name],
        args.seed,
        condition=Condition("clean"),
    )
    evaluation_batch_size = args.eval_batch_size
    if name == "medmamba" and not medmamba_native_scan_available():
        evaluation_batch_size = min(args.eval_batch_size, 4)
    if name == "cxformer_s":
        evaluation_batch_size = min(args.eval_batch_size, 8)
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    cal_labels, cal_logits, _ = predict_logits(model, calibration_loader, device, use_amp)
    if not np.isfinite(np.asarray(cal_logits, dtype=np.float64)).all():
        raise FloatingPointError(f"{name} produced non-finite calibration logits")
    temperature = fit_temperature(cal_logits, cal_labels)
    print(f"[CALIBRATION] {name}: global temperature={temperature:.6f}", flush=True)
    results = []
    for condition in conditions:
        dataset = Dataset(
            test_rows,
            project.DATASET_ROOT,
            BASELINE_SPECS[name],
            args.seed,
            condition=condition,
        )
        loader = DataLoader(
            dataset,
            batch_size=evaluation_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
        labels, logits, latency_ms = predict_logits(model, loader, device, use_amp)
        if not np.isfinite(np.asarray(logits, dtype=np.float64)).all():
            raise FloatingPointError(
                f"{name} produced non-finite test logits for {condition.name}"
            )
        probabilities = sigmoid(np.asarray(logits) / temperature).tolist()
        row = result_row(
            name,
            condition.name,
            labels,
            probabilities,
            args.threshold,
            params,
            latency_ms,
            temperature,
        )
        results.append(row)
        print(
            f"[EVAL] {name:10s} {condition.name:24s} "
            f"AUROC={row['auroc']:.4f} AUPRC={row['auprc']:.4f} F1={row['f1']:.4f}",
            flush=True,
        )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def check_ours_artifacts() -> None:
    required = (
        project.LIGHT_CHECKPOINT,
        project.HEAVY_CHECKPOINT,
        project.QUALITY_MODEL_PATH,
        project.CALIBRATOR_PATH,
        project.FROZEN_CONFIG_PATH,
    )
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "The proposed method is not fully trained/calibrated. Missing: " + ", ".join(missing)
        )


def evaluate_ours(args, conditions: Sequence[Condition]) -> list[dict[str, object]]:
    check_ours_artifacts()
    torch, _, _ = require_torch()
    light = ModelRunner(project.LIGHT_CHECKPOINT, args.device)
    heavy = ModelRunner(project.HEAVY_CHECKPOINT, args.device)
    cascade = CascadeInference(
        light,
        heavy,
        QualityModel.load(project.QUALITY_MODEL_PATH),
        TemperatureCalibrator.load(project.CALIBRATOR_PATH),
        project.active_config().wavelet,
        project.active_config().router,
    )
    params = parameter_count(light.model) + parameter_count(heavy.model)
    rows = rows_for_split(read_manifest(project.MANIFEST_PATH), "test", args.max_samples)
    results = []
    for condition in conditions:
        labels: list[int] = []
        probabilities: list[float] = []
        latencies: list[float] = []
        heavy_calls: list[bool] = []
        for index, row in enumerate(rows):
            sample_id = row.get("sample_id") or f"test:{index}"
            with Image.open(project.DATASET_ROOT / row["image_path"]) as source:
                image = source.convert("L")
            if condition.degradation is not None:
                rng = np.random.default_rng(stable_seed(args.seed, sample_id, condition.name))
                image = apply_degradation(
                    image, condition.degradation, condition.severity, rng
                )
            result = cascade.predict(image, "full")
            labels.append(int(row["label"]))
            probabilities.append(float(result.probability))
            latencies.append(float(result.latency_ms["total"]))
            heavy_calls.append(bool(result.routing.heavy_called))
        row_result = result_row(
            "ours",
            condition.name,
            labels,
            probabilities,
            project.active_config().router.classification_threshold,
            params,
            float(np.mean(latencies)),
            None,
            float(np.mean(heavy_calls)),
        )
        results.append(row_result)
        print(
            f"[EVAL] {'ours':10s} {condition.name:24s} "
            f"AUROC={row_result['auroc']:.4f} AUPRC={row_result['auprc']:.4f} "
            f"F1={row_result['f1']:.4f}",
            flush=True,
        )
    del cascade, light, heavy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def add_degraded_macro(frame: pd.DataFrame) -> pd.DataFrame:
    degraded = frame[frame["condition"] != "clean"]
    if degraded.empty:
        return frame
    numeric = [
        "auroc",
        "auprc",
        "sensitivity",
        "specificity",
        "f1",
        "n",
        "threshold",
        "temperature",
        "parameters_m",
        "latency_ms_per_image",
        "heavy_call_rate",
    ]
    rows = []
    for model_name, group in degraded.groupby("model", sort=False):
        row: dict[str, object] = {"model": model_name, "condition": "DEGRADED_MACRO"}
        for column in numeric:
            values = pd.to_numeric(group[column], errors="coerce")
            row[column] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)


def print_results(frame: pd.DataFrame) -> None:
    display = frame.copy()
    display = display[
        [
            "model",
            "condition",
            "auroc",
            "auprc",
            "sensitivity",
            "specificity",
            "f1",
            "n",
        ]
    ]
    display.columns = [
        "Model",
        "Condition",
        "AUROC",
        "AUPRC",
        "Sensitivity",
        "Specificity",
        "F1",
        "N",
    ]
    print("\n=== FIVE-METHOD CLASSIFICATION COMPARISON ===", flush=True)
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.expand_frame_repr",
        False,
    ):
        print(display.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def write_run_metadata(args, models: Sequence[str], conditions: Sequence[Condition]) -> None:
    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(project.MANIFEST_PATH),
        "dataset_root": str(project.DATASET_ROOT),
        "models": list(models),
        "conditions": [condition.name for condition in conditions],
        "seed": args.seed,
        "threshold_for_baselines": args.threshold,
        "official_source_commits": {
            **{key: value["commit"] for key, value in REPOSITORIES.items()},
            "cxformer_s": model_source_commit("cxformer_s"),
        },
        "notes": {
            "split": "Fixed patient-level train/validation/calibration/test manifest",
            "calibration": "Global temperature scaling for baselines; quality-conditional scaling for ours",
            "test_threshold": "Frozen before test evaluation",
            "degraded_macro": "Unweighted arithmetic mean across requested non-clean conditions",
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified five-method chest-X-ray classification comparison"
    )
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    parser.add_argument("--models", default="all", help="all or comma-separated model names")
    parser.add_argument("--conditions", default="all", help="all, clean, or e.g. noise:heavy,blur")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--fetch-sources",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically fetch missing official source repositories (default: enabled)",
    )
    parser.add_argument(
        "--download-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically download EVA-X and CXFormer-S chest-X-ray checkpoints (default: enabled)",
    )
    parser.add_argument("--medvitv2-pretrained", default="")
    parser.add_argument("--medmamba-pretrained", default="")
    parser.add_argument("--eva-x-pretrained", default="")
    parser.add_argument("--cxformer-s-pretrained", default="")
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Allow EVA-X/CXFormer-S without chest-X-ray pretraining; not reportable",
    )
    parser.add_argument("--allow-slow-medmamba-fallback", action="store_true")
    parser.add_argument("--retrain-ours", action="store_true")
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain all requested methods even when complete checkpoints already exist",
    )
    parser.add_argument("--no-robust-training", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--progress-every-batches",
        type=int,
        default=50,
        help="Print in-epoch training progress every N batches (default: 50)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=project.SEED)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must lie strictly between 0 and 1")
    if args.progress_every_batches < 1:
        raise ValueError("--progress-every-batches must be at least 1")
    models = parse_models(args.models)
    conditions = parse_conditions(args.conditions)
    ensure_manifest_and_quality_model(args)
    if args.fetch_sources:
        fetch_sources(args.source_root)
    for name in models:
        if name in REPOSITORIES:
            verify_source(args.source_root, name)
    pretrained = resolve_pretrained_paths(args)
    seed_everything(args.seed)
    if args.stage in {"train", "all"}:
        if "ours" in models:
            if args.force_retrain or args.retrain_ours or not ours_artifacts_complete():
                train_ours(args)
            else:
                print("[TRAIN] ours: complete artifacts found; reusing them", flush=True)
        for name in models:
            if name != "ours":
                checkpoint = baseline_checkpoint_path(name, args)
                if baseline_checkpoint_complete(name, args) and not args.force_retrain:
                    print(f"[TRAIN] {name}: checkpoint found; reusing {checkpoint}", flush=True)
                else:
                    if checkpoint.exists() and not args.force_retrain:
                        print(
                            f"[TRAIN] {name}: checkpoint is incomplete; resuming it",
                            flush=True,
                        )
                    train_baseline(name, args, pretrained[name])
    if args.stage == "train":
        return
    all_results: list[dict[str, object]] = []
    failures: list[tuple[str, str]] = []
    for name in models:
        try:
            if name == "ours":
                all_results.extend(evaluate_ours(args, conditions))
            else:
                all_results.extend(evaluate_baseline(name, args, conditions))
        except Exception as exc:
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if not all_results:
        raise RuntimeError("No model produced evaluation results")
    frame = add_degraded_macro(pd.DataFrame(all_results))
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_root / "classification_comparison.csv"
    frame.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
    write_run_metadata(args, models, conditions)
    print_results(frame)
    print(f"\nResults CSV: {output_csv}", flush=True)
    if failures:
        print("\nModels that did not complete:", file=sys.stderr)
        for model_name, message in failures:
            print(f"  - {model_name}: {message}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
