from __future__ import annotations

"""Evaluate clean-source and BDCXR-3257 external generalization.

Protocol
--------
* No synthetic degradation is added to either evaluation domain.
* Ours uses the frozen quality model, adaptive wavelet gate, calibrated
  MobileNetV2/ResNet50 cascade, and frozen source-domain threshold.
* MedViTV2, MedMamba, and EVA-X receive the untouched decoded image and only
  their model-specific resize/pad/normalization pipeline from
  ``14_compare_classification_models.py``.  They never use wavelet denoising.
* All weights, temperature parameters, and thresholds are fitted on the source
  dataset only.  BDCXR-3257 is evaluation-only.
* The default BDCXR evaluation cohort is a deterministic 700-image subset:
  350 Normal and 350 Pneumonia, with bacterial/viral cases sampled
  proportionally inside the Pneumonia class.
* The paper table contains AUROC, AUPRC, sensitivity, specificity, F1, and ECE.

Full run:
    python 18_evaluate_generalization.py

Small pipeline check:
    python 18_evaluate_generalization.py --models ours --max-samples 4
"""

import argparse
import csv
import importlib.util
import json
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
from PIL import Image

import project_config as project
from cxr_cascade.calibration import TemperatureCalibrator, fit_temperature, sigmoid
from cxr_cascade.data import read_manifest
from cxr_cascade.inference import CascadeInference, ModelRunner
from cxr_cascade.metrics import calibration_metrics
from cxr_cascade.quality import QualityModel

PROJECT_ROOT = Path(__file__).resolve().parent
COMPARISON_SCRIPT = PROJECT_ROOT / "14_compare_classification_models.py"
DEFAULT_SOURCE_ROOT = PROJECT_ROOT.parent / ".reference_repos"
DEFAULT_COMPARISON_OUTPUT_ROOT = project.OUTPUT_ROOT / "classification_comparison"

MODEL_ORDER = ("medvitv2", "medmamba", "eva_x", "ours")
DISPLAY_NAMES = {
    "medvitv2": "MedViTv2",
    "medmamba": "MedMamba",
    "eva_x": "EVA-X",
    "ours": "Ours",
}
DATASET_ORDER = ("Source test", "BDCXR-3257")


def load_comparison_module() -> ModuleType:
    """Reuse the already-audited official baseline implementations."""

    spec = importlib.util.spec_from_file_location(
        "classification_comparison_for_generalization", COMPARISON_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import comparison implementation: {COMPARISON_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


comparison = load_comparison_module()


def parse_models(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(MODEL_ORDER)
    aliases = {
        "medvit-v2": "medvitv2",
        "medvit2": "medvitv2",
        "eva-x": "eva_x",
    }
    models = [
        aliases.get(token.strip().lower(), token.strip().lower())
        for token in value.split(",")
        if token.strip()
    ]
    unknown = sorted(set(models) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}; choose from {MODEL_ORDER}")
    return [name for name in MODEL_ORDER if name in models]


def validate_rows(
    rows: Sequence[dict[str, str]],
    root: Path,
    expected_split: str,
    dataset_name: str,
) -> None:
    selected = [row for row in rows if row.get("split") == expected_split]
    if not selected:
        raise ValueError(f"{dataset_name} has no rows with split={expected_split!r}")
    labels = {int(row["label"]) for row in selected}
    if labels != {0, 1}:
        raise ValueError(f"{dataset_name} must contain both binary labels; found {labels}")
    missing = [str(root / row["image_path"]) for row in selected if not (root / row["image_path"]).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{dataset_name} contains {len(missing)} missing images; first: {missing[0]}"
        )


def selected_rows(
    manifest: Path,
    root: Path,
    split: str,
    dataset_name: str,
    max_samples: int | None,
) -> list[dict[str, str]]:
    rows = read_manifest(manifest)
    validate_rows(rows, root, split, dataset_name)
    return comparison.rows_for_split(rows, split, max_samples)


def _sample_rows(
    rows: Sequence[dict[str, str]], size: int, seed: int, group_name: str
) -> list[dict[str, str]]:
    if size > len(rows):
        raise ValueError(
            f"Cannot sample {size} rows from {group_name}; only {len(rows)} are available"
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("sample_id", ""),
            row.get("image_path", ""),
        ),
    )
    if size == len(ordered):
        return ordered
    rng = np.random.default_rng(comparison.stable_seed(seed, "bdcxr700", group_name))
    indices = sorted(rng.choice(len(ordered), size=size, replace=False).tolist())
    return [ordered[index] for index in indices]


def fixed_external_subset(
    rows: Sequence[dict[str, str]], size: int = 700, seed: int = project.SEED
) -> list[dict[str, str]]:
    """Build the frozen balanced BDCXR cohort used by every method."""

    if size < 2 or size > len(rows):
        raise ValueError(f"External subset size must lie in [2, {len(rows)}]")
    negative_target = size // 2
    positive_target = size - negative_target
    negatives = [row for row in rows if int(row["label"]) == 0]
    positives = [row for row in rows if int(row["label"]) == 1]
    selected = _sample_rows(negatives, negative_target, seed, "Normal")

    positive_groups: dict[str, list[dict[str, str]]] = {}
    for row in positives:
        positive_groups.setdefault(row.get("original_class", "PNEUMONIA"), []).append(row)
    group_names = sorted(positive_groups)
    raw_targets = {
        name: positive_target * len(positive_groups[name]) / len(positives)
        for name in group_names
    }
    targets = {name: int(np.floor(raw_targets[name])) for name in group_names}
    remainder = positive_target - sum(targets.values())
    remainder_order = sorted(
        group_names,
        key=lambda name: (raw_targets[name] - targets[name], name),
        reverse=True,
    )
    for name in remainder_order[:remainder]:
        targets[name] += 1
    for name in group_names:
        selected.extend(_sample_rows(positive_groups[name], targets[name], seed, name))

    # A final deterministic shuffle avoids presenting one class at a time while
    # keeping the selected cohort identical across all four methods and reruns.
    rng = np.random.default_rng(comparison.stable_seed(seed, "bdcxr700", "final_order"))
    order = rng.permutation(len(selected))
    return [selected[int(index)] for index in order]


def write_subset_manifest(rows: Sequence[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def six_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
    ece_bins: int,
) -> dict[str, float | int]:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.shape != p.shape or y.ndim != 1 or y.size == 0:
        raise ValueError("Labels and probabilities must be non-empty aligned vectors")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both labels are required to compute all six metrics")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    classification = comparison.classification_metrics(y, p, threshold)
    calibration = calibration_metrics(y, p, bins=ece_bins)
    return {
        "AUROC": float(classification["auroc"]),
        "AUPRC": float(classification["auprc"]),
        "sensitivity": float(classification["sensitivity"]),
        "specificity": float(classification["specificity"]),
        "F1": float(classification["f1"]),
        "ECE": float(calibration["ece"]),
        "n": len(y),
        "positive_n": int(y.sum()),
        "negative_n": int(len(y) - y.sum()),
    }


def predict_baseline_loader(model, loader, device, use_amp: bool):
    torch, _, _ = comparison.require_torch()
    labels: list[int] = []
    logits: list[float] = []
    sample_ids: list[str] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for images, targets, batch_ids in loader:
            images = images.to(device, non_blocking=True)
            with comparison.autocast_context(
                torch, device, use_amp and device.type == "cuda"
            ):
                output = model(images).reshape(-1)
            logits.extend(output.detach().float().cpu().tolist())
            labels.extend(np.asarray(targets).astype(int).tolist())
            sample_ids.extend(str(value) for value in batch_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency_ms = 1000.0 * (time.perf_counter() - started) / max(len(labels), 1)
    return labels, logits, sample_ids, latency_ms


def baseline_batch_size(name: str, requested: int) -> int:
    if name == "medmamba" and not comparison.medmamba_native_scan_available():
        return min(requested, 4)
    return requested


def evaluate_baseline(
    name: str,
    args: argparse.Namespace,
    domains: Sequence[tuple[str, Path, list[dict[str, str]]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    torch, DataLoader, _ = comparison.require_torch()
    device = comparison.resolve_device(args.device)
    loader_args = argparse.Namespace(
        output_root=args.comparison_output_root,
        source_root=args.source_root,
        allow_slow_medmamba_fallback=args.allow_slow_medmamba_fallback,
    )
    model, checkpoint = comparison.load_baseline_checkpoint(name, loader_args)
    model.to(device).eval()
    spec = comparison.BASELINE_SPECS[name]
    use_amp = bool(args.amp and device.type == "cuda" and spec.amp_safe)
    Dataset = comparison.make_baseline_dataset_class()
    batch_size = baseline_batch_size(name, args.eval_batch_size)

    # Temperature is fitted once on the clean source calibration split.  No
    # external labels are used for probability calibration.
    calibration_rows = selected_rows(
        project.MANIFEST_PATH,
        project.DATASET_ROOT,
        "calibration",
        "Source calibration",
        args.max_calibration_samples,
    )
    calibration_dataset = Dataset(
        calibration_rows,
        project.DATASET_ROOT,
        spec,
        args.seed,
        training=False,
        augment_robustness=False,
        condition=comparison.Condition("clean"),
    )
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )
    cal_labels, cal_logits, _, _ = predict_baseline_loader(
        model, calibration_loader, device, use_amp
    )
    temperature = fit_temperature(cal_logits, cal_labels)
    print(
        f"[SOURCE CALIBRATION] {DISPLAY_NAMES[name]} temperature={temperature:.6f}",
        flush=True,
    )

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for dataset_name, data_root, rows in domains:
        dataset = Dataset(
            rows,
            data_root,
            spec,
            args.seed,
            training=False,
            augment_robustness=False,
            condition=comparison.Condition("clean"),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and device.type == "cuda",
        )
        labels, logits, sample_ids, latency_ms = predict_baseline_loader(
            model, loader, device, use_amp
        )
        probabilities = np.asarray(sigmoid(np.asarray(logits) / temperature), dtype=np.float64)
        metrics = six_metrics(
            labels, probabilities, args.baseline_threshold, args.ece_bins
        )
        summary_rows.append(
            {
                "dataset": dataset_name,
                "method": DISPLAY_NAMES[name],
                **metrics,
                "threshold": float(args.baseline_threshold),
                "temperature": float(temperature),
                "adaptive_denoising_rate": 0.0,
                "latency_ms_per_image": float(latency_ms),
            }
        )
        for sample_id, label, logit, probability in zip(
            sample_ids, labels, logits, probabilities
        ):
            prediction_rows.append(
                {
                    "dataset": dataset_name,
                    "method": DISPLAY_NAMES[name],
                    "sample_id": sample_id,
                    "label": int(label),
                    "logit": float(logit),
                    "probability": float(probability),
                    "prediction": int(probability >= args.baseline_threshold),
                    "threshold": float(args.baseline_threshold),
                    "temperature": float(temperature),
                    "adaptive_denoising_applied": False,
                    "quality_grade": "",
                    "route": "",
                }
            )
        print(
            f"[EVAL] {DISPLAY_NAMES[name]:10s} {dataset_name:12s} "
            f"AUROC={metrics['AUROC']:.4f} AUPRC={metrics['AUPRC']:.4f} "
            f"Sensitivity={metrics['sensitivity']:.4f} "
            f"Specificity={metrics['specificity']:.4f} "
            f"F1={metrics['F1']:.4f} ECE={metrics['ECE']:.4f}",
            flush=True,
        )

    metadata = {
        "checkpoint": str(
            args.comparison_output_root / "checkpoints" / f"{name}.pt"
        ),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_validation_auroc": checkpoint.get("best_validation_auroc"),
        "source_commit": checkpoint.get("source_commit"),
        "input_protocol": asdict(spec),
        "synthetic_degradation": False,
        "wavelet_denoising": False,
        "temperature_source": "source calibration split only",
        "temperature": float(temperature),
        "threshold": float(args.baseline_threshold),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary_rows, prediction_rows, metadata


def evaluate_ours(
    args: argparse.Namespace,
    domains: Sequence[tuple[str, Path, list[dict[str, str]]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    comparison.check_ours_artifacts()
    torch, _, _ = comparison.require_torch()
    light = ModelRunner(project.LIGHT_CHECKPOINT, args.device)
    heavy = ModelRunner(project.HEAVY_CHECKPOINT, args.device)
    quality_model = QualityModel.load(project.QUALITY_MODEL_PATH)
    calibrator = TemperatureCalibrator.load(project.CALIBRATOR_PATH)
    config = project.active_config()
    cascade = CascadeInference(
        light,
        heavy,
        quality_model,
        calibrator,
        config.wavelet,
        config.router,
    )
    threshold = float(config.router.classification_threshold)
    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for dataset_name, data_root, rows in domains:
        labels: list[int] = []
        probabilities: list[float] = []
        latencies: list[float] = []
        denoising_flags: list[bool] = []
        for index, row in enumerate(rows, start=1):
            with Image.open(data_root / row["image_path"]) as source:
                # This is the decoded original dataset image.  No synthetic
                # noise or degradation is inserted before adaptive inference.
                image = source.convert("L")
            result = cascade.predict(image, "full")
            label = int(row["label"])
            probability = float(result.probability)
            denoised = bool(result.quality.wavelet_enabled)
            labels.append(label)
            probabilities.append(probability)
            latencies.append(float(result.latency_ms["total"]))
            denoising_flags.append(denoised)
            prediction_rows.append(
                {
                    "dataset": dataset_name,
                    "method": DISPLAY_NAMES["ours"],
                    "sample_id": row.get("sample_id") or f"{dataset_name}:{index}",
                    "label": label,
                    "logit": "",
                    "probability": probability,
                    "prediction": int(probability >= threshold),
                    "threshold": threshold,
                    "temperature": "quality-conditional",
                    "adaptive_denoising_applied": denoised,
                    "quality_grade": result.quality.grade.value,
                    "quality_score": float(result.quality.score),
                    "primary_degradation": result.quality.primary_degradation,
                    "route": result.routing.route.value,
                    "heavy_called": bool(result.routing.heavy_called),
                }
            )
            if args.progress_every and (
                index == 1 or index % args.progress_every == 0 or index == len(rows)
            ):
                print(
                    f"[OURS PROGRESS] {dataset_name} {index}/{len(rows)} "
                    f"adaptive_denoising={np.mean(denoising_flags):.3f}",
                    flush=True,
                )

        metrics = six_metrics(labels, probabilities, threshold, args.ece_bins)
        denoising_rate = float(np.mean(denoising_flags))
        summary_rows.append(
            {
                "dataset": dataset_name,
                "method": DISPLAY_NAMES["ours"],
                **metrics,
                "threshold": threshold,
                "temperature": "quality-conditional",
                "adaptive_denoising_rate": denoising_rate,
                "latency_ms_per_image": float(np.mean(latencies)),
            }
        )
        print(
            f"[EVAL] {'Ours':10s} {dataset_name:12s} "
            f"AUROC={metrics['AUROC']:.4f} AUPRC={metrics['AUPRC']:.4f} "
            f"Sensitivity={metrics['sensitivity']:.4f} "
            f"Specificity={metrics['specificity']:.4f} "
            f"F1={metrics['F1']:.4f} ECE={metrics['ECE']:.4f} "
            f"AdaptiveDenoising={denoising_rate:.4f}",
            flush=True,
        )

    metadata = {
        "light_checkpoint": str(project.LIGHT_CHECKPOINT),
        "heavy_checkpoint": str(project.HEAVY_CHECKPOINT),
        "quality_model": str(project.QUALITY_MODEL_PATH),
        "calibrator": str(project.CALIBRATOR_PATH),
        "frozen_config": str(project.FROZEN_CONFIG_PATH),
        "input_protocol": (
            "decoded original image -> frozen quality assessment -> adaptive wavelet gate "
            "-> letterbox/ImageNet normalization -> frozen calibrated cascade"
        ),
        "synthetic_degradation": False,
        "wavelet_denoising": "adaptive gate only",
        "calibration_source": "source calibration split only",
        "threshold": threshold,
    }
    del cascade, light, heavy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary_rows, prediction_rows, metadata


def ordered_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame["dataset"] = pd.Categorical(
        frame["dataset"], categories=DATASET_ORDER, ordered=True
    )
    frame["method"] = pd.Categorical(
        frame["method"],
        categories=[DISPLAY_NAMES[name] for name in MODEL_ORDER],
        ordered=True,
    )
    return frame.sort_values(["dataset", "method"]).reset_index(drop=True)


def write_outputs(
    args: argparse.Namespace,
    summary_rows: list[dict[str, object]],
    prediction_rows: list[dict[str, object]],
    model_metadata: dict[str, object],
    failures: list[tuple[str, str]],
    domains: Sequence[tuple[str, Path, list[dict[str, str]]]],
) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary = ordered_frame(summary_rows)
    summary_path = args.output_root / "generalization_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    paper_columns = [
        "dataset",
        "method",
        "AUROC",
        "AUPRC",
        "sensitivity",
        "specificity",
        "F1",
        "ECE",
    ]
    paper_table = summary[paper_columns].copy()
    paper_path = args.output_root / "table_generalization.csv"
    paper_table.to_csv(paper_path, index=False, encoding="utf-8-sig")

    predictions_path = args.output_root / "generalization_predictions.csv"
    pd.DataFrame(prediction_rows).to_csv(
        predictions_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL
    )

    metadata = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "protocol": {
            "external_dataset_role": "evaluation only",
            "synthetic_noise_or_degradation": False,
            "baseline_denoising": False,
            "ours_denoising": "frozen quality-gated adaptive wavelet denoising",
            "threshold_and_calibration": "frozen or fitted on source calibration only",
            "ece_bins": args.ece_bins,
            "external_sampling": (
                f"fixed seed={args.seed}; n={args.external_samples}; "
                "balanced Normal/Pneumonia; pneumonia subtypes proportional"
            ),
        },
        "domains": {
            name: {
                "root": str(root),
                "n": len(rows),
                "positive_n": sum(int(row["label"]) for row in rows),
                "negative_n": sum(1 - int(row["label"]) for row in rows),
            }
            for name, root, rows in domains
        },
        "models": model_metadata,
        "failures": [{"model": name, "error": message} for name, message in failures],
    }
    metadata_path = args.output_root / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== GENERALIZATION TABLE (probabilities in [0, 1]) ===", flush=True)
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 220,
        "display.expand_frame_repr", False,
    ):
        print(paper_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nPaper table: {paper_path}")
    print(f"Full summary: {summary_path}")
    print(f"Per-image predictions: {predictions_path}")
    print(f"Run metadata: {metadata_path}")
    subset_manifest_path = args.output_root / f"bdcxr_{args.external_samples}_manifest.csv"
    print(f"Fixed external subset: {subset_manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean source-test and BDCXR-3257 external generalization for "
            "MedViTv2, MedMamba, EVA-X, and Ours"
        )
    )
    parser.add_argument("--models", default="all")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--comparison-output-root",
        type=Path,
        default=DEFAULT_COMPARISON_OUTPUT_ROOT,
        help="Directory produced by 14_compare_classification_models.py",
    )
    parser.add_argument(
        "--external-manifest",
        type=Path,
        default=project.BDCXR_EXTERNAL_MANIFEST_PATH,
    )
    parser.add_argument(
        "--external-root", type=Path, default=project.BDCXR_BINARY_ROOT
    )
    parser.add_argument(
        "--output-root", type=Path, default=project.GENERALIZATION_OUTPUT_ROOT
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=project.SEED)
    parser.add_argument("--baseline-threshold", type=float, default=0.5)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--external-samples",
        type=int,
        default=700,
        help="Fixed balanced BDCXR cohort size before optional smoke-test limiting",
    )
    parser.add_argument("--max-calibration-samples", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pin-memory", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--allow-slow-medmamba-fallback", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.comparison_output_root = args.comparison_output_root.resolve()
    args.external_manifest = args.external_manifest.resolve()
    args.external_root = args.external_root.resolve()
    args.output_root = args.output_root.resolve()
    if not 0.0 < args.baseline_threshold < 1.0:
        raise ValueError("--baseline-threshold must lie strictly between 0 and 1")
    if args.ece_bins < 2:
        raise ValueError("--ece-bins must be at least 2")
    if args.eval_batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid DataLoader settings")
    if args.progress_every < 0:
        raise ValueError("--progress-every cannot be negative")

    models = parse_models(args.models)
    comparison.seed_everything(args.seed)
    source_test_rows = selected_rows(
        project.MANIFEST_PATH,
        project.DATASET_ROOT,
        "test",
        "Source test",
        args.max_samples,
    )
    all_external_rows = selected_rows(
        args.external_manifest,
        args.external_root,
        "external_test",
        "BDCXR-3257",
        None,
    )
    fixed_external_rows = fixed_external_subset(
        all_external_rows, size=args.external_samples, seed=args.seed
    )
    external_rows = fixed_external_rows
    if args.max_samples is not None:
        external_rows = comparison.rows_for_split(
            external_rows, "external_test", args.max_samples
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_subset_manifest(
        fixed_external_rows,
        args.output_root / f"bdcxr_{args.external_samples}_manifest.csv",
    )
    domains = (
        ("Source test", project.DATASET_ROOT, source_test_rows),
        ("BDCXR-3257", args.external_root, external_rows),
    )
    print(
        f"[DATA] Source test={len(source_test_rows)}; "
        f"BDCXR-3257={len(external_rows)}; synthetic degradation=OFF",
        flush=True,
    )

    summary_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    model_metadata: dict[str, object] = {}
    failures: list[tuple[str, str]] = []
    for name in models:
        try:
            if name == "ours":
                summary, predictions, metadata = evaluate_ours(args, domains)
            else:
                summary, predictions, metadata = evaluate_baseline(name, args, domains)
            summary_rows.extend(summary)
            prediction_rows.extend(predictions)
            model_metadata[DISPLAY_NAMES[name]] = metadata
        except Exception as exc:  # noqa: BLE001 - preserve other completed model results
            message = f"{type(exc).__name__}: {exc}"
            failures.append((DISPLAY_NAMES[name], message))
            print(f"[ERROR] {DISPLAY_NAMES[name]}: {message}", file=sys.stderr, flush=True)

    if not summary_rows:
        raise RuntimeError("No model produced generalization results")
    write_outputs(
        args,
        summary_rows,
        prediction_rows,
        model_metadata,
        failures,
        domains,
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
