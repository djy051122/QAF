from __future__ import annotations

import argparse

import pandas as pd

import project_config as config
from cxr_cascade.calibration import TemperatureCalibrator
from cxr_cascade.inference import ModelRunner
from cxr_cascade.quality import QualityModel
from cxr_cascade.quality_control_ablation import (
    evaluate_quality_control_ablation,
    save_quality_control_outputs,
    summarize_quality_control,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 2: ablate true, shuffled, and absent quality control."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shuffle-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default=str(config.QUALITY_CONTROL_ABLATION_OUTPUT_ROOT))
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = config.active_config()
    detail = evaluate_quality_control_ablation(
        config.MANIFEST_PATH,
        config.DATASET_ROOT,
        args.split,
        QualityModel.load(config.QUALITY_MODEL_PATH),
        TemperatureCalibrator.load(config.CALIBRATOR_PATH),
        ModelRunner(config.LIGHT_CHECKPOINT, args.device),
        ModelRunner(config.HEAVY_CHECKPOINT, args.device),
        app_config.wavelet,
        app_config.router,
        seed=args.seed,
        max_samples=args.max_samples,
        shuffle_repeats=args.shuffle_repeats,
    )
    paths = save_quality_control_outputs(
        detail, args.output_root, make_figures=not args.no_figures
    )
    _, _, overall = summarize_quality_control(detail)
    display = overall.groupby("control_variant", as_index=False).mean(numeric_only=True)
    columns = [
        "control_variant", "auroc", "auprc", "sensitivity", "specificity", "f1",
        "ece", "nll", "mean_latency_ms", "heavy_call_rate", "fusion_rate",
    ]
    print("\n=== EXPERIMENT 2: QUALITY-CONTROL ABLATION (DEGRADED MACRO) ===")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(display[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
