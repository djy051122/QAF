from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

import project_config as config
from cxr_cascade.calibration import TemperatureCalibrator
from cxr_cascade.inference import ModelRunner
from cxr_cascade.quality import QualityModel
from cxr_cascade.reporting import print_script_result
from cxr_cascade.resource_allocation_evaluation import (
    DEGRADATIONS,
    DEGRADATION_LEVELS,
)
from cxr_cascade.routing_strategy_evaluation import (
    apply_routing_strategies,
    collect_component_records,
    ensure_light_routing_scores,
    fit_routing_thresholds,
    patient_cluster_bootstrap,
    save_routing_outputs,
    save_routing_thresholds,
    summarize_routing_strategies,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("在统一轻/重模型和退化输入上比较 TinySaver、VRM式置信度门控和本文动态路由")
    )
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--calibration-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--vrm-target-heavy-rate",
        type=float,
        default=None,
        help="默认与Ours在calibration集上的重模型调用率匹配；例如0.20表示VRM论文式20%%升级率",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.OUTPUT_ROOT / "routing_strategy_comparison",
    )
    parser.add_argument(
        "--degradations",
        nargs="+",
        choices=DEGRADATIONS,
        default=list(DEGRADATIONS),
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=DEGRADATION_LEVELS,
        default=list(DEGRADATION_LEVELS),
    )
    parser.add_argument("--include-original", action="store_true")
    args = parser.parse_args()
    if args.vrm_target_heavy_rate is not None and not 0.0 <= args.vrm_target_heavy_rate <= 1.0:
        parser.error("--vrm-target-heavy-rate must lie in [0, 1]")

    started = time.perf_counter()
    output_root = args.output_root.resolve()
    calibration_path = output_root / "calibration_components.csv"
    test_path = output_root / "test_components.csv"
    thresholds_path = output_root / "routing_thresholds.json"

    app = config.active_config()
    light = ModelRunner(config.LIGHT_CHECKPOINT, device=args.device)
    heavy = ModelRunner(config.HEAVY_CHECKPOINT, device=args.device)
    quality = QualityModel.load(config.QUALITY_MODEL_PATH)
    calibrator = TemperatureCalibrator.load(config.CALIBRATOR_PATH)

    calibration = collect_component_records(
        manifest_path=config.MANIFEST_PATH,
        data_root=config.DATASET_ROOT,
        split="calibration",
        sample_size=args.calibration_size,
        light=light,
        heavy=heavy,
        quality_model=quality,
        calibrator=calibrator,
        app_config=app,
        output_csv=calibration_path,
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        degradations=args.degradations,
        levels=args.levels,
        include_original=args.include_original,
        resume=args.resume,
    )
    calibration = ensure_light_routing_scores(
        calibration,
        manifest_path=config.MANIFEST_PATH,
        data_root=config.DATASET_ROOT,
        split="calibration",
        light=light,
        quality_model=quality,
        calibrator=calibrator,
        app_config=app,
        output_csv=calibration_path,
        seed=args.seed,
    )
    thresholds = fit_routing_thresholds(
        calibration,
        app,
        vrm_target_heavy_rate=args.vrm_target_heavy_rate,
    )
    save_routing_thresholds(thresholds, thresholds_path)

    test_components = collect_component_records(
        manifest_path=config.MANIFEST_PATH,
        data_root=config.DATASET_ROOT,
        split="test",
        sample_size=args.sample_size,
        light=light,
        heavy=heavy,
        quality_model=quality,
        calibrator=calibrator,
        app_config=app,
        output_csv=test_path,
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        degradations=args.degradations,
        levels=args.levels,
        include_original=args.include_original,
        resume=args.resume,
    )
    test_components = ensure_light_routing_scores(
        test_components,
        manifest_path=config.MANIFEST_PATH,
        data_root=config.DATASET_ROOT,
        split="test",
        light=light,
        quality_model=quality,
        calibrator=calibrator,
        app_config=app,
        output_csv=test_path,
        seed=args.seed,
    )
    detail = apply_routing_strategies(test_components, thresholds, app)
    overall, scenarios, severity = summarize_routing_strategies(detail)
    confidence_intervals, paired = patient_cluster_bootstrap(
        detail,
        overall,
        repeats=args.bootstrap_repeats,
        seed=args.seed,
    )
    paths = save_routing_outputs(
        detail=detail,
        overall=overall,
        scenarios=scenarios,
        severity=severity,
        confidence_intervals=confidence_intervals,
        paired=paired,
        output_root=output_root,
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.max_rows",
        None,
        "display.width",
        220,
        "display.expand_frame_repr",
        False,
    ):
        scenario_count = int(overall["n_scenarios"].max())
        print(f"\n主结果（{scenario_count}个退化场景宏平均；耗时和调用率为全部记录平均）：")
        print(overall.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        print("\n各退化强度下的路径比例：")
        print(severity.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print_script_result(
        "17_compare_routing_strategies.py",
        {
            "device": light.device,
            "test_images": int(test_components["sample_id"].nunique()),
            "test_patients": int(test_components["patient_id"].nunique()),
            "scenario_count": int(
                test_components[["degradation", "degradation_level"]].drop_duplicates().shape[0]
            ),
            "thresholds": thresholds_path,
            "main_table": paths["overall"],
            "route_proportions": paths["severity"],
            "model_call_rates_by_severity": paths["call_rates"],
            "confidence_intervals": paths["confidence_intervals"],
            "paired_bootstrap": paths["paired"],
            "wall_clock_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
