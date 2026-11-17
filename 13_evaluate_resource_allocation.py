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
    evaluate_resource_allocation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "在固定测试图像上比较 M1-M6 六种模型与两阶段路由配置；"
            "默认测试五种退化的轻度、中度和重度输入，不包含原始图像"
        )
    )
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-root",
        default=None,
        help="可选的独立输出目录；冒烟测试时可用它避免覆盖正式结果",
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
    parser.add_argument(
        "--exclude-original",
        action="store_true",
        help="仅评估退化图像，不评估原始图像",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="在三种退化强度之外额外评估原始图像",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    app_config = config.active_config()
    light = ModelRunner(config.LIGHT_CHECKPOINT, device=args.device)
    heavy = ModelRunner(config.HEAVY_CHECKPOINT, device=args.device)
    if args.output_root:
        output_root = Path(args.output_root)
        selection_path = output_root / "resource_allocation_selection.csv"
        detail_path = output_root / "resource_allocation_details.csv"
        scenario_summary_path = output_root / "tables" / "resource_allocation_scenarios.csv"
        severity_summary_path = output_root / "tables" / "resource_allocation_by_severity.csv"
        level_comparison_path = output_root / "tables" / "resource_allocation_all_levels.csv"
    else:
        selection_path = config.RESOURCE_ALLOCATION_SELECTION_PATH
        detail_path = config.RESOURCE_ALLOCATION_DETAIL_PATH
        scenario_summary_path = config.RESOURCE_ALLOCATION_SCENARIO_SUMMARY_PATH
        severity_summary_path = config.RESOURCE_ALLOCATION_SEVERITY_SUMMARY_PATH
        level_comparison_path = config.RESOURCE_ALLOCATION_LEVEL_COMPARISON_PATH
    detail, scenarios, severity, level_comparison = evaluate_resource_allocation(
        manifest_path=config.MANIFEST_PATH,
        data_root=config.DATASET_ROOT,
        split="test",
        light=light,
        heavy=heavy,
        quality_model=QualityModel.load(config.QUALITY_MODEL_PATH),
        calibrator=TemperatureCalibrator.load(config.CALIBRATOR_PATH),
        app_config=app_config,
        selection_csv=selection_path,
        detail_csv=detail_path,
        scenario_summary_csv=scenario_summary_path,
        severity_summary_csv=severity_summary_path,
        level_comparison_csv=level_comparison_path,
        sample_size=args.sample_size,
        seed=args.seed,
        warmup_runs=args.warmup_runs,
        degradations=args.degradations,
        levels=args.levels,
        include_original=args.include_original and not args.exclude_original,
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.max_rows",
        None,
        "display.width",
        240,
        "display.expand_frame_repr",
        False,
    ):
        print("\n按退化强度汇总：")
        print(severity.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        print("\nM1-M6 三种退化强度横向对比：")
        print(
            level_comparison.to_string(
                index=False, float_format=lambda value: f"{value:.6f}"
            )
        )

    print_script_result(
        "13_evaluate_resource_allocation.py",
        {
            "device": light.device,
            "selected_images": args.sample_size,
            "scenario_count": int(
                detail[["degradation", "degradation_level"]].drop_duplicates().shape[0]
            ),
            "method_scenario_rows": len(scenarios),
            "detail_records": len(detail),
            "selection_csv": selection_path,
            "detail_csv": detail_path,
            "scenario_summary_csv": scenario_summary_path,
            "severity_summary_csv": severity_summary_path,
            "level_comparison_csv": level_comparison_path,
            "wall_clock_seconds": time.perf_counter() - started,
        },
    )


if __name__ == "__main__":
    main()
