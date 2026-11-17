from __future__ import annotations

import argparse

import pandas as pd

import project_config as config
from cxr_cascade.calibration import TemperatureCalibrator
from cxr_cascade.explanation_evaluation import evaluate_explanation_stability
from cxr_cascade.inference import ModelRunner
from cxr_cascade.quality import QualityModel
from cxr_cascade.reporting import print_script_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="比较原始 Grad-CAM 与小波引导 Grad-CAM 在质量退化下的解释稳定性"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=config.EXPLANATION_MAX_SAMPLES,
        help="仅用于冒烟测试；正式实验保持为空以使用全部测试样本",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=("light", "medium", "heavy"),
        default=list(config.EXPLANATION_DEGRADATION_LEVELS),
    )
    parser.add_argument(
        "--include-misclassified",
        action="store_true",
        help="同时纳入原图分类错误的样本；论文正式统计默认不纳入",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        default=config.EXPLANATION_POSITIVE_ONLY,
        help="只统计肺炎阳性样本",
    )
    parser.add_argument(
        "--visual-samples",
        type=int,
        default=config.EXPLANATION_VISUAL_MAX_SAMPLES,
    )
    args = parser.parse_args()
    app = config.active_config()
    _, summary, comparison = evaluate_explanation_stability(
        config.MANIFEST_PATH,
        config.DATASET_ROOT,
        "test",
        ModelRunner(config.LIGHT_CHECKPOINT),
        ModelRunner(config.HEAVY_CHECKPOINT),
        QualityModel.load(config.QUALITY_MODEL_PATH),
        TemperatureCalibrator.load(config.CALIBRATOR_PATH),
        app.wavelet,
        app.router,
        config.EXPLANATION_STABILITY_DETAIL_PATH,
        config.EXPLANATION_STABILITY_SUMMARY_PATH,
        config.EXPLANATION_STABILITY_COMPARISON_PATH,
        config.EXPLANATION_HEATMAP_ROOT,
        beta=config.EXPLANATION_BETA,
        seed=config.SEED,
        levels=tuple(args.levels),
        max_samples=args.max_samples,
        only_correct_clean=(
            config.EXPLANATION_ONLY_CORRECT_CLEAN and not args.include_misclassified
        ),
        positive_only=args.positive_only,
        visual_max_samples=args.visual_samples,
        visual_levels=config.EXPLANATION_VISUAL_LEVELS,
        bootstrap_repeats=config.EXPLANATION_BOOTSTRAP_REPEATS,
    )
    with pd.option_context(
        "display.max_columns", None,
        "display.max_rows", None,
        "display.width", 260,
        "display.expand_frame_repr", False,
    ):
        print("\n解释稳定性汇总（SSIM、Pearson、Top10% IoU 越高越好；质心漂移越低越好）：")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        print("\n小波引导相对原始 Grad-CAM 的配对增益（正值表示小波引导更优）：")
        print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print_script_result(
        "09_evaluate_explanation_stability.py",
        {
            "detail_csv": config.EXPLANATION_STABILITY_DETAIL_PATH,
            "summary_csv": config.EXPLANATION_STABILITY_SUMMARY_PATH,
            "paired_gain_csv": config.EXPLANATION_STABILITY_COMPARISON_PATH,
            "heatmaps": config.EXPLANATION_HEATMAP_ROOT,
            "detail_records": len(pd.read_csv(config.EXPLANATION_STABILITY_DETAIL_PATH)),
            "summary_records": len(summary),
            "comparison_records": len(comparison),
            "max_samples": args.max_samples,
            "levels": args.levels,
        },
    )


if __name__ == "__main__":
    main()
