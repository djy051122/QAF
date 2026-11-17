from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import project_config as config
from cxr_cascade.calibration import TemperatureCalibrator
from cxr_cascade.explanation_evaluation import CascadeExplanationEngine
from cxr_cascade.inference import ModelRunner
from cxr_cascade.localization_evaluation import (
    evaluate_lesion_localization,
    reanalyze_existing_localization,
)
from cxr_cascade.quality import QualityModel
from cxr_cascade.reporting import print_script_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="在 RSNA 最终仲裁病灶框上比较原始 Grad-CAM 与小波引导 Grad-CAM 的定位能力"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=config.LOCALIZATION_MAX_SAMPLES,
        help="仅用于冒烟测试；正式实验不要设置",
    )
    parser.add_argument(
        "--visual-samples",
        type=int,
        default=config.LOCALIZATION_VISUAL_MAX_SAMPLES,
        help="按 SOP UID 排序后固定保存前 N 例，避免按结果挑选病例",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有逐例结果继续；只在阈值和模型权重未改变时使用",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.LOCALIZATION_OUTPUT_ROOT,
        help="定位实验输出目录；冒烟测试建议指定临时目录",
    )
    parser.add_argument(
        "--reanalyze-existing",
        action="store_true",
        help="不重复模型推理，直接为已有逐例结果补算病灶覆盖率和框外响应统计",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    table_root = output_root / "tables"
    detail_path = output_root / "localization_details.csv"
    summary_path = table_root / "localization_summary.csv"
    comparison_path = table_root / "localization_paired_comparison.csv"
    conclusion_path = output_root / "localization_conclusion.json"
    figure_path = output_root / "localization_metrics_comparison.png"
    case_root = output_root / "case_examples"

    if args.reanalyze_existing:
        detail, summary, comparison, conclusion = reanalyze_existing_localization(
            detail_path,
            config.RSNA_LOCALIZATION_MANIFEST_PATH,
            summary_path,
            comparison_path,
            conclusion_path,
            figure_path,
            bootstrap_repeats=config.LOCALIZATION_BOOTSTRAP_REPEATS,
            seed=config.SEED,
        )
        with pd.option_context(
            "display.max_columns",
            None,
            "display.max_rows",
            None,
            "display.width",
            260,
            "display.expand_frame_repr",
            False,
        ):
            print("\n病灶覆盖与定位汇总（↑ 越高越好，质心距离/框外能量 ↓ 越低越好）：")
            print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
            print("\n小波引导相对原始 Grad-CAM 的成对改进（统一为正值有利于小波引导）：")
            print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        print(f"\n覆盖结论：{conclusion['coverage_statement']}")
        print(f"整体定位结论：{conclusion['statement']}")
        print_script_result(
            "12_evaluate_lesion_localization.py --reanalyze-existing",
            {
                "annotated_images": int(detail["sop_instance_uid"].nunique()),
                "detail_csv": detail_path,
                "summary_csv": summary_path,
                "paired_comparison_csv": comparison_path,
                "conclusion_json": conclusion_path,
                "coverage_verdict": conclusion["coverage_verdict"],
                "localization_verdict": conclusion["verdict"],
            },
        )
        return

    app = config.active_config()
    engine = CascadeExplanationEngine(
        ModelRunner(config.LIGHT_CHECKPOINT),
        ModelRunner(config.HEAVY_CHECKPOINT),
        QualityModel.load(config.QUALITY_MODEL_PATH),
        TemperatureCalibrator.load(config.CALIBRATOR_PATH),
        app.wavelet,
        app.router,
        config.EXPLANATION_BETA,
    )
    detail, summary, comparison, conclusion = evaluate_lesion_localization(
        config.RSNA_LOCALIZATION_MANIFEST_PATH,
        config.RSNA_DICOM_ROOT,
        engine,
        detail_path,
        summary_path,
        comparison_path,
        conclusion_path,
        figure_path,
        case_root,
        top_fraction=config.LOCALIZATION_TOP_SALIENCY_FRACTION,
        max_predicted_boxes=config.LOCALIZATION_MAX_PREDICTED_BOXES,
        minimum_component_fraction=config.LOCALIZATION_MIN_COMPONENT_FRACTION,
        bootstrap_repeats=config.LOCALIZATION_BOOTSTRAP_REPEATS,
        seed=config.SEED,
        max_samples=args.max_samples,
        visual_samples=args.visual_samples,
        resume=args.resume,
    )

    with pd.option_context(
        "display.max_columns",
        None,
        "display.max_rows",
        None,
        "display.width",
        260,
        "display.expand_frame_repr",
        False,
    ):
        print("\n病灶定位汇总（↑ 越高越好，质心距离 ↓ 越低越好）：")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        print("\n小波引导相对原始 Grad-CAM 的成对改进（统一为正值有利于小波引导）：")
        print(comparison.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\n覆盖结论：{conclusion['coverage_statement']}")
    print(f"整体定位结论：{conclusion['statement']}")
    print(f"判定规则：{conclusion['decision_rule']}")
    print_script_result(
        "12_evaluate_lesion_localization.py",
        {
            "annotated_images": int(detail["sop_instance_uid"].nunique()),
            "detail_csv": detail_path,
            "summary_csv": summary_path,
            "paired_comparison_csv": comparison_path,
            "conclusion_json": conclusion_path,
            "comparison_figure": figure_path,
            "case_images": case_root,
            "verdict": conclusion["verdict"],
            "coverage_verdict": conclusion["coverage_verdict"],
            "max_samples": args.max_samples,
        },
    )


if __name__ == "__main__":
    main()
