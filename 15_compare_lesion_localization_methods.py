from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import project_config as config
from cxr_cascade.calibration import TemperatureCalibrator
from cxr_cascade.comparative_explainability import ComparativeExplanationEngine
from cxr_cascade.inference import ModelRunner
from cxr_cascade.method_comparison_evaluation import evaluate_localization_methods
from cxr_cascade.quality import QualityModel
from cxr_cascade.reporting import print_script_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=("在 RSNA 病灶框上统一比较 Grad-CAM、ShapleyCAM、LayerCAM 和小波引导 Grad-CAM")
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=config.METHOD_COMPARISON_MAX_SAMPLES,
        help="仅用于冒烟测试；正式实验不要设置",
    )
    parser.add_argument(
        "--visual-samples",
        type=int,
        default=config.METHOD_COMPARISON_VISUAL_MAX_SAMPLES,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.METHOD_COMPARISON_OUTPUT_ROOT,
    )
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    detail_path = output_root / "method_comparison_details.csv"
    summary_path = output_root / "tables" / "method_comparison_summary.csv"
    paired_path = output_root / "tables" / "method_comparison_pairwise.csv"
    case_root = output_root / "case_examples"

    app = config.active_config()
    light_runner = ModelRunner(config.LIGHT_CHECKPOINT)
    heavy_runner = ModelRunner(config.HEAVY_CHECKPOINT)
    quality_model = QualityModel.load(config.QUALITY_MODEL_PATH)
    print("\n开始四种方法的病灶覆盖对比。", flush=True)
    engine = ComparativeExplanationEngine(
        light_runner,
        heavy_runner,
        quality_model,
        TemperatureCalibrator.load(config.CALIBRATOR_PATH),
        app.wavelet,
        app.router,
        config.EXPLANATION_BETA,
    )
    detail, summary, paired = evaluate_localization_methods(
        config.RSNA_LOCALIZATION_MANIFEST_PATH,
        config.RSNA_DICOM_ROOT,
        engine,
        detail_path,
        summary_path,
        paired_path,
        case_root,
        top_fraction=config.LOCALIZATION_TOP_SALIENCY_FRACTION,
        max_predicted_boxes=config.LOCALIZATION_MAX_PREDICTED_BOXES,
        minimum_component_fraction=config.LOCALIZATION_MIN_COMPONENT_FRACTION,
        bootstrap_repeats=config.METHOD_COMPARISON_BOOTSTRAP_REPEATS,
        seed=config.SEED,
        max_samples=args.max_samples,
        visual_samples=args.visual_samples,
        resume=args.resume,
    )
    with pd.option_context("display.max_columns", None, "display.width", 260, "display.max_rows", None):
        print("\n四种解释方法的病灶定位汇总：")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
        print("\n四种方法两两之间的患者级成对比较：")
        print(paired.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print_script_result(
        "15_compare_lesion_localization_methods.py",
        {
            "annotated_images": int(detail["sop_instance_uid"].nunique()),
            "methods": sorted(detail["explanation_method"].unique().tolist()),
            "detail_csv": detail_path,
            "summary_csv": summary_path,
            "pairwise_comparison_csv": paired_path,
            "case_images": case_root,
            "max_samples": args.max_samples,
        },
    )


if __name__ == "__main__":
    main()
