from __future__ import annotations

import argparse

import pandas as pd

import project_config as config
from cxr_cascade.inference import ModelRunner
from cxr_cascade.preprocessing_evaluation import evaluate_preprocessing_ablation
from cxr_cascade.quality import QualityModel
from cxr_cascade.reporting import print_script_result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估自适应小波预处理对不同质量退化胸片的影响"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="仅用于冒烟测试；正式实验不设置该参数",
    )
    args = parser.parse_args()
    runners = {
        name: ModelRunner(path) for name, path in config.PREPROCESSING_CHECKPOINTS.items()
    }
    _, summary = evaluate_preprocessing_ablation(
        config.MANIFEST_PATH,
        config.DATASET_ROOT,
        "test",
        runners,
        QualityModel.load(config.QUALITY_MODEL_PATH),
        config.APP_CONFIG.wavelet,
        config.PREPROCESSING_DETAIL_PATH,
        config.PREPROCESSING_SUMMARY_PATH,
        seed=config.SEED,
        max_samples=args.max_samples,
    )
    with pd.option_context(
        "display.max_columns", None,
        "display.max_rows", None,
        "display.width", 240,
        "display.expand_frame_repr", False,
    ):
        print("\n自适应预处理消融实验汇总：")
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print_script_result(
        "08_evaluate_preprocessing.py",
        {
            "detail_csv": config.PREPROCESSING_DETAIL_PATH,
            "summary_csv": config.PREPROCESSING_SUMMARY_PATH,
            "detail_records": len(pd.read_csv(config.PREPROCESSING_DETAIL_PATH)),
            "summary_records": len(summary),
            "max_samples": args.max_samples,
        },
    )


if __name__ == "__main__":
    main()
