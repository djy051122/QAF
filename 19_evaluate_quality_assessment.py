from __future__ import annotations

import argparse

import project_config as config
from cxr_cascade.quality import QualityModel
from cxr_cascade.quality_assessment_validation import (
    collect_quality_assessments,
    save_quality_validation_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 1: validate the image-quality assessment mechanism."
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--output-root", default=str(config.QUALITY_ASSESSMENT_OUTPUT_ROOT))
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    quality_model = QualityModel.load(config.QUALITY_MODEL_PATH)
    detail = collect_quality_assessments(
        config.MANIFEST_PATH,
        config.DATASET_ROOT,
        args.split,
        quality_model,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    paths = save_quality_validation_outputs(
        detail,
        args.output_root,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=args.seed,
        make_figures=not args.no_figures,
    )
    print("\n=== EXPERIMENT 1: QUALITY-ASSESSMENT VALIDATION ===")
    print(f"Images: {detail['sample_id'].nunique()} | observations: {len(detail)}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
