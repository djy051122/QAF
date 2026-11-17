from __future__ import annotations

from PIL import Image

import project_config as config
from cxr_cascade.data import read_manifest
from cxr_cascade.quality import QualityModel, raw_quality_metrics
from cxr_cascade.reporting import print_script_result


def main() -> None:
    grouped = {"train": [], "calibration": []}
    for row in read_manifest(config.MANIFEST_PATH):
        if row["split"] not in grouped:
            continue
        with Image.open(config.DATASET_ROOT / row["image_path"]) as image:
            grouped[row["split"]].append(
                raw_quality_metrics(
                    image,
                    original_shape=(image.height, image.width),
                    analysis_max_side=config.APP_CONFIG.quality.analysis_max_side,
                )
            )
    model = QualityModel.fit(
        grouped["train"], grouped["calibration"], config.APP_CONFIG.quality
    )
    config.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    model.save(config.QUALITY_MODEL_PATH)
    print(f"Quality model: {config.QUALITY_MODEL_PATH}")
    print(f"theta_high={model.theta_high:.6f}, theta_low={model.theta_low:.6f}")
    print_script_result(
        "02_fit_quality.py",
        {
            "quality_model": config.QUALITY_MODEL_PATH,
            "train_images": len(grouped["train"]),
            "calibration_images": len(grouped["calibration"]),
            "theta_high": model.theta_high,
            "theta_low": model.theta_low,
        },
    )


if __name__ == "__main__":
    main()
