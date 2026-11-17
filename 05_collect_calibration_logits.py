from __future__ import annotations

import project_config as config
from cxr_cascade.inference import ModelRunner
from cxr_cascade.prediction import collect_logits
from cxr_cascade.quality import QualityModel
from cxr_cascade.reporting import csv_row_count, print_script_result


def main() -> None:
    collect_logits(
        config.MANIFEST_PATH,
        config.DATASET_ROOT,
        "calibration",
        ModelRunner(config.LIGHT_CHECKPOINT),
        ModelRunner(config.HEAVY_CHECKPOINT),
        QualityModel.load(config.QUALITY_MODEL_PATH),
        config.APP_CONFIG.wavelet,
        config.CALIBRATION_LOGITS_PATH,
    )
    print(f"Calibration logits: {config.CALIBRATION_LOGITS_PATH}")
    print_script_result(
        "05_collect_calibration_logits.py",
        {
            "calibration_logits": config.CALIBRATION_LOGITS_PATH,
            "records": csv_row_count(config.CALIBRATION_LOGITS_PATH),
        },
    )


if __name__ == "__main__":
    main()
