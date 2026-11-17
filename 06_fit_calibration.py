from __future__ import annotations

import pandas as pd

import project_config as config
from cxr_cascade.calibration import TemperatureCalibrator
from cxr_cascade.reporting import print_script_result


def main() -> None:
    frame = pd.read_csv(config.CALIBRATION_LOGITS_PATH)
    calibrator = TemperatureCalibrator(
        "quality_conditional",
        minimum_group_size=config.MINIMUM_CALIBRATION_GROUP_SIZE,
    )
    for model in ("light", "heavy"):
        calibrator.fit(
            model,
            frame[f"{model}_logit"],
            frame["label"],
            frame["quality_grade"],
        )
    calibrator.save(config.CALIBRATOR_PATH)
    print(f"Calibrator: {config.CALIBRATOR_PATH}")
    print(calibrator.temperatures)
    print_script_result(
        "06_fit_calibration.py",
        {
            "calibrator": config.CALIBRATOR_PATH,
            "records": len(frame),
            "temperatures": calibrator.temperatures,
        },
    )


if __name__ == "__main__":
    main()
