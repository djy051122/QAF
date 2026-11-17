from __future__ import annotations

import pandas as pd

import project_config as config
from cxr_cascade.calibration import TemperatureCalibrator
from cxr_cascade.config import AppConfig, save_config_module
from cxr_cascade.reporting import print_script_result
from cxr_cascade.tuning import tune_router


def main() -> None:
    frame = pd.read_csv(config.CALIBRATION_LOGITS_PATH)
    calibrator = TemperatureCalibrator.load(config.CALIBRATOR_PATH)
    for model in ("light", "heavy"):
        frame[f"{model}_probability"] = [
            calibrator.transform(model, logit, grade)
            for logit, grade in zip(frame[f"{model}_logit"], frame["quality_grade"])
        ]
    router, metrics = tune_router(
        frame,
        base=config.APP_CONFIG.router,
        min_sensitivity=config.MINIMUM_ROUTER_SENSITIVITY,
    )
    frozen = AppConfig(
        quality=config.APP_CONFIG.quality,
        wavelet=config.APP_CONFIG.wavelet,
        router=router,
    )
    save_config_module(config.FROZEN_CONFIG_PATH, frozen)
    print(f"Frozen config: {config.FROZEN_CONFIG_PATH}")
    print(metrics)
    print_script_result(
        "07_tune_router.py",
        {
            "frozen_config": config.FROZEN_CONFIG_PATH,
            "records": len(frame),
            "router": router.__dict__,
            "metrics": metrics,
        },
    )


if __name__ == "__main__":
    main()
