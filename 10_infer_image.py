from __future__ import annotations

import argparse
import json

from PIL import Image

import project_config as config
from cxr_cascade.audit import JsonlAuditWriter
from cxr_cascade.calibration import TemperatureCalibrator
from cxr_cascade.inference import CascadeInference, ModelRunner
from cxr_cascade.quality import QualityModel
from cxr_cascade.reporting import print_script_result


def main() -> None:
    parser = argparse.ArgumentParser(description="对单张胸片运行完整质量感知级联")
    parser.add_argument("image", help="胸片文件路径")
    args = parser.parse_args()
    app = config.active_config()
    audit = JsonlAuditWriter(config.AUDIT_LOG_PATH)
    engine = CascadeInference(
        ModelRunner(config.LIGHT_CHECKPOINT),
        ModelRunner(config.HEAVY_CHECKPOINT),
        QualityModel.load(config.QUALITY_MODEL_PATH),
        TemperatureCalibrator.load(config.CALIBRATOR_PATH),
        app.wavelet,
        app.router,
        audit,
    )
    try:
        with Image.open(args.image) as image:
            result = engine.predict(image.convert("L"))
    finally:
        audit.close()
    result_dict = result.to_dict()
    print(json.dumps(result_dict, ensure_ascii=False, indent=2))
    print_script_result(
        "10_infer_image.py",
        {"image": args.image, "prediction": result_dict, "audit_log": config.AUDIT_LOG_PATH},
    )


if __name__ == "__main__":
    main()
