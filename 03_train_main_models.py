from __future__ import annotations

import project_config as config
from cxr_cascade.quality import QualityModel
from cxr_cascade.reporting import print_script_result
from cxr_cascade.training import TrainConfig, train_model


def run(model_name: str, output, batch_size: int, learning_rate: float) -> dict:
    return train_model(
        config.MANIFEST_PATH,
        config.DATASET_ROOT,
        QualityModel.load(config.QUALITY_MODEL_PATH),
        config.APP_CONFIG.wavelet,
        TrainConfig(
            model_name=model_name,
            batch_size=batch_size,
            epochs=config.EPOCHS,
            learning_rate=learning_rate,
            seed=config.SEED,
            preprocessing_variant="adaptive",
        ),
        output,
    )
def main() -> None:
    results = {
        "mobilenet_v2": run(
            "mobilenet_v2",
            config.LIGHT_CHECKPOINT,
            config.LIGHT_BATCH_SIZE,
            config.LIGHT_LEARNING_RATE,
        ),
        "resnet50_adaptive": run(
            "resnet50",
            config.HEAVY_CHECKPOINT,
            config.HEAVY_BATCH_SIZE,
            config.HEAVY_LEARNING_RATE,
        ),
    }
    print_script_result("03_train_main_models.py", results)


if __name__ == "__main__":
    main()
