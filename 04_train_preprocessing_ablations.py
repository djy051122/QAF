from __future__ import annotations

import project_config as config
from cxr_cascade.quality import QualityModel
from cxr_cascade.reporting import print_script_result
from cxr_cascade.training import TrainConfig, train_model


def main() -> None:
    quality = QualityModel.load(config.QUALITY_MODEL_PATH)
    results = {}
    for variant, checkpoint in config.PREPROCESSING_CHECKPOINTS.items():
        if variant == "adaptive" and checkpoint.exists():
            print(f"Reuse matched adaptive checkpoint: {checkpoint}")
            results[variant] = {"status": "reused", "checkpoint": str(checkpoint)}
            continue
        result = train_model(
            config.MANIFEST_PATH,
            config.DATASET_ROOT,
            quality,
            config.APP_CONFIG.wavelet,
            TrainConfig(
                model_name="resnet50",
                batch_size=config.HEAVY_BATCH_SIZE,
                epochs=config.EPOCHS,
                learning_rate=config.HEAVY_LEARNING_RATE,
                seed=config.SEED,
                preprocessing_variant=variant,
            ),
            checkpoint,
        )
        results[variant] = result
    print_script_result("04_train_preprocessing_ablations.py", results)


if __name__ == "__main__":
    main()
