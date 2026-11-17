from __future__ import annotations

import project_config as config
from cxr_cascade.reporting import print_script_result
from cxr_cascade.rsna import build_rsna_localization_manifest


def main() -> None:
    images, boxes = build_rsna_localization_manifest(
        config.RSNA_ANNOTATIONS_PATH,
        config.RSNA_MAPPINGS_PATH,
        config.RSNA_DICOM_ROOT,
        config.RSNA_LOCALIZATION_MANIFEST_PATH,
        config.RSNA_LOCALIZATION_MANIFEST_SUMMARY_PATH,
        sample_size=config.RSNA_LOCALIZATION_SAMPLE_SIZE,
        seed=config.SEED,
    )
    print(f"RSNA 最终仲裁阳性影像: {images}")
    print(f"RSNA 最终仲裁病灶框: {boxes}")
    print_script_result(
        "11_prepare_rsna_localization.py",
        {
            "selected_annotation": "Calculated / Lung Opacity",
            "sampling": (
                "使用全部最终仲裁阳性病例"
                if config.RSNA_LOCALIZATION_SAMPLE_SIZE is None
                else "按病灶数量分层的患者级固定随机抽样"
            ),
            "sampling_seed": (
                None if config.RSNA_LOCALIZATION_SAMPLE_SIZE is None else config.SEED
            ),
            "annotated_images": images,
            "lesion_boxes": boxes,
            "manifest": config.RSNA_LOCALIZATION_MANIFEST_PATH,
            "summary": config.RSNA_LOCALIZATION_MANIFEST_SUMMARY_PATH,
        },
    )


if __name__ == "__main__":
    main()
