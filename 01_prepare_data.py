from __future__ import annotations

from collections import Counter

import project_config as config
from cxr_cascade.data import build_presplit_manifest, write_manifest
from cxr_cascade.reporting import print_script_result


def main() -> None:
    rows = build_presplit_manifest(
        config.DATASET_ROOT,
        None,
        repartition_patients=True,
        seed=config.SEED,
    )
    sample_counts = Counter(str(row["sample_id"]) for row in rows)
    duplicate_ids = [sample_id for sample_id, count in sample_counts.items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate sample identifiers detected: {duplicate_ids[:5]}")
    write_manifest(rows, config.MANIFEST_PATH)
    split_counts = Counter(str(row["split"]) for row in rows)
    label_counts = Counter(int(row["label"]) for row in rows)
    print(f"Manifest: {config.MANIFEST_PATH}")
    print(f"Images: {len(rows)}")
    print(f"Splits: {dict(split_counts)}")
    print(f"Labels: {dict(label_counts)}")
    print_script_result(
        "01_prepare_data.py",
        {
            "manifest": config.MANIFEST_PATH,
            "images": len(rows),
            "splits": dict(split_counts),
            "labels": dict(label_counts),
        },
    )


if __name__ == "__main__":
    main()
