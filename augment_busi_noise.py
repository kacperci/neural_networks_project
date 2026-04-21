#!/usr/bin/env python3
"""Create a noisy duplicated BUSI dataset for class folders.

The script reads images from class directories (benign/malignant/normal),
skips segmentation masks, and writes a new dataset with:
- original images
- noisy duplicates with randomized Gaussian noise parameters
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class AugmentConfig:
    input_root: Path
    output_root: Path
    classes: tuple[str, ...]
    copies_per_image: int
    sigma_min: float
    sigma_max: float
    seed: int
    include_original: bool


def parse_args() -> AugmentConfig:
    parser = argparse.ArgumentParser(
        description="Duplicate BUSI class images by adding random Gaussian noise."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("Dataset_BUSI_with_GT"),
        help="Path to dataset root containing benign/malignant/normal folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Dataset_BUSI_noisy"),
        help="Path where the augmented dataset will be saved.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["benign", "malignant", "normal"],
        help="Class subfolders to process.",
    )
    parser.add_argument(
        "--copies-per-image",
        type=int,
        default=1,
        help="How many noisy duplicates to create for each source image.",
    )
    parser.add_argument(
        "--sigma-min",
        type=float,
        default=8.0,
        help="Minimum stddev of Gaussian noise.",
    )
    parser.add_argument(
        "--sigma-max",
        type=float,
        default=30.0,
        help="Maximum stddev of Gaussian noise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-original",
        action="store_true",
        help="If set, do not copy original images to output.",
    )

    args = parser.parse_args()

    if args.copies_per_image < 1:
        raise ValueError("copies-per-image must be >= 1")
    if args.sigma_min < 0 or args.sigma_max < 0:
        raise ValueError("sigma values must be >= 0")
    if args.sigma_min > args.sigma_max:
        raise ValueError("sigma-min must be <= sigma-max")

    return AugmentConfig(
        input_root=args.input_root,
        output_root=args.output_root,
        classes=tuple(args.classes),
        copies_per_image=args.copies_per_image,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        seed=args.seed,
        include_original=not args.no_original,
    )


def is_source_image(path: Path) -> bool:
    suffix_ok = path.suffix.lower() in VALID_EXTENSIONS
    is_mask = "_mask" in path.stem.lower()
    return suffix_ok and not is_mask


def add_gaussian_noise(image_arr: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(loc=0.0, scale=sigma, size=image_arr.shape)
    noisy = image_arr.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def augment_class(config: AugmentConfig, class_name: str) -> tuple[int, int]:
    input_dir = config.input_root / class_name
    output_dir = config.output_root / class_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input class folder: {input_dir}")

    images = sorted(p for p in input_dir.iterdir() if p.is_file() and is_source_image(p))

    originals_saved = 0
    noisy_saved = 0

    for src_path in images:
        image = Image.open(src_path).convert("RGB")
        base_arr = np.array(image)

        if config.include_original:
            original_target = output_dir / src_path.name
            image.save(original_target)
            originals_saved += 1

        stem = src_path.stem
        suffix = src_path.suffix.lower()

        for idx in range(1, config.copies_per_image + 1):
            sigma = random.uniform(config.sigma_min, config.sigma_max)
            noisy_arr = add_gaussian_noise(base_arr, sigma=sigma)
            noisy_img = Image.fromarray(noisy_arr)

            noisy_name = f"{stem}__noise_{idx:02d}{suffix}"
            noisy_target = output_dir / noisy_name
            noisy_img.save(noisy_target)
            noisy_saved += 1

    return originals_saved, noisy_saved


def main() -> None:
    config = parse_args()

    random.seed(config.seed)
    np.random.seed(config.seed)

    config.output_root.mkdir(parents=True, exist_ok=True)

    total_originals = 0
    total_noisy = 0

    print("Starting BUSI noise augmentation...")
    print(f"Input root:  {config.input_root}")
    print(f"Output root: {config.output_root}")
    print(f"Classes:     {', '.join(config.classes)}")
    print(f"Copies/img:  {config.copies_per_image}")
    print(f"Sigma range: [{config.sigma_min}, {config.sigma_max}]")

    for class_name in config.classes:
        originals_saved, noisy_saved = augment_class(config, class_name)
        total_originals += originals_saved
        total_noisy += noisy_saved
        print(
            f"[{class_name}] originals: {originals_saved}, noisy: {noisy_saved}, total: {originals_saved + noisy_saved}"
        )

    print("Done.")
    print(
        f"Saved files -> originals: {total_originals}, noisy: {total_noisy}, all: {total_originals + total_noisy}"
    )


if __name__ == "__main__":
    main()
