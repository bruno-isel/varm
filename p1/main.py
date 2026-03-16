#!/usr/bin/env python3
"""
Main script to normalize face dataset and run evaluation.
"""

import os
import cv2
import argparse
from normalizer import FaceNormalizer


def normalize_dataset(input_dir, output_dir, verbose=True):
    """
    Normalizes all face images in the input directory and saves them to output_dir.
    Supports both flat directories and nested structures (train/test/Person/).

    Returns:
        Tuple of (success_count, fail_count)
    """
    normalizer = FaceNormalizer()

    success_count = 0
    fail_count = 0

    # Walk through all subdirectories to find images
    for root, dirs, files in os.walk(input_dir):
        for filename in sorted(files):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                continue

            img_path = os.path.join(root, filename)

            # Preserve directory structure relative to input_dir
            rel_path = os.path.relpath(root, input_dir)
            out_dir = os.path.join(output_dir, rel_path)
            os.makedirs(out_dir, exist_ok=True)

            # Read image
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                if verbose:
                    print(f"  ✗ Failed to read: {os.path.join(rel_path, filename)}")
                fail_count += 1
                continue

            # Normalize
            normalized = normalizer.normalize(image)

            if normalized is None:
                if verbose:
                    print(f"  ✗ No eyes detected: {os.path.join(rel_path, filename)}")
                fail_count += 1
                continue

            # Save normalized image
            output_path = os.path.join(out_dir, filename)
            cv2.imwrite(output_path, normalized)
            success_count += 1
            if verbose:
                print(f"  ✓ {os.path.join(rel_path, filename)}")

    return success_count, fail_count


def main():
    parser = argparse.ArgumentParser(
        description="Normalize face images according to MPEG-7 specification."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="celebrityDatasetNorm",
        help="Input directory containing face images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="celebrityMPEG7",
        help="Output directory for normalized images"
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input directory does not exist: {args.input}")
        return

    print(f"--- MPEG-7 Face Normalization ---")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Target: 56x46, eyes at (16,24) and (31,24)")
    print()

    success_count, fail_count = normalize_dataset(args.input, args.output, verbose=True)

    print()
    print(f"--- Done ---")
    print(f"✓ Normalized: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print()
    print(f"Evaluate with:")
    print(f"  python test_accuracy.py --dataset {args.output}")


if __name__ == "__main__":
    main()
