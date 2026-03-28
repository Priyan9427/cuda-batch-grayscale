#!/usr/bin/env python3
"""
download_images.py
Downloads a batch of free sample images from Unsplash Source
for use as test data with the CUDA grayscale converter.

Usage:
    python3 download_images.py [--count N] [--output DIR]
"""

import argparse
import os
import urllib.request
import sys

# Each URL fetches a different random image at 640×480
BASE_URL = "https://picsum.photos/640/480"

def download_images(count: int, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Downloading {count} images to '{output_dir}' ...")

    for i in range(1, count + 1):
        # ?random=N forces a unique image per request
        url = f"{BASE_URL}?random={i}"
        filename = os.path.join(output_dir, f"image_{i:04d}.jpg")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"  [{i:3d}/{count}] Saved: {filename}")
        except Exception as exc:
            print(f"  [{i:3d}/{count}] FAILED: {exc}", file=sys.stderr)

    print("\nDownload complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download sample images for the CUDA grayscale project."
    )
    parser.add_argument(
        "--count", type=int, default=20,
        help="Number of images to download (default: 20)"
    )
    parser.add_argument(
        "--output", type=str, default="data/images",
        help="Output directory (default: data/images)"
    )
    args = parser.parse_args()
    download_images(args.count, args.output)


if __name__ == "__main__":
    main()
