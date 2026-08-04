"""
Download HCC datasets from GEO using Python requests.
"""

import requests
import pandas as pd
from pathlib import Path
import gzip
import time
import sys

DATA_DIR = Path("F:/ACM/data/external")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5


def download_with_retry(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """Download file with retry logic."""
    if dest_path.exists():
        print(f"  Already exists: {dest_path.name}")
        return True

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Downloading {dest_path.name} (attempt {attempt + 1}/{MAX_RETRIES})...")
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = 100 * downloaded / total_size
                            print(f"\r    Progress: {pct:.1f}% ({downloaded/1024/1024:.1f}MB)", end='', flush=True)

            print()  # New line after progress
            print(f"    Downloaded: {dest_path.name} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return True

        except requests.exceptions.RequestException as e:
            print(f"    Error: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"    Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    Failed after {MAX_RETRIES} attempts")
                return False

    return False


def main():
    print("=" * 60)
    print("Downloading GEO HCC Datasets")
    print("=" * 60)

    # GSE14520 - TCGA Liver Cancer (247 samples)
    # Using the correct matrix file path from GEO
    gse14520_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE14520&format=file"
    gse14520_path = DATA_DIR / "GSE14520_raw.txt.gz"

    if download_with_retry(gse14520_url, gse14520_path):
        print("  Processing GSE14520...")
        try:
            # Try to load the downloaded file
            df = pd.read_csv(gse14520_path, sep='\t', index_col=0)
            print(f"    Loaded: {df.shape}")
            df.to_parquet(DATA_DIR / "GSE14520_raw.parquet")
            print(f"    Saved: GSE14520_raw.parquet")
        except Exception as e:
            print(f"    Error processing: {e}")

    # GSE76427 - Another HCC dataset
    gse76427_url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE76427&format=file"
    gse76427_path = DATA_DIR / "GSE76427_raw.txt.gz"

    if download_with_retry(gse76427_url, gse76427_path):
        print("  Processing GSE76427...")
        try:
            df = pd.read_csv(gse76427_path, sep='\t', index_col=0)
            print(f"    Loaded: {df.shape}")
            df.to_parquet(DATA_DIR / "GSE76427_raw.parquet")
            print(f"    Saved: GSE76427_raw.parquet")
        except Exception as e:
            print(f"    Error processing: {e}")

    print("\nDownload complete!")
    print(f"Files saved to: {DATA_DIR}")


if __name__ == "__main__":
    main()
