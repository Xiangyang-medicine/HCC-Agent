#!/usr/bin/env python3
"""
Download TCGA-LIHC expression files from GDC API.
Uses direct API download since gdc-client is not available.
"""
import os
import sys
import time
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Constants
RAW_DIR = Path("data/raw/gdc/20260713")
OUTPUT_DIR = RAW_DIR / "raw_expression"
MANIFEST_PATH = RAW_DIR / "gdc_manifest_primary_tumor.tsv"
BATCH_SIZE = 20  # Download 20 files concurrently
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# GDC API endpoint
GDC_DATA_URL = "https://api.gdc.cancer.gov/data"

def get_downloaded_files():
    """Get list of already downloaded files to resume."""
    if not OUTPUT_DIR.exists():
        return set()
    # Match by full filename (from manifest) or UUID prefix
    downloaded = set()
    for f in OUTPUT_DIR.glob("*.tsv"):
        # The manifest filename is like: UUID.rna_seq.augmented_star_gene_counts.tsv
        # Match by the full manifest filename
        downloaded.add(f.name)
    return downloaded

def load_manifest():
    """Load manifest file."""
    downloads = []
    with open(MANIFEST_PATH, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                downloads.append({
                    'file_id': parts[0],
                    'filename': parts[1],
                    'md5': parts[2],
                    'size': parts[3]
                })
    return downloads

def download_file(file_info, session=None):
    """Download a single file from GDC API."""
    file_id = file_info['file_id']
    filename = file_info['filename']
    expected_md5 = file_info['md5']

    # Check if already downloaded
    output_path = OUTPUT_DIR / filename
    if output_path.exists():
        # Verify MD5
        import hashlib
        with open(output_path, 'rb') as f:
            actual_md5 = hashlib.md5(f.read()).hexdigest()
        if actual_md5 == expected_md5:
            return {'status': 'already_downloaded', 'file_id': file_id, 'filename': filename}
        else:
            # Corrupted, delete and re-download
            output_path.unlink()

    url = f"{GDC_DATA_URL}/{file_id}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for attempt in range(MAX_RETRIES):
        try:
            if session:
                response = session.get(url, headers=headers, timeout=300)
            else:
                response = requests.get(url, headers=headers, timeout=300)

            if response.status_code == 200:
                # Verify it's actual TSV content (not error JSON)
                if response.content.startswith(b'#') or response.content.startswith(b'gene_id'):
                    with open(output_path, 'wb') as f:
                        f.write(response.content)

                    # Verify MD5
                    import hashlib
                    actual_md5 = hashlib.md5(response.content).hexdigest()
                    if actual_md5 == expected_md5:
                        return {'status': 'success', 'file_id': file_id, 'filename': filename, 'size': len(response.content)}
                    else:
                        output_path.unlink()
                        return {'status': 'md5_mismatch', 'file_id': file_id, 'expected': expected_md5, 'actual': actual_md5}
                else:
                    return {'status': 'invalid_content', 'file_id': file_id, 'content_type': response.headers.get('Content-Type', 'unknown')}
            else:
                return {'status': 'http_error', 'file_id': file_id, 'code': response.status_code}

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return {'status': 'error', 'file_id': file_id, 'error': str(e)}

    return {'status': 'failed', 'file_id': file_id}

def download_batch(files_batch, max_workers=BATCH_SIZE):
    """Download a batch of files concurrently."""
    results = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(download_file, f, session): f for f in files_batch}
            for future in as_completed(futures):
                results.append(future.result())
    return results

def main():
    print("=" * 60)
    print("TCGA-LIHC Expression File Downloader")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load manifest
    print(f"\nLoading manifest: {MANIFEST_PATH}")
    downloads = load_manifest()
    print(f"Total files to download: {len(downloads)}")

    # Check already downloaded
    downloaded = get_downloaded_files()
    print(f"Already downloaded: {len(downloaded)}")

    # Filter pending downloads
    pending = [d for d in downloads if d['file_id'] not in downloaded]
    print(f"Pending downloads: {len(pending)}")

    if not pending:
        print("\nAll files already downloaded!")
        return

    # Download in batches
    total = len(pending)
    success = 0
    failed = 0
    skipped = 0

    start_time = time.time()

    for i in range(0, total, BATCH_SIZE):
        batch = pending[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\nBatch {batch_num}/{total_batches} ({len(batch)} files)...")

        results = download_batch(batch)

        for r in results:
            if r['status'] == 'success':
                success += 1
                print(f"  [OK] {r['filename'][:50]}...")
            elif r['status'] == 'already_downloaded':
                skipped += 1
                print(f"  [SKIP] {r['filename'][:50]} (already exists)")
            elif r['status'] == 'http_error':
                failed += 1
                print(f"  [FAIL] {r['file_id']}: HTTP {r['code']}")
            elif r['status'] == 'error':
                failed += 1
                print(f"  [FAIL] {r['file_id']}: {r['error'][:50]}")
            else:
                failed += 1
                print(f"  [FAIL] {r['file_id']}: {r['status']}")

        # Rate limiting
        time.sleep(0.5)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total files: {total}")
    print(f"Successful: {success}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()
