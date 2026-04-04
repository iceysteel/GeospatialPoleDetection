#!/usr/bin/env python3
"""
EagleView Imagery Data Acquisition Pipeline
Discovers and downloads ortho + oblique imagery for the Omaha sandbox area.
"""
import sys
import os
import time

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from discovery import discover_imagery
from download import download_all


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def make_progress(label, start_time):
    """Return a progress callback that shows ETA."""
    def on_progress(i, total):
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        remaining = (total - i) / rate if rate > 0 else 0
        pct = i / total * 100
        bar_len = 30
        filled = int(bar_len * i / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r  {label} {bar} {pct:5.1f}% [{i}/{total}] ETA: {fmt_time(remaining)}  ", end="", flush=True)
        if i == total:
            print(f"\r  {label} {bar} 100.0% [{total}/{total}] Done in {fmt_time(elapsed)}     ")
    return on_progress


def main():
    print("=" * 60)
    print("EagleView Data Acquisition Pipeline")
    print("=" * 60)

    # Step 1: Discover imagery
    print("\n--- Step 1: Discovery ---")
    t0 = time.time()
    images = discover_imagery(on_progress=make_progress("Discovery", t0))

    if not images:
        print("No images found. Check credentials and bounding box.")
        sys.exit(1)

    # Step 2: Download imagery
    print("\n--- Step 2: Download ---")
    t1 = time.time()
    results = download_all(images, on_progress=make_progress("Download ", t1))

    # Summary
    elapsed_total = time.time() - t0
    print(f"\n--- Summary (total: {fmt_time(elapsed_total)}) ---")
    by_type = {}
    for meta in results.values():
        key = f"{meta['type']}/{meta['direction']}"
        by_type.setdefault(key, []).append(meta)

    for key, items in sorted(by_type.items()):
        ok = sum(1 for m in items if m.get("local_path"))
        print(f"  {key}: {ok}/{len(items)} downloaded")

    total_size = 0
    for meta in results.values():
        p = meta.get("local_path")
        if p and os.path.exists(p):
            total_size += os.path.getsize(p)
    print(f"  Total data: {total_size / 1024 / 1024:.1f} MB")

    print("\nDone! Open viewer.html to explore the imagery.")


if __name__ == "__main__":
    main()
