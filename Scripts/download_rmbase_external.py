#!/usr/bin/env python3
"""Download RMBase v3.0 source tables needed for external validation.

RMBase currently exposes files under:
  https://bioinformaticsscience.cn/rmbase/download/modSingleSiteFiles/hg38/

The site certificate may be expired on some systems, so this script supports
--insecure. Downloaded files are raw provenance inputs and should be kept
separate from processed H5 files.
"""

import argparse
import shutil
import ssl
import subprocess
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_BASE_URL = "https://bioinformaticsscience.cn/rmbase/download/modSingleSiteFiles"
DEFAULT_ASSEMBLY = "hg38"
DEFAULT_MOD_TYPES = ["m6A", "m1A", "m5C", "m7G", "Pseudo", "Nm", "RNA-editing", "otherMod"]


def download_file(url, output_path, insecure=False, timeout=120):
    curl = shutil.which("curl")
    if curl is not None:
        cmd = [
            curl,
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "--speed-time",
            "120",
            "--speed-limit",
            "1024",
            "-C",
            "-",
            "-o",
            str(output_path),
            url,
        ]
        if insecure:
            cmd.insert(1, "-k")
        subprocess.run(cmd, check=True)
        return

    context = ssl._create_unverified_context() if insecure else None
    request = urllib.request.Request(url, headers={"User-Agent": "MultiRM-external-validation/1.0"})
    with urllib.request.urlopen(request, context=context, timeout=timeout) as response:
        total = response.headers.get("Content-Length")
        total = int(total) if total is not None else None
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        received = 0
        with tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                received += len(chunk)
                if total:
                    pct = 100.0 * received / total
                    print(f"\r{output_path.name}: {received / 1048576:.1f}/{total / 1048576:.1f} MB ({pct:.1f}%)", end="")
                else:
                    print(f"\r{output_path.name}: {received / 1048576:.1f} MB", end="")
        print()
        tmp_path.replace(output_path)


def is_valid_tar_gz(path):
    try:
        with tarfile.open(path, "r:gz") as tar:
            tar.getmembers()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly", default=DEFAULT_ASSEMBLY)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--out_dir", default="Data/external_rmbase/raw/rmbase_v3")
    parser.add_argument("--mod_types", nargs="+", default=DEFAULT_MOD_TYPES)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for mod_type in args.mod_types:
        filename = f"{args.assembly}.{mod_type}.tar.gz"
        url = f"{args.base_url.rstrip('/')}/{args.assembly}/{filename}"
        output_path = out_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite and is_valid_tar_gz(output_path):
            print(f"exists: {output_path}", flush=True)
            continue
        print(f"download: {url}", flush=True)
        try:
            download_file(url, output_path, insecure=args.insecure, timeout=args.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"failed: {url}: {exc}", file=sys.stderr)
            failures.append((mod_type, str(exc)))

    if failures:
        print("download failures:", file=sys.stderr)
        for mod_type, error in failures:
            print(f"  {mod_type}: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
