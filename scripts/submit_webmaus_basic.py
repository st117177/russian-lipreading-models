#!/usr/bin/env python3
"""Submit MAUS-ready wav/txt pairs to BAS WebMAUS Basic and download TextGrid."""

from __future__ import annotations

import argparse
import re
import shutil
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
import urllib3


SERVICE_URL = "https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSBasic"


def safe_upload_name(path: Path) -> str:
    """WebMAUS rejects whitespace, non-ASCII and regex metacharacters in upload names."""
    suffix = path.suffix or ""
    stem = path.stem or "upload"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    safe_stem = safe_stem.strip("._-") or "upload"
    return f"{safe_stem}{suffix}"


def submit_job(
    signal_path: Path,
    text_path: Path,
    language: str,
    out_format: str,
    verify_ssl: bool,
) -> tuple[str, str]:
    with signal_path.open("rb") as signal_file, text_path.open("rb") as text_file:
        files = {
            "SIGNAL": (safe_upload_name(signal_path), signal_file, "audio/wav"),
            "TEXT": (safe_upload_name(text_path), text_file, "text/plain; charset=utf-8"),
        }
        data = {
            "LANGUAGE": language,
            "OUTFORMAT": out_format,
        }
        response = requests.post(
            SERVICE_URL,
            data=data,
            files=files,
            timeout=3600,
            verify=verify_ssl,
        )

    response.raise_for_status()

    xml_root = ET.fromstring(response.text)
    success = (xml_root.findtext("success") or "").strip().lower() == "true"
    if not success:
        message = xml_root.findtext("output") or xml_root.findtext("warnings") or response.text
        raise RuntimeError(f"WebMAUS request failed: {message}")

    download_link = xml_root.findtext("downloadLink")
    warnings_text = (xml_root.findtext("warnings") or "").strip()
    if not download_link:
        raise RuntimeError("WebMAUS returned success=true but no downloadLink")
    return download_link, warnings_text


def download_file(url: str, out_path: Path, verify_ssl: bool) -> None:
    with requests.get(url, stream=True, timeout=3600, verify=verify_ssl) as response:
        response.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            shutil.copyfileobj(response.raw, f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit wav/txt to BAS WebMAUS and download TextGrid")
    parser.add_argument("--signal", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--language", default="rus-RU")
    parser.add_argument("--out-format", default="TextGrid")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--backup-existing", action="store_true")
    args = parser.parse_args()

    verify_ssl = not args.insecure
    if not verify_ssl:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)

    if args.backup_existing and args.out.exists():
        backup_path = args.out.with_suffix(args.out.suffix + ".bak")
        shutil.copy2(args.out, backup_path)
        print(f"Backed up existing file: {backup_path}")

    download_link, warnings_text = submit_job(
        signal_path=args.signal,
        text_path=args.text,
        language=args.language,
        out_format=args.out_format,
        verify_ssl=verify_ssl,
    )
    download_file(download_link, args.out, verify_ssl=verify_ssl)

    print(f"Downloaded: {args.out}")
    print(f"Download link: {download_link}")
    if warnings_text:
        print(f"Warnings: {warnings_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
