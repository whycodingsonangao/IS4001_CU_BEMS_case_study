"""Fetch the 14 CU-BEMS raw CSVs into data/raw/ and verify them.

    python scripts/get_data.py                 # download from the authors' official site (~150 MB of zips)
    python scripts/get_data.py --from <folder> # copy from a folder you already have (e.g. a Kaggle download)

Files already present with the right checksum are skipped, so it is safe to re-run.
Source: https://sgrudata.github.io/ (Pipattanasomporn et al., 2020, Scientific Data 7:241, CC BY 4.0).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import shutil
import sys
import zipfile
from pathlib import Path

import requests

from cubems import config

BASE_URL = "https://github.com/SGRUdata/SGRUdata.github.io/raw/master/data/{name}.zip"

# SHA-256 of each CSV as published by the authors (verified Nov 2026).
EXPECTED = {
    "2018Floor1.csv": "875d0ef9b6d80569f38999e7decb4743718698563e23856755d87187dc9abe49",
    "2018Floor2.csv": "610125fee377cb06d2ce55b2866279f96fcc1396b2fd9052858e946c27ec85dd",
    "2018Floor3.csv": "43ba1ba6f18a5e7acd54f7f0761bd1c075e66ca4ccd74b310496bb67a8c74d72",
    "2018Floor4.csv": "f61689d370500f1f1073e16c1dda7b0e4d93fa7e0d7b380c565bea542f0c38d1",
    "2018Floor5.csv": "7f4fc1570bc52f5cc654f342f93a88bbd94ff5266948e8aa0bfeb03d09267122",
    "2018Floor6.csv": "6f702c25089669312058d8eee12a29779d5d4dd58bf0e80b4a7fb25aa3484462",
    "2018Floor7.csv": "988bc5803ef622c98dd48abd452dc71920de56ab119e98f8f3f88c18e102936d",
    "2019Floor1.csv": "1b3d1cb4b4f070abf59807462deb915acf08d8861528c1e9af54bca12996d173",
    "2019Floor2.csv": "ae19902ef0ff1058e4cc185cae9e9213b618d0d6ab11de3801d52bc81e9ddb77",
    "2019Floor3.csv": "1742aaf67232b38c29ed6fd2df44c12e08626b5827ff1c9121c10d89b15e19d2",
    "2019Floor4.csv": "285e176cc3cfda05a327078b78019058dd1a7bf63088e543d4830db197c6c6a3",
    "2019Floor5.csv": "218028fe19b327106272a6de908bdddc53da724793a20d9e3c63a341c96f460d",
    "2019Floor6.csv": "3c364fa9329c43bb3354623aa7a8c32d4fc143dac8e2a4bfb6e2f8c89762d6b9",
    "2019Floor7.csv": "253f7ccfcd42c519f64776ad53606941c6203d4452cd1dc08aab749bfa469185",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid(path: Path) -> bool:
    return path.exists() and sha256(path) == EXPECTED[path.name]


def download(name: str, dest: Path) -> None:
    r = requests.get(BASE_URL.format(name=name), timeout=300)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # the zips also contain macOS "__MACOSX/._*" junk; take only the real CSV
        member = next(m for m in zf.namelist() if Path(m).name == name and not m.startswith("__MACOSX"))
        with zf.open(member) as src, dest.open("wb") as out:
            shutil.copyfileobj(src, out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", type=Path, help="copy CSVs from this folder instead of downloading")
    ap.add_argument("--force", action="store_true", help="re-fetch even if a valid file is present")
    args = ap.parse_args()

    raw = config.RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    failed = []
    for name in EXPECTED:
        dest = raw / name
        if not args.force and is_valid(dest):
            print(f"  ok       {name}")
            continue
        try:
            if args.src:
                src = args.src / name
                if not src.exists():
                    raise FileNotFoundError(f"{src} not found")
                shutil.copyfile(src, dest)
                how = "copied"
            else:
                download(name, dest)
                how = "fetched"
        except Exception as e:  # keep going so one bad file doesn't block the rest
            print(f"  FAILED   {name}: {e}", file=sys.stderr)
            failed.append(name)
            continue
        if is_valid(dest):
            print(f"  {how:<8} {name}")
        else:
            print(f"  BAD HASH {name}: content differs from the published file", file=sys.stderr)
            dest.unlink(missing_ok=True)
            failed.append(name)

    if failed:
        print(f"\n{len(failed)} file(s) missing or invalid: {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nAll {len(EXPECTED)} CSVs verified in {raw}. Next: python scripts/build_dataset.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
