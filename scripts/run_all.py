"""Build everything in one go: fetch data -> processed tables -> Question A -> Question B.

    python scripts/run_all.py                      # full pipeline (~2 min, plus the ~150 MB download on first run)
    python scripts/run_all.py --from "C:\\path"     # copy raw CSVs from a folder instead of downloading
    python scripts/run_all.py --quick              # main 57/60 rule only, no diagnostic (about half the time)
    python scripts/run_all.py --no-weather         # offline (Question B needs weather, so it is skipped)
    python scripts/run_all.py --serve              # build, then open the dashboard (add --skip-build to just open it)

Each step runs the existing script, so the steps can still be re-run on their own. Stops at the first failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def step(name: str, script: str, *args: str) -> None:
    print(f"\n=== {name} ===", flush=True)
    t = time.perf_counter()
    r = subprocess.run([sys.executable, str(SCRIPTS / script), *args])
    if r.returncode != 0:
        print(f"\nFAILED at '{name}' (exit {r.returncode}). Fix the error above, then re-run.", file=sys.stderr)
        raise SystemExit(r.returncode)
    print(f"--- {name} done in {time.perf_counter() - t:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", help="copy raw CSVs from this folder instead of downloading")
    ap.add_argument("--no-weather", action="store_true", help="skip the Open-Meteo call (also skips Question B)")
    ap.add_argument("--quick", action="store_true", help="main 57/60 rule only; skip the 60/60 run and diagnostic")
    ap.add_argument("--serve", action="store_true", help="launch the Streamlit dashboard when the build finishes")
    ap.add_argument("--skip-build", action="store_true", help="with --serve: open the dashboard without rebuilding")
    args = ap.parse_args()

    if not args.skip_build:
        build(args)
    if args.serve or args.skip_build:
        print("\n=== Launching dashboard on http://localhost:8501 (Ctrl+C to stop) ===", flush=True)
        app = SCRIPTS.parent / "app" / "Home.py"
        return subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], cwd=SCRIPTS.parent).returncode
    return 0


def build(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    step("1/4 Fetch and verify raw data", "get_data.py", *(["--from", args.src] if args.src else []))
    step("2/4 Build processed tables", "build_dataset.py", *(["--no-weather"] if args.no_weather else []))
    step("3/4 Question A: k-means patterns", "run_question_a.py", *(["--main-only"] if args.quick else []))
    if args.no_weather:
        print("\n=== 4/4 Question B skipped: it needs weather features (run without --no-weather) ===")
    else:
        step("4/4 Question B: day-ahead forecast + exceptions", "run_question_b.py",
             *(["--main-only", "--skip-diagnostic"] if args.quick else []))
    tail = "" if args.serve else " Next: streamlit run app/Home.py (or add --serve next time)"
    print(f"\nAll done in {time.perf_counter() - start:.0f}s.{tail}")


if __name__ == "__main__":
    raise SystemExit(main())
