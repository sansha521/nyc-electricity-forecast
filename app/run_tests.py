from pathlib import Path
import unittest
import sys

def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    suite = unittest.TestLoader().discover(
        start_dir=str(project_root / "tests"),
        top_level_dir=str(project_root),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    success = result.wasSuccessful()

    if success and result.testsRun and not (result.skipped or result.expectedFailures):
        print(f"All {result.testsRun} tests passed.", file=sys.stderr)

    return 0 if success else 1

if __name__ == "__main__":
    raise SystemExit(main())