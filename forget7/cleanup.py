"""Delete the model packages a run registered, except the ones you still want.

  python -m forget7.cleanup --keep 6 30          # dry run: lists what would be deleted
  python -m forget7.cleanup --keep 6 30 --yes    # deletes the packages and their S3 artifacts
  python -m forget7.cleanup --run run-20261023-full --keep 6 30 --yes

Keeps the weights and state packages of the listed steps plus the final merged package.
A 30-step run registers about 40 versions, each with weights in S3; the package holds the
only reference to its S3 location, so this is the way to free the storage.
"""

import argparse

from . import config as C
from .util import dump_json, load_json, run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run folder under out/; default: the latest")
    ap.add_argument("--keep", type=int, nargs="*", default=[], metavar="STEP")
    ap.add_argument("--yes", action="store_true", help="actually delete; without it, only list")
    a = ap.parse_args()
    run = run_dir(a.run)
    ck = load_json(run / "ckpts.json")
    keep = {str(s) for s in a.keep}
    doomed = [(kind, step, arn) for kind in ("weights", "state") for step, arn in ck[kind].items() if step not in keep]
    kept = [(kind, step) for kind in ("weights", "state") for step in ck[kind] if step in keep]
    keeping = [*kept, "final_merged"]
    print(f"{run}: keeping {keeping}, deleting {len(doomed)} packages")
    if not a.yes:
        for kind, step, arn in doomed:
            print(f"  would delete {kind:7} step {step:>3}  {arn}")
        print("re-run with --yes to delete")
        return
    svc = C.service_client()
    for kind, step, arn in doomed:
        svc.delete_model_package(arn, delete_artifacts=True)
        del ck[kind][step]
        dump_json(run / "ckpts.json", ck)
        print(f"  deleted {kind:7} step {step:>3}")


if __name__ == "__main__":
    main()
