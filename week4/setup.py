#!/usr/bin/env python

import argparse
from glob import glob
from pathlib import Path
import shutil


def main(debug=False):
    this_file = Path(__file__).name
    nawi_dir = Path.home() / ".nawi"
    custom_fs = nawi_dir / "custom_flowsheets"
    for pyfile in glob("*.py"):
        if not pyfile.startswith("custom_flowsheet") and pyfile != this_file:
            if debug:
                print(f"copy '{pyfile}' -> {custom_fs / pyfile}'")
            shutil.copyfile(pyfile, custom_fs / pyfile)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--debug", action="store_true")
    args = p.parse_args()
    main(debug=args.debug)
