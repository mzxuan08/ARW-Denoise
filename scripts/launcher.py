import sys

from arw_denoise.cli import main

raise SystemExit(main(sys.argv[1:] or ["gui"]))
