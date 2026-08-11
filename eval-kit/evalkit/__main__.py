"""Allow `python -m evalkit ...` to invoke the CLI."""
import sys

from .evaluator import main

if __name__ == "__main__":
    sys.exit(main())
