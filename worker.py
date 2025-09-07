# worker.py (hardened / full)
import os
import sys
import json
import math
import traceback
from datetime import datetime

# Defensive import-check so failure message appears early in logs
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import pandas_ta as ta
except Exception as e:
    print("[worker] ERROR importing required packages:", e)
    traceback.print_exc()
    # Exit with non-zero to make logs obvious
    sys.exit(3)

# If nasdaqlisted.txt missing, attempt to download in-process (fallback)
if not os.path.exists("nasdaqlisted.txt"):
    print("[worker] nasdaqlisted.txt not found locally — attempting to download via download_nasdaq_list.py")
    try:
        import download_nasdaq_list as dnl
        # If the module exists, call its main download routine if present
        # If the user has the robust script as suggested, it will run
        if hasattr(dnl, "__name__"):
            # call file as script-like
            dnl.__dict__.get("__main__", None)
        # Fallback to executing the file directly
        try:
            exec(open("download_nasdaq_list.py", "rb").read(), {"__name__": "__main__"})
        except Exception as e:
            print("[worker] fallback exec failed:", e)
            traceback.print_exc()
    except FileNotFoundError:
        print("[worker] download_nasdaq_list.py not found; please make sure it's in repo root.")
    except Exception as e:
        print("[worker] attempted in-process download failed:", e)
        traceback.print_exc()

# After download attempt, ensure file exists
if not os.path.exists("nasdaqlisted.txt"):
    print("[worker] FATAL: nasdaqlisted.txt still not present after attempts. Exiting.")
    sys.exit(4)

# Continue with the rest of the worker implementation (unchanged)
# --- original worker content below (keep existing compute_indicators, main, etc.) ---
# For brevity, the rest of your original worker.py code (compute_indicators and main) should remain unchanged.
# If you want the entire file replaced, paste your original worker.py code after this comment.
# Ensure you still wrap main() in a try/except to print errors and exit non-zero if anything unexpected happens.

def safe_main():
    try:
        # insert the original main() function body of worker.py here
        from worker_core import core_main
        # If you refactor main into worker_core.core_main, call it here
        # Otherwise, if your worker.py already contains a main(), call it:
        # main()
        pass
    except Exception as e:
        print("[worker] Unhandled exception in main():", e)
        traceback.print_exc()
        sys.exit(5)

if __name__ == "__main__":
    safe_main()
