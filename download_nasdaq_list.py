# download_nasdaq_list.py  (robust version)
import requests
import ftplib
import time
import os
from pathlib import Path

NASDAQ_HTTP = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
NASDAQ_FTP_HOST = "ftp.nasdaqtrader.com"
NASDAQ_FTP_PATH = "SymbolDirectory/nasdaqlisted.txt"

OUT_FN = "nasdaqlisted.txt"
RETRIES = 3
SLEEP = 3

def save_bytes(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def try_http():
    for attempt in range(1, RETRIES + 1):
        try:
            print(f"[download] Trying HTTP ({attempt}/{RETRIES}) {NASDAQ_HTTP}")
            r = requests.get(NASDAQ_HTTP, timeout=20)
            r.raise_for_status()
            save_bytes(OUT_FN, r.content)
            print(f"[download] Saved via HTTP -> {OUT_FN}")
            return True
        except Exception as e:
            print(f"[download] HTTP attempt {attempt} failed: {e}")
            time.sleep(SLEEP)
    return False

def try_ftp():
    for attempt in range(1, RETRIES + 1):
        try:
            print(f"[download] Trying FTP ({attempt}/{RETRIES}) {NASDAQ_FTP_HOST}")
            ftp = ftplib.FTP(NASDAQ_FTP_HOST, timeout=30)
            ftp.login()  # anonymous
            ftp.cwd("SymbolDirectory")
            with open(OUT_FN, "wb") as fh:
                ftp.retrbinary("RETR nasdaqlisted.txt", fh.write)
            ftp.quit()
            print(f"[download] Saved via FTP -> {OUT_FN}")
            return True
        except Exception as e:
            print(f"[download] FTP attempt {attempt} failed: {e}")
            time.sleep(SLEEP)
    return False

if __name__ == "__main__":
    # Try HTTP first (more likely to succeed on hosted CI); fallback to FTP.
    ok = False
    try:
        import requests  # ensure requests is present
    except Exception:
        print("[download] requests package not found; will try FTP directly.")
        ok = try_ftp()
    else:
        ok = try_http() or try_ftp()
    if not ok:
        print("[download] ERROR: Could not download nasdaqlisted.txt via HTTP or FTP.")
        raise SystemExit(2)
