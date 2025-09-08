# download_nasdaq_list.py
"""
Robust downloader for nasdaqlisted.txt
Tries HTTP first, then FTP; has retries and clear logging.
"""
import time
import requests
import ftplib
from pathlib import Path

OUT_FN = "nasdaqlisted.txt"
HTTP_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
FTP_HOST = "ftp.nasdaqtrader.com"
FTP_PATH = "SymbolDirectory/nasdaqlisted.txt"

RETRIES = 3
SLEEP = 3

def save_bytes(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)

def try_http():
    for attempt in range(1, RETRIES+1):
        try:
            print(f"[download] HTTP attempt {attempt} -> {HTTP_URL}")
            r = requests.get(HTTP_URL, timeout=20)
            r.raise_for_status()
            save_bytes(OUT_FN, r.content)
            print(f"[download] Saved {OUT_FN} via HTTP")
            return True
        except Exception as e:
            print(f"[download] HTTP attempt {attempt} failed: {e}")
            time.sleep(SLEEP)
    return False

def try_ftp():
    for attempt in range(1, RETRIES+1):
        try:
            print(f"[download] FTP attempt {attempt} -> {FTP_HOST}/{FTP_PATH}")
            ftp = ftplib.FTP(FTP_HOST, timeout=30)
            ftp.login()
            ftp.cwd("SymbolDirectory")
            with open(OUT_FN, "wb") as fh:
                ftp.retrbinary("RETR nasdaqlisted.txt", fh.write)
            ftp.quit()
            print(f"[download] Saved {OUT_FN} via FTP")
            return True
        except Exception as e:
            print(f"[download] FTP attempt {attempt} failed: {e}")
            time.sleep(SLEEP)
    return False

def main():
    ok = False
    try:
        ok = try_http()
    except Exception as e:
        print("[download] HTTP raised:", e)
    if not ok:
        print("[download] HTTP failed — trying FTP fallback")
        ok = try_ftp()
    if not ok:
        print("[download] ERROR: could not obtain nasdaqlisted.txt via HTTP or FTP.")
        raise SystemExit(2)

if __name__ == "__main__":
    main()
