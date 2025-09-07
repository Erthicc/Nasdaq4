# download_nasdaq_list.py
import ftplib, os
ftp = ftplib.FTP("ftp.nasdaqtrader.com")
ftp.login(user="anonymous", passwd="")
ftp.cwd("SymbolDirectory")
for fname in ["nasdaqlisted.txt", "otherlisted.txt"]:
    with open(fname, "wb") as f:
        ftp.retrbinary(f"RETR {fname}", f.write)
ftp.quit()
