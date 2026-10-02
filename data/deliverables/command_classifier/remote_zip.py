#!/usr/bin/env python3
"""Read selected members of a remote ZIP (HTTP Range requests) without downloading the whole archive.

    python remote_zip.py <url> --list
    python remote_zip.py <url> --extract <regex> --out <dir>
"""
import argparse, io, re, sys, zipfile
from pathlib import Path

import requests


class HttpFile(io.RawIOBase):
    """Seekable, read-only view of a remote file."""

    def __init__(self, url):
        r = requests.head(url, allow_redirects=True, timeout=60)
        self.url, self.size, self.pos = r.url, int(r.headers["Content-Length"]), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        for attempt in range(5):
            try:
                data = requests.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120).content
                break
            except requests.RequestException:
                if attempt == 4:
                    raise
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--extract")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(a.url), buffer_size=1 << 22))
    if a.list:
        for i in z.infolist():
            print(i.filename, i.file_size)
        return
    rx = re.compile(a.extract)
    sel = [i for i in z.infolist() if rx.search(i.filename) and not i.is_dir()]
    print(f"extracting {len(sel)} files ({sum(i.file_size for i in sel) / 1e6:.0f} MB)", flush=True)
    for k, i in enumerate(sel):
        z.extract(i, a.out)
        if k % 500 == 0:
            print(f"  {k}/{len(sel)}", flush=True)
    print("done")


if __name__ == "__main__":
    sys.exit(main())
