"""Duplicate check before sorting ([sorter] dedupe).

exact:   same file content (SHA-256).
similar: same picture after resizing or re-encoding (64-bit difference hash, Hamming distance).

New images are compared with everything already in the output folder and with images earlier in
the same run; the first copy is kept and later ones go to the duplicates folder. Hashes of the
output folder are cached in <output>/.dedupe_index.json (by size and modification time), so
only new or changed files are read again.
"""
import hashlib
import json
from datetime import datetime

import numpy as np
from PIL import Image

from .config import DEDUPE_INDEX, IGNORE_MARKER
from .files import collect_images, transfer_unique
from .logs import log, progress

LOG_NAME = "duplicates.txt"


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dhash(path):
    """64-bit difference hash: is each pixel brighter than its right neighbour, on a 9x8 thumbnail."""
    with Image.open(path) as image:
        small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(values, h):
    """Bit differences between h and each value in a uint64 array."""
    return np.unpackbits((values ^ np.uint64(h)).view(np.uint8)).reshape(-1, 64).sum(axis=1)


class Deduper:
    def __init__(self, output, dup_dir, mode, distance, transfer_mode):
        self.output, self.dup_dir, self.mode = output, dup_dir, mode
        self.distance, self.transfer_mode = distance, transfer_mode
        self.index_file = output / DEDUPE_INDEX
        self.entries = {}       # key -> (sha, dhash or None); key = path relative to output, or "run:<source>"
        self.by_sha = {}
        self.recent = []        # similar-hash keys added since the last build(), checked one by one
        self.rank = {}          # key -> order seen; a file already sorted only counts as a copy of an earlier one
        self.scan()

    def hashes(self, path):
        sha = file_sha(path)
        if self.mode != "similar":
            return sha, None
        try:
            return sha, dhash(path)
        except Exception as e:
            log(f"Dedupe: cannot decode {path.name}, checking exact copies only: {e}")
            return sha, None

    def scan(self):
        cache = {}
        if self.index_file.exists():
            try:
                cache = json.loads(self.index_file.read_text(encoding="utf-8"))
            except ValueError:
                log(f"Ignoring unreadable {self.index_file}")
        files = collect_images(self.output, self.output)
        for n, path in enumerate(files, 1):
            rel = path.relative_to(self.output).as_posix()
            stat = path.stat()
            hit = cache.get(rel)
            if hit and hit[0] == stat.st_size and hit[1] == stat.st_mtime_ns and (self.mode != "similar" or hit[3] is not None):
                self.add(rel, hit[2], hit[3])
            else:
                try:
                    self.add(rel, *self.hashes(path))
                except Exception as e:
                    log(f"Dedupe: cannot read {rel}: {e}")
            if n % 200 == 0 or n == len(files):
                progress("Hashing sorted images for dedupe", n, len(files))
        self.save()
        self.build()

    def add(self, key, sha, dh):
        self.rank.setdefault(key, len(self.rank))
        self.entries[key] = (sha, dh)
        self.by_sha.setdefault(sha, key)
        if dh is not None:
            self.recent.append(key)

    def drop(self, key):
        sha, _ = self.entries.pop(key, (None, None))
        if sha and self.by_sha.get(sha) == key:
            self.by_sha.pop(sha)
            other = next((k for k, (s, _) in self.entries.items() if s == sha), None)
            if other:
                self.by_sha[sha] = other

    def build(self):
        """Array of similar hashes for fast matching; entries added later are checked one by one."""
        keys = [k for k, (_, dh) in self.entries.items() if dh is not None]
        self.array_keys = keys
        self.array = np.array([self.entries[k][1] for k in keys], dtype=np.uint64)
        self.recent = []

    def earlier(self, other, key):
        """other may be the original of key: new images match anything, sorted ones only earlier files."""
        return other != key and other in self.entries and (key not in self.rank or self.rank.get(other, -1) < self.rank[key])

    def match(self, key, sha, dh):
        found = self.by_sha.get(sha)
        if found and self.earlier(found, key):
            return found, "exact"
        if self.mode != "similar" or dh is None:
            return None
        if len(self.array):
            dist = hamming(self.array, dh)
            for i in np.argsort(dist):
                if dist[i] > self.distance:
                    break
                other = self.array_keys[i]
                if self.earlier(other, key):
                    return other, f"similar ({int(dist[i])} bits)"
        for other in self.recent:
            if self.earlier(other, key):
                bits = bin(self.entries[other][1] ^ dh).count("1")
                if bits <= self.distance:
                    return other, f"similar ({bits} bits)"
        return None

    def key_for(self, path):
        return path.relative_to(self.output).as_posix() if self.output in path.parents else f"run:{path}"

    def filter(self, paths):
        """Split a chunk into images to sort and duplicates; duplicates are transferred right away.
        Returns (paths to sort, duplicate source paths)."""
        keep, dupes = [], []
        for n, path in enumerate(paths, 1):
            key = self.key_for(path)
            try:
                sha, dh = self.entries[key] if key in self.entries else self.hashes(path)
            except Exception as e:
                log(f"Dedupe: cannot read {path.name}, sorting it anyway: {e}")
                keep.append(path)
                continue
            found = self.match(key, sha, dh)
            if found is None:
                self.add(key, sha, dh)
                keep.append(path)
            else:
                self.move(path, key, *found)
                dupes.append(path)
            progress("Checking duplicates", n, len(paths))
        if dupes:
            log(f"{len(dupes)} duplicates -> {self.dup_dir}")
        return keep, dupes

    def move(self, path, key, original, how):
        self.dup_dir.mkdir(parents=True, exist_ok=True)
        (self.dup_dir / IGNORE_MARKER).touch()
        dest = transfer_unique(path, self.dup_dir, self.transfer_mode, path.stem)
        self.drop(key)
        shown = original[4:] if original.startswith("run:") else original
        with (self.dup_dir / LOG_NAME).open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M}\t{dest.name}\t{how} duplicate of\t{shown}\t(from {path})\n")
        log(f"Duplicate: {path.name} -> {dest} ({how} of {shown})")

    def record(self, source, dest):
        """After an image is sorted: track it under its new place in the output folder."""
        old_key = self.key_for(source)
        if old_key not in self.entries or self.output not in dest.parents:
            return
        new_key = dest.relative_to(self.output).as_posix()
        sha, dh = self.entries.pop(old_key)
        self.entries[new_key] = (sha, dh)
        self.rank[new_key] = self.rank.get(old_key, len(self.rank))
        if self.by_sha.get(sha) == old_key:
            self.by_sha[sha] = new_key
        if dh is not None:
            self.recent.append(new_key)

    def save(self):
        data = {}
        for key, (sha, dh) in self.entries.items():
            if key.startswith("run:"):
                continue
            path = self.output / key
            try:
                stat = path.stat()
            except OSError:
                continue
            data[key] = [stat.st_size, stat.st_mtime_ns, sha, dh]
        self.output.mkdir(parents=True, exist_ok=True)
        self.index_file.write_text(json.dumps(data), encoding="utf-8")
        self.build()
