"""Drive-as-source-of-truth storage contract.

Drive (MyDrive/s2cd/) holds everything permanently. /content scratch is
disposable and repopulated per session. Only archives move between the two —
never loose files. Every fetch is manifest-checked (md5) so a stale or
partial Drive state is caught at session start instead of mid-training.

Drive layout:
    MyDrive/s2cd/
      raw/oscd/*.zip                       # uploaded manually (see notebook 01)
      processed/chips_{split}.zip          # built by 02, one zip per split
      processed/manifest.json
      runs/<exp>/{last.ckpt,best.ckpt,history.json}
      reports/

Scratch layout:
    /content/oscd/          # extracted raw dataset
    /content/chips/<split>/ # unzipped memmap chips
    /content/runs/<exp>/
    /content/reports/
"""

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

DRIVE_ROOT = Path("/content/drive/MyDrive/s2cd")
SCRATCH = Path("/content")


def _md5(path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class DriveStore:
    """Manifest-driven fetch/push between Drive and session scratch."""

    def __init__(self, drive_root: str | Path = DRIVE_ROOT, scratch: str | Path = SCRATCH):
        self.drive = Path(drive_root)
        self.scratch = Path(scratch)
        if not self.drive.is_dir():
            raise FileNotFoundError(
                f"Drive root not found: {self.drive}. "
                "Mount Drive first (google.colab.drive.mount) or pass drive_root."
            )
        self._ensure_layout()

    # ------------------------------------------------------------------ setup
    def _ensure_layout(self):
        for sub in ["raw/oscd", "processed", "runs", "reports"]:
            (self.drive / sub).mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------- processed
    def read_manifest(self) -> dict:
        path = self.drive / "processed" / "manifest.json"
        if not path.exists():
            raise FileNotFoundError("No manifest.json on Drive — run notebook 02 first.")
        return json.loads(path.read_text())

    def fetch_chips(self, split: str, verify: bool = True) -> Path:
        """Unzip chips_{split}.zip from Drive into scratch. Idempotent per session."""
        local = self.scratch / "chips" / split
        if (local / "t0.npy").exists():
            return local  # already staged this session
        zpath = self.drive / "processed" / f"chips_{split}.zip"
        if not zpath.exists():
            raise FileNotFoundError(f"Missing on Drive: {zpath} — run notebook 02.")
        if verify:
            manifest = self.read_manifest()
            expected = manifest["splits"][split]["md5"]
            got = _md5(zpath)
            if got != expected:
                raise IOError(
                    f"chips_{split}.zip md5 mismatch (expected {expected}, got {got}). "
                    "The Drive copy is stale/partial — rebuild via notebook 02."
                )
        local.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(local)
        return local

    # -------------------------------------------------------------------- runs
    def fetch_run(self, run: str) -> Path:
        local = self.scratch / "runs" / run
        local.mkdir(parents=True, exist_ok=True)
        for name in ["best.ckpt", "last.ckpt", "history.json"]:
            src = self.drive / "runs" / run / name
            if src.exists():
                shutil.copy(src, local / name)
        return local

    def push_run_file(self, run: str, name: str):
        """Atomically push runs/<run>/<name> to Drive (copy to .tmp, then move)."""
        src = self.scratch / "runs" / run / name
        if not src.exists():
            return
        dest_dir = self.drive / "runs" / run
        dest_dir.mkdir(parents=True, exist_ok=True)
        tmp = dest_dir / (name + ".tmp")
        shutil.copy(src, tmp)
        shutil.move(str(tmp), str(dest_dir / name))

    # ----------------------------------------------------------------- reports
    def push_report(self, filename: str) -> Path:
        dest = self.drive / "reports" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.scratch / "reports" / filename, dest)
        return dest

    # ------------------------------------------------------------------- oscd
    def oscd_zips(self) -> list[Path]:
        p = self.drive / "raw" / "oscd"
        zips = sorted(p.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(
                f"No zips in {p}. Download the OSCD zips from IEEE DataPort "
                "(Images, Train Labels, Test Labels) and upload them to Drive."
            )
        return zips

    def extract_oscd(self) -> Path:
        """Extract raw OSCD zips into /content/oscd (idempotent per session)."""
        dest = self.scratch / "oscd"
        marker = dest / ".extracted"
        if marker.exists():
            return dest
        dest.mkdir(parents=True, exist_ok=True)
        for zpath in self.oscd_zips():
            print(f"extracting {zpath.name} ...")
            with zipfile.ZipFile(zpath) as z:
                z.extractall(dest)
        marker.touch()
        return dest