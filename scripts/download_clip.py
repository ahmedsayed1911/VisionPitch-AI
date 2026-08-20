"""Fetch a CC-licensed football clip for validation.

Clips are CC BY-SA 4.0 from Wikimedia Commons. Attribution is written next to
the file, and the same attribution must accompany any redistributed derivative
(including annotated output videos).
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

CLIPS = {
    "versailles_nancy": {
        "title": (
            "FC Versailles 78 v. AS Nancy-Lorraine (Championnat National) - 24e journee "
            "- Stade Jean-Bouin (Paris XVI, FR75) - 08-03-2024 157"
        ),
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/5/52/"
            "FC_Versailles_78_v._AS_Nancy-Lorraine_%28Championnat_National%29_-_24e_journ"
            "%C3%A9e_-_Stade_Jean-Bouin_%28Paris_XVI%2C_FR75%29_-_08-03-2024_157.webm"
        ),
        "author": "Manchesterunited1234",
        "license": "CC BY-SA 4.0",
        "source": (
            "https://commons.wikimedia.org/wiki/File:FC_Versailles_78_v._AS_Nancy-Lorraine_"
            "(Championnat_National)_-_24e_journ%C3%A9e_-_Stade_Jean-Bouin_(Paris_XVI,_FR75)_"
            "-_08-03-2024_157.webm"
        ),
        "notes": "French Championnat National (3rd tier), 1920x1080, wide side-on view.",
    },
    "nz_canada_u17": {
        "title": "2018 FIFA U-17 Women's World Cup - New Zealand vs Canada - 25",
        "url": (
            "https://upload.wikimedia.org/wikipedia/commons/f/f9/"
            "2018_FIFA_U-17_Women%27s_World_Cup_-_New_Zealand_vs_Canada_-_25.webm"
        ),
        "author": "NaBUru38",
        "license": "CC BY-SA 4.0",
        "source": (
            "https://commons.wikimedia.org/wiki/File:2018_FIFA_U-17_Women%27s_World_Cup_-_"
            "New_Zealand_vs_Canada_-_25.webm"
        ),
        "notes": "FIFA U-17 Women's World Cup, 1280x720, elevated wide view.",
    },
}

UA = "VisionPitchAI/0.1 (research; https://github.com/) python-urllib"


def fetch(key: str, out_dir: Path) -> Path:
    spec = CLIPS[key]
    webm = out_dir / f"{key}.webm"
    mp4 = out_dir / f"{key}.mp4"

    if not webm.exists():
        print(f"[get ] {spec['title']}")
        request = urllib.request.Request(spec["url"], headers={"User-Agent": UA})
        with urllib.request.urlopen(request, timeout=300) as response:
            webm.write_bytes(response.read())
    print(f"       {webm.name}  {webm.stat().st_size / 1e6:.1f} MB")

    if not mp4.exists():
        # Transcode to H.264 mp4: OpenCV's VP9 support is inconsistent across
        # builds, and a decode failure mid-clip is a confusing way to lose a run.
        print("[conv] transcoding to H.264 mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(webm),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-an",
                str(mp4),
            ],
            check=True,
        )
    print(f"       {mp4.name}  {mp4.stat().st_size / 1e6:.1f} MB")

    (out_dir / f"{key}.ATTRIBUTION.json").write_text(
        json.dumps(spec, indent=2), encoding="utf-8"
    )
    return mp4


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = sys.argv[1:] or list(CLIPS)
    for key in keys:
        if key not in CLIPS:
            print(f"unknown clip {key!r}; choose from {list(CLIPS)}")
            return 1
        fetch(key, out_dir)

    print("\nAll clips are CC BY-SA 4.0. Attribution files are written alongside them.")
    print("Derivative videos you publish must carry the same attribution and licence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
