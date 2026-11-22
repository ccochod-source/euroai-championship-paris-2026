"""Shared paths and helpers for the watermark-localisation solution (runs on JURECA)."""
import os, sys
W = os.environ.get("WM_ROOT", "/p/scratch/training2666/dasilvalopes3")
KIT = f"{W}/wm/kit"
RELEASE = f"{W}/wm/release"
CLEAN = f"{W}/clean"
SYN = os.environ.get("WM_SYN", f"{W}/syn")
OUT = f"{W}/wm/out"
METHODS = ["pixelseal", "trustmark", "stegastamp"]
MAX_ASPECT = {"pixelseal": 16.0, "stegastamp": 16.0, "trustmark": 2.0}
sys.path.insert(0, KIT)
