#!/usr/bin/env python3
"""Compare this PC's CPU, GPU and RAM against the community reference specs.

Detects the hardware automatically (the lists in the window are only a fallback).
Standard library only; tkinter is needed for the window (Linux: python3-tk).

    python -I tools/pc_checker.py              # window
    python -I tools/pc_checker.py --cli        # text only
    python -I tools/pc_checker.py --check      # verdict as exit code: 0 good, 1 weak, 2 unknown
    python -I tools/pc_checker.py --check --json --warn

From Python (e.g. a launcher):
    from pc_checker import SpecChecker
    if not SpecChecker().warn_if_weak():   # popup only if the PC is weak
        sys.exit(0)
    python -I tools/pc_checker.py --cli --cpu "Intel Core i5-11400F" \
        --gpu "RTX 3070" --ram 32 --profile 1

The CPU and GPU scores are rough relative estimates made for this tool. They are
not official benchmarks.
Usage: `python -I tools/pc_checker.py --help` lists the options.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception:  # no tkinter: CLI only
    tk = None
    ttk = None
    messagebox = None

IN_RANGE, BELOW, ABOVE, UNKNOWN = "in_range", "below", "above", "unknown"
# Above and in range read the same to the user; the difference stays internal
STATUS_TEXT = {
    IN_RANGE: "Good",
    ABOVE: "Good",
    BELOW: "Weak",
    UNKNOWN: "Unknown",
}
BOOT_NOTE = ("Leonida compatibility and FPS: unknown",
             "The game doesn't boot yet.")

# ----------------------------------------------------------------------------
# GPU scores (gaming performance relative to an RTX 3070 = 100)
# ----------------------------------------------------------------------------
GPU_TABLE = [
    # NVIDIA
    ("NVIDIA GeForce GTX 1050 Ti", 30), ("NVIDIA GeForce GTX 1060", 40),
    ("NVIDIA GeForce GTX 1070", 52), ("NVIDIA GeForce GTX 1070 Ti", 58),
    ("NVIDIA GeForce GTX 1080", 62), ("NVIDIA GeForce GTX 1080 Ti", 78),
    ("NVIDIA GeForce GTX 1650", 36), ("NVIDIA GeForce GTX 1650 Super", 46),
    ("NVIDIA GeForce GTX 1660", 48), ("NVIDIA GeForce GTX 1660 Super", 53),
    ("NVIDIA GeForce GTX 1660 Ti", 55),
    ("NVIDIA GeForce RTX 2060", 62), ("NVIDIA GeForce RTX 2060 Super", 72),
    ("NVIDIA GeForce RTX 2070", 74), ("NVIDIA GeForce RTX 2070 Super", 82),
    ("NVIDIA GeForce RTX 2080", 88), ("NVIDIA GeForce RTX 2080 Super", 91),
    ("NVIDIA GeForce RTX 2080 Ti", 105),
    ("NVIDIA GeForce RTX 3050", 56), ("NVIDIA GeForce RTX 3060", 75),
    ("NVIDIA GeForce RTX 3060 Ti", 92), ("NVIDIA GeForce RTX 3070", 100),
    ("NVIDIA GeForce RTX 3070 Ti", 108), ("NVIDIA GeForce RTX 3080", 125),
    ("NVIDIA GeForce RTX 3080 Ti", 140), ("NVIDIA GeForce RTX 3090", 145),
    ("NVIDIA GeForce RTX 3090 Ti", 155),
    ("NVIDIA GeForce RTX 4060", 88), ("NVIDIA GeForce RTX 4060 Ti", 98),
    ("NVIDIA GeForce RTX 4070", 130), ("NVIDIA GeForce RTX 4070 Super", 148),
    ("NVIDIA GeForce RTX 4070 Ti", 158), ("NVIDIA GeForce RTX 4070 Ti Super", 172),
    ("NVIDIA GeForce RTX 4080", 200), ("NVIDIA GeForce RTX 4080 Super", 207),
    ("NVIDIA GeForce RTX 4090", 270),
    ("NVIDIA GeForce RTX 5050", 82), ("NVIDIA GeForce RTX 5060", 105),
    ("NVIDIA GeForce RTX 5060 Ti", 122), ("NVIDIA GeForce RTX 5070", 150),
    ("NVIDIA GeForce RTX 5070 Ti", 190), ("NVIDIA GeForce RTX 5080", 220),
    ("NVIDIA GeForce RTX 5090", 310),
    # AMD
    ("AMD Radeon RX 570", 33), ("AMD Radeon RX 580", 38), ("AMD Radeon RX 590", 42),
    ("AMD Radeon RX Vega 56", 55), ("AMD Radeon RX Vega 64", 63),
    ("AMD Radeon RX 5500 XT", 40), ("AMD Radeon RX 5600 XT", 62),
    ("AMD Radeon RX 5700", 70), ("AMD Radeon RX 5700 XT", 78),
    ("AMD Radeon RX 6400", 28), ("AMD Radeon RX 6500 XT", 42),
    ("AMD Radeon RX 6600", 70), ("AMD Radeon RX 6600 XT", 82),
    ("AMD Radeon RX 6650 XT", 85), ("AMD Radeon RX 6700", 92),
    ("AMD Radeon RX 6700 XT", 100), ("AMD Radeon RX 6750 XT", 106),
    ("AMD Radeon RX 6800", 115), ("AMD Radeon RX 6800 XT", 130),
    ("AMD Radeon RX 6900 XT", 140), ("AMD Radeon RX 6950 XT", 148),
    ("AMD Radeon RX 7600", 78), ("AMD Radeon RX 7600 XT", 86),
    ("AMD Radeon RX 7700 XT", 115), ("AMD Radeon RX 7800 XT", 135),
    ("AMD Radeon RX 7900 GRE", 150), ("AMD Radeon RX 7900 XT", 170),
    ("AMD Radeon RX 7900 XTX", 195), ("AMD Radeon RX 9060 XT", 120),
    ("AMD Radeon RX 9070", 175), ("AMD Radeon RX 9070 XT", 195),
    # Intel Arc
    ("Intel Arc A380", 30), ("Intel Arc A580", 70), ("Intel Arc A750", 80),
    ("Intel Arc A770", 88), ("Intel Arc B570", 85), ("Intel Arc B580", 100),
]

_GPU_JUNK = re.compile(
    r"\((?:r|tm)\)|[®™]|\b(?:nvidia|geforce|amd|radeon|intel|graphics|corporation|"
    r"laptop|gpu|mobile|max-q|with|design|series|vram)\b|\d+\s?gb|[()\[\]]"
)


_SPLIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")


def _norm_gpu(name: str) -> str:
    s = _GPU_JUNK.sub(" ", name.lower())
    s = _SPLIT.sub(" ", s)                       # "rx9070xt" -> "rx 9070 xt"
    s = re.sub(r"\bti\s*super\b", "ti super", s)
    return re.sub(r"\s+", " ", s).strip()


GPU_INDEX = {_norm_gpu(n): (n, s) for n, s in GPU_TABLE}

# Fallback for cards missing from GPU_TABLE: score by generation and tier,
# the same idea as the CPU tables. Newer generations than the table scale up.
NV_BASE = {
    10: {50: 30, 60: 40, 70: 52, 80: 62}, 16: {50: 36, 60: 48},
    20: {60: 62, 70: 74, 80: 88}, 30: {50: 56, 60: 75, 70: 100, 80: 125, 90: 145},
    40: {60: 88, 70: 130, 80: 200, 90: 270}, 50: {50: 82, 60: 105, 70: 150, 80: 220, 90: 310},
}
RX_BASE = {
    5: {5: 40, 6: 62, 7: 70}, 6: {4: 28, 5: 42, 6: 70, 7: 92, 8: 115, 9: 140},
    7: {6: 78, 7: 115, 8: 135, 9: 150}, 9: {6: 107, 7: 175},
}
ARC_BASE = {1: {3: 30, 5: 70, 7: 85}, 2: {5: 92, 7: 120}}
GEN_GROWTH = 1.25   # assumed uplift per generation beyond the tables
_NV_SUFFIX = {"ti": 1.15, "super": 1.1, "ti super": 1.25}
_RX_SUFFIX = {"xt": 1.12, "xtx": 1.3, "gre": 0.9}
_VARIANT_WORDS = {"ti", "super", "xt", "xtx", "gre"}


def _gen_score(table: dict, series: int, tier: int, step: int):
    top = max(table)
    if series > top:
        row, scale = table[top], GEN_GROWTH ** ((series - top) / step)
    else:
        known = [s for s in table if s <= series]
        row, scale = table[max(known) if known else min(table)], 1.0
    return _pick(row, tier) * scale


def _estimate_gpu(n: str):
    """Score from the model number of a normalised name, or None."""
    m = re.search(r"\b(?:rtx|gtx)\s*(\d{2})(\d{2})\b(?:\s*(ti super|super|ti)\b)?", n)
    if m:
        return (_gen_score(NV_BASE, int(m[1]), int(m[2]), 10)
                * _NV_SUFFIX.get(m[3], 1.0))
    m = re.search(r"\brx\s*(\d{4})\b(?:\s*(xtx|xt|gre)\b)?", n)
    if m:
        num = int(m[1])
        tier = num // 100 % 10 if num < 9000 else num % 100 // 10   # RDNA4: 9060, 9070
        return _gen_score(RX_BASE, num // 1000, tier, 1) * _RX_SUFFIX.get(m[2], 1.0)
    m = re.search(r"\barc\s*([abc])\s*(\d)(\d{2})\b", n)
    if m:
        return _gen_score(ARC_BASE, "abc".index(m[1]) + 1, int(m[2]), 1)
    return None


def gpu_lookup(name: str):
    """(score, exact). exact is False when the score is estimated from the model."""
    if not name or not name.strip():
        return None, False
    raw = name.lower()
    n = _norm_gpu(name)
    if ("uhd" in raw or "iris" in raw or "hd graphics" in raw or n == ""
            or re.fullmatch(r"vega \d+", n)):
        return 15, True  # integrated graphics
    mult = 0.8 if ("laptop" in raw or "mobile" in raw or "max-q" in raw) else 1.0
    best = None
    for key, (_disp, score) in GPU_INDEX.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", n):
            if best is None or len(key) > len(best[0]):
                best = (key, score)
    if best is not None:
        # "rx 9070 gre" matches "rx 9070": a leftover variant suffix means the
        # listed score is for a different card, so estimate instead
        if set(n.replace(best[0], "", 1).split()) & _VARIANT_WORDS:
            est = _estimate_gpu(n)
            if est is not None:
                return est * mult, False
            return best[1] * mult, False
        return best[1] * mult, True
    est = _estimate_gpu(n)
    return (est * mult, False) if est is not None else (None, False)


def gpu_score(name: str):
    """Relative GPU score, or None if the name isn't recognised."""
    return gpu_lookup(name)[0]


# ----------------------------------------------------------------------------
# CPU scores (relative to a Ryzen 5 5600 = 100). Known models are listed;
# the rest are estimated from generation and tier.
# ----------------------------------------------------------------------------
CPU_EXPLICIT = {
    "i3-12100": 85, "i5-12400": 102, "i5-12600": 118, "i5-12600k": 125,
    "i7-12700": 135, "i7-12700k": 140, "i9-12900k": 155,
    "i5-13400": 115, "i5-13500": 130, "i5-13600k": 150, "i7-13700k": 170,
    "i9-13900k": 195, "i5-14400": 118, "i5-14600k": 155, "i7-14700k": 180,
    "i9-14900k": 200, "i5-11400": 92, "i5-11600k": 100, "i7-11700": 103,
    "i7-10700": 98, "i5-10400": 82, "i5-10600k": 92, "i9-10900k": 108,
    "i7-9700k": 93, "i9-9900k": 105, "i5-9400": 75, "i7-8700k": 90, "i5-8400": 78,
    "r5-5600": 100, "r5-5600x": 104, "r7-3700x": 100, "r5-3600": 90,
    "r5-5500": 90, "r7-5700x": 115, "r7-5800x": 120, "r7-5800x3d": 140,
    "r9-5900x": 135, "r5-7600": 130, "r7-7800x3d": 175, "r7-9800x3d": 190,
}
INTEL_BASE = {
    4: {3: 40, 5: 48, 7: 56, 9: 56}, 6: {3: 45, 5: 55, 7: 62, 9: 62},
    7: {3: 52, 5: 60, 7: 70, 9: 70}, 8: {3: 68, 5: 80, 7: 90, 9: 100},
    9: {3: 72, 5: 85, 7: 95, 9: 108}, 10: {3: 75, 5: 88, 7: 98, 9: 108},
    11: {3: 80, 5: 95, 7: 103, 9: 112}, 12: {3: 85, 5: 108, 7: 135, 9: 155},
    13: {3: 100, 5: 125, 7: 165, 9: 195}, 14: {3: 100, 5: 130, 7: 175, 9: 200},
}
AMD_BASE = {
    1: {3: 45, 5: 60, 7: 70, 9: 70}, 2: {3: 55, 5: 68, 7: 80, 9: 80},
    3: {3: 70, 5: 90, 7: 100, 9: 120}, 4: {3: 75, 5: 85, 7: 95, 9: 95},
    5: {3: 85, 5: 100, 7: 118, 9: 138}, 7: {3: 100, 5: 130, 7: 148, 9: 175},
    8: {3: 100, 5: 120, 7: 135, 9: 150}, 9: {3: 115, 5: 140, 7: 155, 9: 190},
}
ULTRA_BASE = {3: 130, 5: 150, 7: 178, 9: 195}
# Intel mobile/low-power suffix -> multiplier (first matching prefix wins)
_MOBILE_MULT = [("hx", 1.0), ("hk", 0.9), ("h", 0.85), ("u", 0.55), ("p", 0.8),
                ("g", 0.6), ("t", 0.8)]


def _pick(table: dict, key: int):
    """Row for `key`, or the closest lower one (the lowest if there is none)."""
    ks = sorted(table)
    lower = [k for k in ks if k <= key]
    return table[lower[-1] if lower else ks[0]]


def _mobile_mult(suffix: str) -> float:
    for prefix, mult in _MOBILE_MULT:
        if suffix.startswith(prefix):
            return mult
    return 1.0


def cpu_lookup(name: str):
    """(score, exact). exact is False when estimated from generation and tier."""
    if not name or not name.strip():
        return None, False
    s = re.sub(r"\((?:r|tm)\)|[®™]", "", name.lower())

    m = re.search(r"core\s*ultra\s*([3579])\s*(\d{3})([a-z]*)", s)
    if m:
        base = ULTRA_BASE[int(m[1])]
        return (base if "k" in m[3] else base * 0.95), False

    m = re.search(r"\bi([3579])[- ]?(\d{4,5})([a-z]{0,2})\d?\b", s)
    if m:
        tier, num, suf = int(m[1]), m[2], m[3]
        gen = int(num[:2]) if (len(num) == 5 or num[0] == "1") else int(num[0])
        is_k = suf in ("k", "kf", "ks")
        key = f"i{tier}-{num}{'k' if is_k else ''}"
        if key in CPU_EXPLICIT:
            return CPU_EXPLICIT[key], True
        base = _pick(INTEL_BASE, gen)[tier]
        return base * (1.06 if is_k else 1.0) * _mobile_mult(suf), False

    m = re.search(
        r"ryzen\s*(?:ai\s*)?([3579])\s*(?:pro\s*)?(\d{4})\s*"
        r"(x3d|xt|x|gt|ge|g|f|hx|hs|h|u)?\b", s)
    if m:
        tier, num, suf = int(m[1]), m[2], m[3] or ""
        sufkey = "x3d" if suf == "x3d" else ("x" if suf in ("x", "xt") else "")
        key = f"r{tier}-{num}{sufkey}"
        if key in CPU_EXPLICIT:
            return CPU_EXPLICIT[key], True
        base = _pick(AMD_BASE, int(num[0]))[tier]
        mult = {"x3d": 1.15, "x": 1.04, "xt": 1.04, "g": 0.95, "ge": 0.9, "gt": 0.95,
                "hx": 1.0, "hs": 0.8, "h": 0.85, "u": 0.55}.get(suf, 1.0)
        return base * mult, False
    return None, False


def cpu_score(name: str):
    """Relative CPU score, or None if the name isn't recognised."""
    return cpu_lookup(name)[0]


CPU_CHOICES = [
    "Intel Core i3-10100F", "Intel Core i5-10400F", "Intel Core i7-10700K",
    "Intel Core i5-11400F", "Intel Core i7-11700K", "Intel Core i3-12100F",
    "Intel Core i5-12400F", "Intel Core i5-12600K", "Intel Core i7-12700K",
    "Intel Core i9-12900K", "Intel Core i5-13400F", "Intel Core i5-13600K",
    "Intel Core i7-13700K", "Intel Core i9-13900K", "Intel Core i5-14400F",
    "Intel Core i5-14600K", "Intel Core i7-14700K", "Intel Core i9-14900K",
    "Intel Core Ultra 5 245K", "Intel Core Ultra 7 265K", "Intel Core Ultra 9 285K",
    "AMD Ryzen 5 2600", "AMD Ryzen 5 3600", "AMD Ryzen 7 3700X", "AMD Ryzen 5 5500",
    "AMD Ryzen 5 5600", "AMD Ryzen 5 5600X", "AMD Ryzen 7 5700X", "AMD Ryzen 7 5800X",
    "AMD Ryzen 7 5800X3D", "AMD Ryzen 9 5900X", "AMD Ryzen 5 7600", "AMD Ryzen 7 7700X",
    "AMD Ryzen 7 7800X3D", "AMD Ryzen 9 7950X", "AMD Ryzen 5 9600X",
    "AMD Ryzen 7 9700X", "AMD Ryzen 7 9800X3D", "AMD Ryzen 9 9950X",
]
GPU_CHOICES = [n for n, _ in GPU_TABLE]

# ----------------------------------------------------------------------------
# Reference profiles
# ----------------------------------------------------------------------------
PROFILES = {
    "Xbox Series X": {
        "short": "Xbox Series X",
        "subtitle": "Community comparison with Xbox Series X",
        "cpu_ref": ["AMD Ryzen 7 3700X", "AMD Ryzen 5 5600"],
        "cpu_text": "Ryzen 7 3700X or Ryzen 5 5600",
        "gpu_ref": ["NVIDIA GeForce RTX 3060 Ti", "NVIDIA GeForce RTX 3070",
                    "AMD Radeon RX 6700 XT", "AMD Radeon RX 6800"],
        "gpu_text": "RTX 3060 Ti / 3070 or RX 6700 XT / 6800",
        "ram": 32, "ram_text": "32 GB suggested for PC",
    },
    "Community 1080p Low estimate": {
        "short": "the 1080p Low estimate",
        "subtitle": "Community estimate for 1080p Low",
        "cpu_ref": ["Intel Core i5-12400F", "AMD Ryzen 5 5600"],
        "cpu_text": "i5-12400F or Ryzen 5 5600",
        "gpu_ref": ["NVIDIA GeForce RTX 3060", "AMD Radeon RX 6700 XT"],
        "gpu_text": "RTX 3060 12 GB or RX 6700 XT",
        "ram": 16, "ram_text": "16 GB",
    },
}
CPU_TOL, GPU_TOL = 0.10, 0.05


def _judge(score, lo, hi, tol):
    if score is None:
        return UNKNOWN
    if score < lo * (1 - tol):
        return BELOW
    if score > hi * (1 + tol):
        return ABOVE
    return IN_RANGE


def _judge_ram(ram, ref):
    if ram is None:
        return UNKNOWN
    if ram < ref * 0.9:
        return BELOW
    if ram <= ref * 1.1:
        return IN_RANGE
    return ABOVE


def _join(items):
    """["CPU", "GPU", "RAM"] -> "CPU, GPU and RAM"."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def evaluate(profile_name: str, cpu: str, gpu: str, ram):
    """Compare the hardware with a profile. Returns (headline, subtitle, rows)."""
    p = PROFILES[profile_name]
    cs = [cpu_score(n) for n in p["cpu_ref"]]
    gs = [gpu_score(n) for n in p["gpu_ref"]]
    cpu_s, cpu_exact = cpu_lookup(cpu)
    gpu_s, gpu_exact = gpu_lookup(gpu)
    rows = [
        ("CPU", _judge(cpu_s, min(cs), max(cs), CPU_TOL), p["cpu_text"], cpu.strip(),
         cpu_s is not None and not cpu_exact),
        ("GPU", _judge(gpu_s, min(gs), max(gs), GPU_TOL), p["gpu_text"], gpu.strip(),
         gpu_s is not None and not gpu_exact),
        ("RAM", _judge_ram(ram, p["ram"]), p["ram_text"], f"{ram} GB" if ram else "", False),
    ]
    weak = [r[0] for r in rows if r[1] == BELOW]
    unknown = [r[3] or r[0] for r in rows if r[1] == UNKNOWN]
    if weak:
        verb = "is" if len(weak) == 1 else "are"
        return ("Probably not enough",
                f"The {_join(weak)} {verb} weaker than {p['short']}.", rows)
    if unknown:
        return ("Can't tell yet",
                f"Not recognised: {_join(unknown)}. Pick it from the list to compare.",
                rows)
    return ("Looks good", p["subtitle"], rows)


def result_as_text(profile_name, headline, sub, rows):
    lines = [f"Leonida PC Checker - vs {profile_name}", headline, sub, ""]
    for label, st, ref, mine, est in rows:
        note = ", score estimated" if est else ""
        lines.append(f"{label}: {STATUS_TEXT[st]} (ref: {ref}; your PC: {mine or '?'}{note})")
    lines += ["", f"{BOOT_NOTE[0]} ({BOOT_NOTE[1].rstrip('.')})."]
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Hardware detection
# ----------------------------------------------------------------------------
def _run(cmd, timeout=10) -> str:
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="ignore", timeout=timeout, **kw)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def clean_name(name: str) -> str:
    s = re.sub(r"\((?:r|tm)\)|[®™]", "", name, flags=re.I)
    s = re.sub(r"@.*$", "", s)
    s = re.sub(r"\b\d+-core processor\b|\bprocessor\b|\bcpu\b", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def detect_cpu() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return clean_name(winreg.QueryValueEx(k, "ProcessorNameString")[0])
        if sys.platform == "darwin":
            out = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
            if out:
                return clean_name(out)
        else:
            with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return clean_name(line.split(":", 1)[1])
    except Exception:
        pass
    return clean_name(platform.processor() or "")


def _win_vram_map() -> dict:
    """VRAM per adapter from the registry (Win32_VideoController caps at 4 GB)."""
    found = {}
    try:
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            for i in range(32):
                try:
                    with winreg.OpenKey(root, f"{i:04d}") as k:
                        desc = winreg.QueryValueEx(k, "DriverDesc")[0]
                        val = winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")[0]
                        if isinstance(val, bytes):
                            val = int.from_bytes(val, "little")
                        found[desc.lower()] = val / 2**30
                except OSError:
                    continue
    except Exception:
        pass
    return found


def detect_gpus() -> list:
    """All GPUs found, as (name, vram_gb or None)."""
    gpus = []
    smi = shutil.which("nvidia-smi")
    if smi:
        out = _run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0]:
                continue
            try:
                gpus.append((parts[0], round(float(parts[1]) / 1024)))
            except (IndexError, ValueError):
                gpus.append((parts[0], None))

    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "Get-CimInstance Win32_VideoController | "
                    "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"])
        try:
            data = json.loads(out) if out else []
            data = [data] if isinstance(data, dict) else data
        except ValueError:
            data = []
        vmap = _win_vram_map()
        for d in data:
            name = (d.get("Name") or "").strip()
            if not name or any(name.lower() == g[0].lower() for g in gpus):
                continue
            vram = vmap.get(name.lower())
            if vram is None and d.get("AdapterRAM"):
                vram = d["AdapterRAM"] / 2**30
            gpus.append((name, round(vram) if vram else None))
    elif sys.platform == "darwin":
        out = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=20)
        try:
            for d in json.loads(out).get("SPDisplaysDataType", []):
                if d.get("sppci_model"):
                    gpus.append((d["sppci_model"], None))
        except ValueError:
            pass
    else:
        for line in _run(["lspci", "-mm"]).splitlines():
            try:
                tok = shlex.split(line)
            except ValueError:
                continue
            if len(tok) < 4 or not re.search(r"vga|3d|display", tok[1], re.I):
                continue
            vend = tok[2].lower()
            vname = ("NVIDIA" if "nvidia" in vend else
                     "AMD" if ("amd" in vend or "advanced micro" in vend) else
                     "Intel" if "intel" in vend else tok[2])
            br = re.search(r"\[(.+)\]", tok[3])
            name = f"{vname} {br.group(1) if br else tok[3]}"
            if vname == "NVIDIA" and any("nvidia" in g[0].lower() for g in gpus):
                continue  # already reported by nvidia-smi
            gpus.append((name, None))
    return gpus


def _total_ram_bytes():
    try:
        if sys.platform == "win32":
            class MemStat(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MemStat()
            st.dwLength = ctypes.sizeof(st)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullTotalPhys
        if sys.platform == "darwin":
            return int(_run(["sysctl", "-n", "hw.memsize"]))
        with open("/proc/meminfo", encoding="ascii") as f:
            return int(f.readline().split()[1]) * 1024
    except Exception:
        return None


def detect_ram_gb():
    total = _total_ram_bytes()
    if not total:
        return None
    gb = total / 2**30
    # The OS reports a bit less than installed (31.8 GB for a 32 GB kit)
    for common in (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128):
        if abs(gb - common) / common < 0.08:
            return common
    return max(4, min(128, round(gb)))


def detect_all() -> dict:
    best = None
    for name, vram in detect_gpus():  # with several GPUs, use the strongest
        sc = gpu_score(name)
        sc = -1 if sc is None else sc
        if best is None or sc > best[0]:
            best = (sc, name, vram)
    gpu = ""
    if best:
        gpu = clean_name(best[1])
        if best[2]:
            gpu += f" ({best[2]} GB VRAM)"
    return {"cpu": detect_cpu(), "gpu": gpu, "ram": detect_ram_gb()}


# ----------------------------------------------------------------------------
# Reusable API (for a launcher or a future boot check)
# ----------------------------------------------------------------------------
SKIP_ENV = "LEONIDA_SKIP_SPEC_CHECK"
GOOD, WEAK = "good", "weak"      # Report.verdict; the third value is UNKNOWN


@dataclass
class Check:
    component: str
    status: str
    reference: str
    detected: str
    estimated: bool = False


@dataclass
class Report:
    profile: str
    headline: str
    summary: str
    checks: list

    @property
    def verdict(self) -> str:
        statuses = [c.status for c in self.checks]
        if BELOW in statuses:
            return WEAK
        return UNKNOWN if UNKNOWN in statuses else GOOD

    @property
    def weak_parts(self) -> list:
        return [c.component for c in self.checks if c.status == BELOW]

    def as_text(self) -> str:
        rows = [(c.component, c.status, c.reference, c.detected, c.estimated)
                for c in self.checks]
        return result_as_text(self.profile, self.headline, self.summary, rows)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["verdict"] = self.verdict
        return d


class SpecChecker:
    """Compare a PC with a reference profile.

        checker = SpecChecker()
        if not checker.warn_if_weak():
            sys.exit(0)          # the player chose not to launch
    """

    def __init__(self, profile=None):
        self.profile = profile or next(iter(PROFILES))
        if self.profile not in PROFILES:
            raise ValueError(f"unknown profile {self.profile!r}; use one of {list(PROFILES)}")

    def check(self, cpu=None, gpu=None, ram=None) -> Report:
        """Anything left as None is detected from this PC."""
        if cpu is None or gpu is None or ram is None:
            info = detect_all()
            cpu = info["cpu"] if cpu is None else cpu
            gpu = info["gpu"] if gpu is None else gpu
            ram = info["ram"] if ram is None else ram
        headline, summary, rows = evaluate(self.profile, cpu, gpu, ram)
        return Report(self.profile, headline, summary, [Check(*r) for r in rows])

    def warn_if_weak(self, report=None) -> bool:
        """Show a popup if the PC is weak. Returns False only if the player
        chooses not to launch; it never blocks otherwise. Set
        LEONIDA_SKIP_SPEC_CHECK=1 to skip the check."""
        if os.environ.get(SKIP_ENV):
            return True
        report = report or self.check()
        if report.verdict != WEAK:
            return True
        msg = f"Leonida may run poorly on this PC.\n\n{report.summary}"
        if tk is None or messagebox is None:
            print(msg, file=sys.stderr)
            return True
        try:
            root = tk.Tk()
            root.withdraw()
            try:
                return messagebox.askyesno("Leonida", msg + "\n\nLaunch anyway?",
                                           icon="warning")
            finally:
                root.destroy()
        except tk.TclError:          # no display available
            print(msg, file=sys.stderr)
            return True


# ----------------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------------
BG = "#161029"
CARD_L, BORDER_L = "#12162c", "#3b6b80"      # teal side: your PC
CARD_R, BORDER_R = "#2b1a3d", "#7a4378"      # pink side: result
DIVIDER, FIELD = "#5b3d63", "#0b0e1f"
FG, MUTED = "#f4f1fb", "#b7aecd"
PINK, MINT, ORANGE, GRAY = "#ff9fd2", "#7ff0d0", "#ffc078", "#9b97ad"
STATUS_COLOR = {IN_RANGE: MINT, ABOVE: MINT, BELOW: ORANGE, UNKNOWN: GRAY}
FONT = "Segoe UI"
NOTE = ("CPU and GPU scores are rough estimates, not benchmarks. Reference ranges "
        "come from community comparisons.")


def _lerp(a: str, b: str, t: float) -> str:
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


class App:
    def __init__(self, root):
        self.root = root
        root.title("Leonida PC Checker")
        root.geometry("1000x640")
        root.minsize(900, 600)
        root.configure(bg=BG)
        self.q = queue.Queue()
        self.last_text = ""
        self.wrapped = []        # (label, margin) pairs re-wrapped on resize
        self._style()

        self.profile_var = tk.StringVar(value=list(PROFILES)[0])
        self.cpu_var = tk.StringVar()
        self.gpu_var = tk.StringVar()
        self.ram_var = tk.StringVar(value="16")

        self._stripe()
        self._title()
        body = tk.Frame(root, bg=BG)
        body.pack(fill="both", expand=True, padx=40, pady=(0, 32))
        body.columnconfigure(0, weight=1, uniform="c")
        body.columnconfigure(1, weight=1, uniform="c")
        body.rowconfigure(0, weight=1)
        self._left(body)
        self._right(body)

        for v in (self.profile_var, self.cpu_var, self.gpu_var, self.ram_var):
            v.trace_add("write", self.refresh)
        self.refresh()
        self.start_detect()

    def _style(self):
        st = ttk.Style()
        try:
            st.theme_use("clam")
        except tk.TclError:
            pass
        for name in ("TCombobox", "TSpinbox"):
            st.configure(name, fieldbackground=FIELD, background=FIELD, foreground=FG,
                         arrowcolor=MINT, bordercolor=BORDER_L, lightcolor=FIELD,
                         darkcolor=FIELD, padding=6)
            st.map(name, bordercolor=[("focus", MINT)])
        st.map("TCombobox", fieldbackground=[("readonly", FIELD)],
               foreground=[("readonly", FG)], selectbackground=[("readonly", FIELD)],
               selectforeground=[("readonly", FG)])
        self.root.option_add("*TCombobox*Listbox.background", FIELD)
        self.root.option_add("*TCombobox*Listbox.foreground", FG)
        self.root.option_add("*TCombobox*Listbox.selectBackground", BORDER_L)
        self.root.option_add("*TCombobox*Listbox.font", (FONT, 11))

    def _stripe(self):
        """Thin pink-to-mint strip across the top."""
        c = tk.Canvas(self.root, height=4, bg=BG, highlightthickness=0)
        c.pack(fill="x")

        def draw(e):
            c.delete("all")
            step = 4
            for x in range(0, e.width, step):
                c.create_rectangle(x, 0, x + step, 4, width=0,
                                   fill=_lerp(PINK, MINT, x / max(e.width, 1)))
        c.bind("<Configure>", draw)

    def _title(self):
        t = tk.Frame(self.root, bg=BG)
        t.pack(fill="x", padx=40, pady=(24, 18))
        f = (FONT, 26, "bold")
        tk.Label(t, text="Can my PC run ", font=f, fg=FG, bg=BG).pack(side="left")
        tk.Label(t, text="Leonida?", font=f, fg=PINK, bg=BG).pack(side="left")

    def _card(self, parent, col, bg, border, padx):
        card = tk.Frame(parent, bg=bg, highlightbackground=border, highlightthickness=1,
                        padx=26, pady=22)
        card.grid(row=0, column=col, sticky="nsew", padx=padx)
        return card

    def _text(self, parent, bg, text="", size=10, bold=False, fg=MUTED, pady=0):
        lb = tk.Label(parent, text=text, bg=bg, fg=fg, anchor="w", justify="left",
                      font=(FONT, size, "bold" if bold else "normal"))
        lb.pack(fill="x", pady=pady)
        self.wrapped.append((lb, 60))
        return lb

    def _rewrap(self, card):
        w = card.winfo_width() - 60
        if w > 100:
            for lb, _ in self.wrapped:
                if lb.master is card or str(lb).startswith(str(card)):
                    lb.config(wraplength=w)

    def _left(self, body):
        card = self._card(body, 0, CARD_L, BORDER_L, (0, 12))

        self.btn = tk.Button(card, text="Detect my PC", font=(FONT, 11, "bold"), bg=MINT,
                             fg="#0b1f1a", activebackground="#a6f6e0", relief="flat",
                             cursor="hand2", pady=9, command=self.start_detect)
        self.btn.pack(fill="x")
        self.status = self._text(card, CARD_L, size=9, pady=(8, 0))

        def label(text):
            tk.Label(card, text=text, font=(FONT, 10), fg=MUTED, bg=CARD_L, anchor="w"
                     ).pack(fill="x", pady=(16, 4))

        label("Compare against")
        ttk.Combobox(card, textvariable=self.profile_var, values=list(PROFILES),
                     state="readonly", font=(FONT, 11)).pack(fill="x")
        label("CPU")
        ttk.Combobox(card, textvariable=self.cpu_var, values=CPU_CHOICES,
                     font=(FONT, 11)).pack(fill="x")
        label("GPU")
        ttk.Combobox(card, textvariable=self.gpu_var, values=GPU_CHOICES,
                     font=(FONT, 11)).pack(fill="x")

        row = tk.Frame(card, bg=CARD_L)
        row.pack(fill="x", pady=(16, 0))
        tk.Label(row, text="RAM", font=(FONT, 10), fg=MUTED, bg=CARD_L).pack(side="left")
        tk.Label(row, text="GB", font=(FONT, 11, "bold"), fg=FG, bg=CARD_L
                 ).pack(side="right")
        ttk.Spinbox(row, from_=4, to=128, width=5, textvariable=self.ram_var,
                    font=(FONT, 11, "bold")).pack(side="right", padx=6)
        card.bind("<Configure>", lambda e: self._rewrap(card))

    def _right(self, body):
        card = self._card(body, 1, CARD_R, BORDER_R, (12, 0))

        self.headline = self._text(card, CARD_R, size=22, bold=True, fg=FG)
        self.sub = self._text(card, CARD_R, pady=(6, 4))

        self.rows = []
        for name in ("CPU", "GPU", "RAM"):
            tk.Frame(card, height=1, bg=DIVIDER).pack(fill="x", pady=(10, 0))
            f = tk.Frame(card, bg=CARD_R)
            f.pack(fill="x", pady=(8, 0))
            top = tk.Frame(f, bg=CARD_R)
            top.pack(fill="x")
            tk.Label(top, text=name, font=(FONT, 11, "bold"), fg=FG, bg=CARD_R
                     ).pack(side="left")
            res = tk.Label(top, text="", font=(FONT, 11, "bold"), bg=CARD_R)
            res.pack(side="right")
            ref = self._text(f, CARD_R, fg=FG)
            mine = self._text(f, CARD_R, size=9)
            self.rows.append((res, ref, mine))

        tk.Frame(card, height=1, bg=DIVIDER).pack(fill="x", pady=(12, 0))
        self._text(card, CARD_R, BOOT_NOTE[0], bold=True, fg=FG, pady=(10, 0))
        self._text(card, CARD_R, BOOT_NOTE[1])

        self.copy_btn = tk.Button(card, text="Copy result", font=(FONT, 10), bg="#3d2650",
                                  fg=FG, activebackground="#52346b", activeforeground=FG,
                                  relief="flat", cursor="hand2", pady=6, command=self.copy)
        self.copy_btn.pack(side="bottom", fill="x", pady=(10, 0))
        self._text(card, CARD_R, NOTE, size=8, pady=(12, 0)).pack_configure(side="bottom")
        card.bind("<Configure>", lambda e: self._rewrap(card))

    def get_ram(self):
        try:
            v = int(float(self.ram_var.get()))
            return v if v > 0 else None
        except ValueError:
            return None

    def refresh(self, *_):
        name = self.profile_var.get()
        headline, sub, rows = evaluate(name, self.cpu_var.get(), self.gpu_var.get(),
                                       self.get_ram())
        self.headline.config(text=headline)
        self.sub.config(text=sub)
        for (res, ref, mine), (_lbl, st, ref_text, mine_text, est) in zip(self.rows, rows):
            res.config(text=STATUS_TEXT[st], fg=STATUS_COLOR[st])
            ref.config(text=f"Reference: {ref_text}")
            mine.config(text=f"Your PC: {mine_text or 'no data'}"
                        + (" (score estimated)" if est else ""))
        self.last_text = result_as_text(name, headline, sub, rows)

    def copy(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.last_text)
        self.copy_btn.config(text="Copied")
        self.root.after(1500, lambda: self.copy_btn.config(text="Copy result"))

    # Detection runs in a thread so the window doesn't freeze
    def start_detect(self):
        self.btn.config(state="disabled")
        self.status.config(text="Detecting...", fg=MUTED)

        def work():
            try:
                info = detect_all()
            except Exception:
                info = {"cpu": "", "gpu": "", "ram": None}
            self.q.put(info)

        threading.Thread(target=work, daemon=True).start()
        self.root.after(150, self._poll)

    def _poll(self):
        try:
            info = self.q.get_nowait()
        except queue.Empty:
            self.root.after(150, self._poll)
            return
        self.btn.config(state="normal")
        missing = []
        for key, var, label in (("cpu", self.cpu_var, "CPU"), ("gpu", self.gpu_var, "GPU"),
                                ("ram", self.ram_var, "RAM")):
            if info[key]:
                var.set(str(info[key]))
            else:
                missing.append(label)
        if missing:
            self.status.config(text=f"Couldn't detect {', '.join(missing)}. "
                                    "Pick from the lists.", fg=ORANGE)
        elif gpu_score(info["gpu"]) is None or cpu_score(info["cpu"]) is None:
            self.status.config(text="Detected, but a component isn't recognised. "
                                    "Pick the closest match from the list.", fg=ORANGE)
        else:
            self.status.config(text="Detected.", fg=MINT)


def run_gui():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    root = tk.Tk()
    App(root)
    root.mainloop()


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def _pick_profile(n: int) -> str:
    names = list(PROFILES)
    return names[max(1, min(n, len(names))) - 1]


def run_check(args) -> int:
    """No window. Exit code: 0 good, 1 weak, 2 unknown, 3 player declined (--warn)."""
    checker = SpecChecker(_pick_profile(args.profile))
    report = checker.check(args.cpu, args.gpu, args.ram)
    print(json.dumps(report.as_dict(), indent=2) if args.json else report.as_text())
    if args.warn and not checker.warn_if_weak(report):
        return 3
    return {GOOD: 0, WEAK: 1}.get(report.verdict, 2)


def run_cli(args):
    profile = _pick_profile(args.profile)
    info = detect_all() if not (args.cpu and args.gpu and args.ram) else {}
    cpu = args.cpu or info.get("cpu", "")
    gpu = args.gpu or info.get("gpu", "")
    ram = args.ram or info.get("ram")
    print(f"PC -> CPU: {cpu or '?'} | GPU: {gpu or '?'} | RAM: {ram or '?'} GB\n")
    print(result_as_text(profile, *evaluate(profile, cpu, gpu, ram)))


def main():
    ap = argparse.ArgumentParser(description="Compare this PC with the Leonida reference specs")
    ap.add_argument("--cli", action="store_true", help="text mode, no window")
    ap.add_argument("--profile", type=int, default=1,
                    help="1 = Xbox Series X, 2 = community 1080p Low estimate")
    ap.add_argument("--cpu", help="override the detected CPU")
    ap.add_argument("--gpu", help="override the detected GPU")
    ap.add_argument("--ram", type=int, help="override the detected RAM, in GB")
    ap.add_argument("--check", action="store_true",
                    help="no window; exit code 0 good, 1 weak, 2 unknown (3 with --warn if declined)")
    ap.add_argument("--json", action="store_true", help="with --check: print JSON")
    ap.add_argument("--warn", action="store_true",
                    help="with --check: show a popup when the PC is weak")
    args = ap.parse_args()
    if args.check:
        sys.exit(run_check(args))
    if args.cli or tk is None:
        if tk is None and not args.cli:
            print("tkinter isn't available; using text mode.\n")
        run_cli(args)
    else:
        run_gui()


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main()