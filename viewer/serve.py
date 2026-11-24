"""Serve the map viewer on http://127.0.0.1:8765 (local only).

Usage:
    python -I serve.py [--world DIR] [--port 8765]

`--world` is the folder holding world.json and the .glb chunks, typically <work>\\world. It's served at
/world/ so game-derived files never need to live in this repo. Without it, the viewer shows test terrain.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))

_WINDOWS_RESERVED_DEVICES = frozenset({
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    # Windows also treats superscript 1/2/3 as device numbers. Keep this fallback
    # complete for Python 3.12, which has no os.path.isreserved.
    *(f"COM{i}" for i in "123456789\u00b9\u00b2\u00b3"),
    *(f"LPT{i}" for i in "123456789\u00b9\u00b2\u00b3"),
})


def _is_reserved_segment(seg: str) -> bool:
    # Spaces before an extension do not make a device into a normal filename:
    # Windows treats e.g. "NUL .glb" as NUL (as does ntpath.isreserved in Python 3.13+).
    stem = seg.split(".", 1)[0].rstrip(" ").upper()
    if stem in _WINDOWS_RESERVED_DEVICES:
        return True
    isreserved = getattr(os.path, "isreserved", None)
    if callable(isreserved) and isreserved(seg):
        return True
    return False


# Binding to 127.0.0.1 keeps other machines out, but not other web pages: a page on any site can point its own
# domain at 127.0.0.1 (DNS rebinding) and read /world/ through this server as same-origin. Its requests still carry
# that domain in the Host header, so only requests that name this machine are answered.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


def host_allowed(host: str | None) -> bool:
    """True when a request's Host header names the loopback address. Browsers always send Host; a request without
    one comes from a local tool, not a web page, and is allowed.

    The value must be `name` or `name:port` (RFC 7230 section 5.4): text after an IPv6 literal's `]` other than
    `:port`, a second colon, or a port that is not 1-5 ASCII digits up to 65535 is malformed and refused."""
    if host is None:
        return True
    host = host.strip().lower()
    if host.startswith("["):                        # [::1] or [::1]:8765
        end = host.find("]")
        if end < 0:
            return False
        name, rest = host[:end + 1], host[end + 1:]
        if rest and not rest.startswith(":"):
            return False                            # [::1]junk, [::1].
        port = rest[1:] if rest else None
    else:
        name, colon, port = host.partition(":")
        if not colon:
            port = None
        elif ":" in port:
            return False                            # a bare IPv6 address, or name:port:more
        if name.endswith("."):                      # one trailing dot: the fully qualified form
            name = name[:-1]
    # The length cap comes before int(): a port of thousands of digits would hit Python's int conversion limit.
    if port is not None and not (0 < len(port) <= 5 and port.isascii() and port.isdigit() and int(port) <= 65535):
        return False
    return name in _LOOPBACK_HOSTS


def safe_world_path(world_dir: str, req_path: str) -> str | None:
    """Resolve a /world/... request path safely inside world_dir, or return None."""
    if not world_dir or not isinstance(req_path, str):
        return None
    clean = req_path.split("#", 1)[0].split("?", 1)[0]
    if not clean.startswith("/world/"):
        return None
    rel = urllib.parse.unquote(clean[len("/world/"):])
    if not rel or "\x00" in rel:
        return None
    if rel.startswith(("/", "\\")) or rel.endswith(("/", "\\")):
        return None
    norm = rel.replace("\\", "/")
    parts = norm.split("/")
    for seg in parts:
        if not seg or seg in (".", "..") or ":" in seg or _is_reserved_segment(seg):
            return None
    if os.path.isabs(rel) or os.path.isabs(norm):
        return None
    real_world = os.path.realpath(world_dir)
    real_target = os.path.realpath(os.path.join(real_world, *parts))
    try:
        common = os.path.commonpath([real_world, real_target])
    except ValueError:
        return None
    if os.path.normcase(common) != os.path.normcase(real_world):
        return None
    if os.path.normcase(real_target) == os.path.normcase(real_world):
        return None
    if os.path.isdir(real_target):
        return None
    return real_target


# Outside /world/ the viewer is one page. Serving only these two request paths, matched before any decoding,
# keeps the guards of safe_world_path from being needed twice: a device name (/NUL), an alternate data stream
# (/index.html::$DATA) or a trailing dot or space (/index.html.) never reaches the file system.
_VIEWER_PAGES = frozenset({"/", "/index.html"})


class Handler(http.server.SimpleHTTPRequestHandler):
    world = None
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".glb": "model/gltf-binary",
    }

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # The viewer can cancel a chunk download while its response is being sent.
            pass

    def send_head(self):
        # headers.get() returns only the first Host field, so a second one would go unchecked. RFC 7230 section 5.4:
        # a request with more than one Host field gets 400.
        hosts = self.headers.get_all("Host", [])
        if len(hosts) > 1:
            self.send_error(400, "Duplicate Host header")
            return None
        if not host_allowed(hosts[0] if hosts else None):
            self.send_error(403, "Forbidden: open the viewer at http://127.0.0.1 or http://localhost")
            return None
        clean = self.path.split("#", 1)[0].split("?", 1)[0]
        if self.world and clean.startswith("/world/"):
            if safe_world_path(self.world, self.path) is None:
                self.send_error(403, "Forbidden")
                return None
        elif clean not in _VIEWER_PAGES:
            self.send_error(404, "File not found")
            return None
        return super().send_head()

    def translate_path(self, path):
        clean = path.split("#", 1)[0].split("?", 1)[0]
        if self.world and clean.startswith("/world/"):
            resolved = safe_world_path(self.world, path)
            return resolved if resolved is not None else ""
        if clean not in _VIEWER_PAGES:
            return ""
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def port_number(text: str) -> int:
    """--port: 0-65535. Anything else used to reach socketserver and end in OverflowError."""
    try:
        value = int(text, 10)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError("must be 0-65535")
    return value


class ViewerServer(http.server.ThreadingHTTPServer):
    """The viewer's server: a port another viewer is serving is an error on every OS.

    `HTTPServer` sets SO_REUSEADDR so a restart can bind while the old connections are still in TIME_WAIT. On
    POSIX that is all it does: a second listener on the same address still fails. On Windows SO_REUSEADDR lets
    a second socket bind a port that is being listened on; that viewer printed its URL and then got no requests,
    while the browser kept showing the first one's world. Windows needs no option for a restart either, so
    the option is left off there.
    """
    allow_reuse_address = os.name != "nt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", help="exported world directory (omit to show test terrain)")
    ap.add_argument("--port", type=port_number, default=8765,
                    help="0 asks the OS for a free port; the bound port is printed")
    a = ap.parse_args()
    if a.world is not None and not os.path.isdir(a.world):
        ap.error(f"--world is not a directory: {a.world}")
    Handler.world = a.world
    try:
        srv = ViewerServer(("127.0.0.1", a.port), functools.partial(Handler, directory=HERE))
    except OSError as e:
        # The usual cause is a viewer that is still running on the port; this ended in a traceback.
        ap.error(f"can't listen on port {a.port}: {e.strerror or e}. Is a viewer already running? "
                 "--port 0 asks the OS for a free port")
    try:
        # Inside the try: a Ctrl+C that arrives once this line is out (the test sends SIGINT as soon as it reads it)
        # must stop the viewer too, not escape between the print and serve_forever() with a traceback.
        print(f"viewer: http://127.0.0.1:{srv.server_address[1]}/  world: {Handler.world or '(none: test terrain)'}",
              flush=True)
        srv.serve_forever()
    except KeyboardInterrupt:      # Ctrl+C is how the viewer is stopped
        print("viewer: stopped", flush=True)
    finally:
        srv.server_close()


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    main()
