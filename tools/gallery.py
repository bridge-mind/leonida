"""Standalone local 3D Web Studio & Texture Gallery server for Leonida.

Launches a lightweight local HTTP server and opens a modern WebGL/glTF
3D asset inspector in the browser with Vice City / Leonida aesthetic.
The texture catalog lists DDS, PNG, JPEG (.jpg or .jpeg) and WebP files.

Usage:
    python -I tools/gallery.py [--port 8080] [--models-dir <path>] [--textures-dir <path>]
"""
from __future__ import annotations

import argparse
import http.server
import json
import mimetypes
import os
import socketserver
import sys
import urllib.parse
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dds_png
except ImportError:
    dds_png = None


TEXTURE_FORMATS = {
    ".dds": "DirectDraw Surface (DDS)",
    ".png": "PNG Image",
    ".jpg": "JPEG Image",
    ".jpeg": "JPEG Image",
    ".webp": "WebP Image",
}


def categorize_model(name: str, folder: str = "") -> str:
    combined = f"{folder}/{name}".lower()
    if any(k in combined for k in ("veh", "car", "truck", "bike", "wheel", "heli", "boat", "chassis", "motor", "engine")):
        return "Транспорт (Vehicles)"
    elif any(k in combined for k in ("weapon", "gun", "pistol", "rifle", "knife", "ammo", "shotgun", "smg")):
        return "Оружие (Weapons)"
    elif any(k in combined for k in ("light", "lamp", "sign", "box", "chair", "table", "prop", "debris", "anm", "amb", "core")):
        return "Объекты и Пропы (Props)"
    elif any(k in combined for k in ("tree", "grass", "plant", "bush", "flower", "water", "palm")):
        return "Растительность (Nature)"
    elif any(k in combined for k in ("build", "structure", "house", "bridge", "farlod", "station", "shop", "wall", "interior", "level", "road", "terrain")):
        return "Карта и Здания (World)"
    elif any(k in combined for k in ("ped", "female", "male", "head", "body", "lucia", "jason", "jacket", "face", "cloth", "skin", "_hair", "hair_")):
        return "Персонажи (Characters)"
    return "Прочее (Miscellaneous)"


_WINDOWS_RESERVED_DEVICES = frozenset({
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    # Windows also treats superscript 1/2/3 as device numbers; keep this fallback
    # complete for Python 3.12, which has no os.path.isreserved.
    *(f"COM{i}" for i in "123456789\u00b9\u00b2\u00b3"),
    *(f"LPT{i}" for i in "123456789\u00b9\u00b2\u00b3"),
})


def _is_reserved_segment(seg: str) -> bool:
    stem = seg.split(".", 1)[0].rstrip(" ").upper()
    if stem in _WINDOWS_RESERVED_DEVICES:
        return True
    isreserved = getattr(os.path, "isreserved", None)
    if callable(isreserved) and isreserved(seg):
        return True
    return False


# Binding to 127.0.0.1 keeps other machines out, but not other web pages: a page on any site can point its own
# domain at 127.0.0.1 (DNS rebinding) and read local assets through this server as same-origin.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})


def host_allowed(host: str | None) -> bool:
    """True when a request's Host header names the loopback address. Browsers always send Host; a request without
    one comes from a local tool, not a web page, and is allowed."""
    if host is None:
        return True
    host = host.strip(" \t").lower()
    if host.startswith("["):                        # [::1] or [::1]:8080
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
    if port is not None and not (0 < len(port) <= 5 and port.isascii() and port.isdigit() and int(port) <= 65535):
        return False
    return name in _LOOPBACK_HOSTS


def safe_subpath(base_dir: str, rel: str) -> str | None:
    """Resolve a relative URL path safely inside base_dir, or return None."""
    if not base_dir or not isinstance(rel, str) or not rel or "\x00" in rel:
        return None
    if rel.startswith(("/", "\\")) or rel.endswith(("/", "\\")):
        return None
    norm = rel.replace("\\", "/")
    if os.path.isabs(rel) or os.path.isabs(norm):
        return None
    parts = norm.split("/")
    for seg in parts:
        if not seg or seg in (".", "..") or ":" in seg or _is_reserved_segment(seg):
            return None
    real_base = os.path.realpath(base_dir)
    real_target = os.path.realpath(os.path.join(real_base, *parts))
    try:
        common = os.path.commonpath([real_base, real_target])
    except ValueError:
        return None
    if os.path.normcase(common) != os.path.normcase(real_base):
        return None
    if os.path.normcase(real_target) == os.path.normcase(real_base):
        return None
    if os.path.isdir(real_target):
        return None
    return real_target


class StudioHandler(http.server.BaseHTTPRequestHandler):
    models_dir: str = ""
    textures_dir: str = ""
    viewer_dir: str = ""

    def log_message(self, format: str, *args: object) -> None:
        pass  # Quiet logging

    def _check_host(self) -> bool:
        hosts = self.headers.get_all("Host", [])
        if len(hosts) > 1:
            self.send_error(400, "Duplicate Host header")
            return False
        if not host_allowed(hosts[0] if hosts else None):
            self.send_error(403, "Forbidden: open the gallery at http://127.0.0.1 or http://localhost")
            return False
        return True

    def do_GET(self) -> None:
        if not self._check_host():
            return

        clean = self.path.split("#", 1)[0].split("?", 1)[0]

        if clean in ("/", "/index.html", "/studio.html"):
            studio_path = os.path.join(self.viewer_dir, "studio.html")
            index_path = os.path.join(self.viewer_dir, "index.html")
            target = studio_path if os.path.isfile(studio_path) else index_path
            if os.path.isfile(target):
                self._send_file(target, "text/html; charset=utf-8")
            else:
                self.send_error(404, "Viewer page not found")
            return

        if clean == "/api/models":
            self._serve_models_json()
            return

        if clean == "/api/textures":
            self._serve_textures_json()
            return

        if clean.startswith("/api/dds/"):
            rel = urllib.parse.unquote(clean[len("/api/dds/"):])
            self._serve_dds(rel, self.path.partition("?")[2])
            return

        if clean.startswith("/vendor/"):
            rel = urllib.parse.unquote(clean[len("/vendor/"):])
            vpath = safe_subpath(os.path.join(self.viewer_dir, "vendor"), rel)
            if vpath is None:
                self.send_error(403, "Forbidden")
                return
            if os.path.isfile(vpath):
                mime = mimetypes.guess_type(vpath)[0] or "application/javascript"
                self._send_file(vpath, mime)
            else:
                self.send_error(404, "Vendor file not found")
            return

        if clean.startswith("/models/"):
            rel = urllib.parse.unquote(clean[len("/models/"):])
            mpath = safe_subpath(self.models_dir, rel)
            if mpath is None:
                self.send_error(403, "Forbidden")
                return
            if os.path.isfile(mpath):
                self._send_file(mpath, "model/gltf-binary")
            else:
                self.send_error(404, "Model not found")
            return

        if clean.startswith("/textures/"):
            rel = urllib.parse.unquote(clean[len("/textures/"):])
            tpath = safe_subpath(self.textures_dir, rel)
            if tpath is None:
                self.send_error(403, "Forbidden")
                return

            if not os.path.isfile(tpath) and rel.lower().endswith(".png"):
                # Try matching .dds file if png requested
                stem = os.path.splitext(rel)[0]
                dds_cand = safe_subpath(self.textures_dir, stem + ".dds")
                if dds_cand is None:
                    self.send_error(403, "Forbidden")
                    return
                if os.path.isfile(dds_cand) and dds_png:
                    try:
                        png_bytes = dds_png.dds_file_to_png_bytes(dds_cand, max_size=1024)
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Content-Length", str(len(png_bytes)))
                        self.send_header("Cache-Control", "max-age=86400")
                        self.end_headers()
                        self.wfile.write(png_bytes)
                        return
                    except Exception:
                        pass

            if os.path.isfile(tpath):
                mime = mimetypes.guess_type(tpath)[0] or "application/octet-stream"
                self._send_file(tpath, mime)
            else:
                self.send_error(404, "Texture not found")
            return

        # Serve static assets from viewer directory (e.g. style.css, app.js)
        if not clean.startswith("/"):
            self.send_error(403, "Forbidden")
            return
        rel_viewer = urllib.parse.unquote(clean[1:])
        vfile = safe_subpath(self.viewer_dir, rel_viewer)
        if vfile is None:
            self.send_error(403, "Forbidden")
            return
        if os.path.isfile(vfile):
            mime = mimetypes.guess_type(vfile)[0] or "application/octet-stream"
            if mime.startswith("text/"):
                mime += "; charset=utf-8"
            self._send_file(vfile, mime)
            return

        self.send_error(404, "Not Found")

    def _serve_dds(self, rel: str, query: str) -> None:
        path = safe_subpath(self.textures_dir, rel)
        if path is None:
            self.send_error(403, "Forbidden")
            return
        if not rel.lower().endswith(".dds") or not os.path.isfile(path):
            self.send_error(404, "DDS texture not found")
            return
        if dds_png is None:
            self.send_error(503, "DDS decoder unavailable")
            return
        try:
            if query == "info":
                with open(path, "rb") as source:
                    info = dds_png.parse_dds_header(source.read(148))
                payload = json.dumps(info).encode("utf-8")
                mime = "application/json"
            else:
                pairs = urllib.parse.parse_qsl(query, keep_blank_values=True, strict_parsing=True,
                                                 max_num_fields=5)
                selection = {}
                for key, value in pairs:
                    if (key not in ("mip", "layer", "z", "x", "y") or key in selection or
                            not value.isascii() or not value.isdecimal()):
                        raise ValueError("Expected a single unsigned index per selector")
                    selection[key] = int(value)
                if "x" in selection or "y" in selection:
                    if "x" not in selection or "y" not in selection:
                        raise ValueError("Expected both texel coordinates")
                    payload = json.dumps(dds_png.read_half_texel(path, **selection)).encode("utf-8")
                    mime = "application/json"
                else:
                    payload = dds_png.dds_file_to_png_bytes(path, max_size=1024, **selection)
                    mime = "image/png"
        except ValueError:
            self.send_error(400, "Invalid DDS texture or subresource selection")
            return
        except OSError:
            self.send_error(500, "Could not read DDS texture")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def _send_file(self, file_path: str, content_type: str) -> None:
        # Open the file and take its size before the status line goes out. Once 200 and Content-Length are sent,
        # an error can no longer become a 500: send_error would write a second response into this one's body.
        try:
            f = open(file_path, "rb")
        except OSError as exc:
            self.send_error(500, f"Could not open the file: {exc.strerror or exc}")
            return
        with f:
            try:
                size = os.fstat(f.fileno()).st_size
            except OSError as exc:
                self.send_error(500, f"Could not read the file: {exc.strerror or exc}")
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            remaining = size
            try:
                while remaining > 0 and (chunk := f.read(min(65536, remaining))):
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except OSError:
                # A read error, or the client went away: the headers are out, so end the connection with a short
                # body instead of appending an error page to it.
                remaining = -1
            if remaining:
                self.close_connection = True

    def _serve_models_json(self) -> None:
        items = []
        if os.path.isdir(self.models_dir):
            for root, dirs, files in os.walk(self.models_dir):
                dirs.sort()
                for fn in sorted(files):
                    if fn.lower().endswith((".glb", ".gltf")):
                        full_path = os.path.join(root, fn)
                        rel_path = os.path.relpath(full_path, self.models_dir).replace("\\", "/")
                        rel_dir = os.path.relpath(root, self.models_dir).replace("\\", "/")
                        folder_name = "Корневая папка" if rel_dir == "." else rel_dir
                        try:
                            fsize = os.path.getsize(full_path)
                        except OSError:
                            fsize = 0
                        items.append({
                            "name": fn,
                            "path": rel_path,
                            "folder": folder_name,
                            "category": categorize_model(fn, folder_name),
                            "size": fsize,
                            "url": f"/models/{urllib.parse.quote(rel_path)}",
                        })

        # Group by category and folders for the UI
        categories = {}
        folders = {}
        for item in items:
            cat = item["category"]
            fld = item["folder"]
            categories.setdefault(cat, []).append(item)
            folders.setdefault(fld, []).append(item)

        payload = json.dumps({
            "count": len(items),
            "models": items,
            "categories": {k: len(v) for k, v in categories.items()},
            "folders": {k: len(v) for k, v in folders.items()},
        }, ensure_ascii=False).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_textures_json(self) -> None:
        items = []
        if os.path.isdir(self.textures_dir):
            for root, dirs, files in os.walk(self.textures_dir):
                dirs.sort()
                for fn in sorted(files):
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in TEXTURE_FORMATS:
                        full_path = os.path.join(root, fn)
                        rel_path = os.path.relpath(full_path, self.textures_dir).replace("\\", "/")
                        try:
                            fsize = os.path.getsize(full_path)
                        except OSError:
                            fsize = 0
                        preview_rel = os.path.splitext(rel_path)[0] + ".png" if ext == ".dds" else rel_path
                        item = {
                            "name": fn,
                            "path": rel_path,
                            "size": fsize,
                            "format": TEXTURE_FORMATS[ext],
                            "url": f"/textures/{urllib.parse.quote(rel_path)}",
                            "preview": f"/textures/{urllib.parse.quote(preview_rel)}",
                        }
                        if ext == ".dds":
                            item["ddsPreview"] = f"/api/dds/{urllib.parse.quote(rel_path)}"
                        items.append(item)

        items.sort(key=lambda x: x["name"])
        payload = json.dumps({
            "count": len(items),
            "textures": items,
            "textures_dir": self.textures_dir,
        }, ensure_ascii=False).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class GalleryListenError(OSError):
    """The gallery could not listen on a local port."""


def start_server(
    port: int = 8080,
    models_dir: str | None = None,
    textures_dir: str | None = None,
    open_browser: bool = True,
    *,
    retry_next_port: bool = True,
) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    viewer_dir = os.path.join(root, "viewer")

    # Auto-locate candidate directories
    if not models_dir or not os.path.isdir(models_dir):
        candidates = [
            os.path.join(root, "models"),
            os.path.join(root, "..", "batch_converted_glb"),
            os.path.join(root, "..", "extracted_models"),
        ]
        best_cand = None
        max_count = 0
        for cand in candidates:
            if os.path.isdir(cand):
                c = sum(1 for _, _, files in os.walk(cand) for f in files if f.lower().endswith(('.glb', '.gltf')))
                if c > max_count:
                    max_count = c
                    best_cand = cand
        if best_cand:
            models_dir = best_cand
        if not models_dir or not os.path.isdir(models_dir):
            models_dir = os.path.join(root, "models")
            os.makedirs(models_dir, exist_ok=True)

    if not textures_dir or not os.path.isdir(textures_dir):
        candidates = [
            os.path.join(root, "textures"),
            os.path.join(root, "..", "clean_vehicles", "ttdb_vehicles"),
            os.path.join(root, "..", "viewable_pngs"),
            os.path.join(root, "..", "real_vehicle_pngs"),
        ]
        best_cand = None
        max_count = 0
        for cand in candidates:
            if os.path.isdir(cand):
                c = sum(1 for f in os.listdir(cand) if os.path.splitext(f)[1].lower() in TEXTURE_FORMATS)
                if c > max_count:
                    max_count = c
                    best_cand = cand
        if best_cand:
            textures_dir = best_cand
        if not textures_dir or not os.path.isdir(textures_dir):
            textures_dir = os.path.join(root, "textures")
            os.makedirs(textures_dir, exist_ok=True)

    StudioHandler.models_dir = os.path.abspath(models_dir)
    StudioHandler.textures_dir = os.path.abspath(textures_dir)
    StudioHandler.viewer_dir = viewer_dir

    addr = ("127.0.0.1", port)
    class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True

    try:
        try:
            httpd = ThreadingServer(addr, StudioHandler)
        except OSError:
            if not retry_next_port or not 0 < port < 65535:
                raise
            addr = ("127.0.0.1", port + 1)
            httpd = ThreadingServer(addr, StudioHandler)
    except OSError as error:
        raise GalleryListenError(f"can't listen on port {addr[1]}: {error}") from error

    url = f"http://localhost:{addr[1]}"
    print(f"======================================================")
    print(f"       LEONIDA 3D ASSET STUDIO (VICE CITY EDITION)    ")
    print(f"======================================================")
    print(f"Web Studio URL:     {url}")
    print(f"Models Directory:   {StudioHandler.models_dir}")
    print(f"Textures Directory: {StudioHandler.textures_dir}")
    print(f"Viewer Directory:   {StudioHandler.viewer_dir}")
    print(f"Press Ctrl+C to stop.")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Leonida 3D Asset Studio local web gallery.")
    parser.add_argument("--port", type=int,
                        help="HTTP port from 0 to 65535 (default 8080, or 8081 if unavailable)")
    parser.add_argument("--models-dir", type=str, default=None, help="Directory containing .glb models")
    parser.add_argument("--textures-dir", type=str, default=None, help="Directory containing textures")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")

    args = parser.parse_args(argv)
    port = 8080 if args.port is None else args.port
    if not 0 <= port <= 65535:
        parser.error("--port must be between 0 and 65535")
    try:
        start_server(
            port=port,
            models_dir=args.models_dir,
            textures_dir=args.textures_dir,
            open_browser=not args.no_browser,
            retry_next_port=args.port is None,
        )
    except GalleryListenError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    sys.exit(main())
