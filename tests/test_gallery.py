"""Unit tests for tools/gallery.py standalone 3D Studio & Asset Gallery server."""
import functools
import json
import io
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
import urllib.parse
import urllib.request
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import gallery


def _make_1x1_rgba_dds() -> bytes:
    """Build a minimal 1x1 uncompressed RGBA DDS image for synthetic .dds -> .png preview tests."""
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    struct.pack_into("<I", hdr, 4, 124)
    struct.pack_into("<III", hdr, 8, 0x1007, 1, 1)
    struct.pack_into("<I", hdr, 76, 32)
    struct.pack_into("<I", hdr, 80, 0x41)  # DDPF_RGB | DDPF_ALPHAPIXELS
    struct.pack_into("<IIIII", hdr, 88, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    return bytes(hdr) + b"\x11\x22\x33\xff"


def _get_request_bytes(target: str, host: str | None, extra_headers: tuple[str, ...]) -> bytes:
    lines = [f"GET {target} HTTP/1.1"]
    if host is not None:
        lines.append(f"Host: {host}")
    lines.extend(extra_headers)
    lines.extend(["Connection: close", "", ""])
    return "\r\n".join(lines).encode("latin-1")


def _split_response(raw: bytes):
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    return status, head.decode("latin-1"), body


def _raw_get(port: int, target: str, host: str | None = "127.0.0.1", extra_headers: tuple[str, ...] = ()):
    """Send a raw HTTP/1.1 GET without client-side path normalization or header merging."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(_get_request_bytes(target, host, extra_headers))
        chunks = []
        while data := sock.recv(65536):
            chunks.append(data)
    return _split_response(b"".join(chunks))


class _MemoryConnection:
    """Stands in for the accepted socket, so the handler reads exactly the bytes a test wrote: no loopback TCP, and so
    nothing between the two ends that could rewrite the request (see _loopback_keeps_a_repeated_host)."""

    def __init__(self, request: bytes):
        self._request = io.BytesIO(request)
        self.response = bytearray()

    def makefile(self, mode, bufsize=-1):
        return self._request

    def sendall(self, data):
        self.response += data


def _memory_get(target: str, host: str | None = "127.0.0.1", extra_headers: tuple[str, ...] = ()):
    """_raw_get without a socket: gallery.StudioHandler answers the same request bytes in this thread."""
    connection = _MemoryConnection(_get_request_bytes(target, host, extra_headers))
    gallery.StudioHandler(connection, ("127.0.0.1", 0), None)
    return _split_response(bytes(connection.response))


@functools.lru_cache(maxsize=1)
def _loopback_keeps_a_repeated_host() -> bool:
    """False when a request sent over loopback TCP arrives without its second Host field.

    Security software that filters HTTP does this on some machines, on 127.0.0.1 too (measured with ESET Security on
    Windows 11: `Host: a`, `Host: b` arrives as `Host: a`). The server then never sees the request the test wrote, so a
    test of the duplicate-Host rule over a socket measures the filter, not gallery.py. Same probe as #4985 adds to
    tests/test_viewer_serve.py."""
    request = b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n"
    received = bytearray()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(10)

        def accept():
            try:
                peer, _ = listener.accept()
            except OSError:
                return
            with peer:
                peer.settimeout(10)
                try:
                    while not received.endswith(b"\r\n\r\n"):
                        data = peer.recv(65536)
                        if not data:
                            break
                        received.extend(data)
                except OSError:
                    pass

        thread = threading.Thread(target=accept, daemon=True)
        thread.start()
        try:
            with socket.create_connection(listener.getsockname(), timeout=10) as sock:
                sock.sendall(request)
                thread.join(10)
        except OSError:
            return True          # no answer about a filter: let the socket test run and report what it sees
    return bytes(received) == request or not received


class TestGalleryCategorization(unittest.TestCase):
    def test_categories(self) -> None:
        self.assertEqual(gallery.categorize_model("ped_lucia_body.glb"), "Персонажи (Characters)")
        self.assertEqual(gallery.categorize_model("veh_infernus.glb"), "Транспорт (Vehicles)")
        self.assertEqual(gallery.categorize_model("w_rifle_m4.glb"), "Оружие (Weapons)")
        self.assertEqual(gallery.categorize_model("build_wall_01.glb"), "Карта и Здания (World)")
        self.assertEqual(gallery.categorize_model("prop_chair_office.glb"), "Объекты и Пропы (Props)")
        self.assertEqual(gallery.categorize_model("tree_palm_02.glb"), "Растительность (Nature)")
        self.assertEqual(gallery.categorize_model("misc_something.glb"), "Прочее (Miscellaneous)")

    def test_folder_inference(self) -> None:
        self.assertEqual(gallery.categorize_model("asset_01.glb", "clean_vehicles"), "Транспорт (Vehicles)")
        self.assertEqual(gallery.categorize_model("model.glb", "unpacked_peds"), "Персонажи (Characters)")


class TestHostAllowed(unittest.TestCase):
    def test_loopback_and_missing_hosts_are_allowed(self) -> None:
        for host in (None, "127.0.0.1", "127.0.0.1:8080", "localhost", "LOCALHOST:8080",
                     "localhost.", "localhost.:8080", "[::1]", "[::1]:8080", " localhost "):
            with self.subTest(host=host):
                self.assertTrue(gallery.host_allowed(host))

    def test_foreign_and_malformed_hosts_are_refused(self) -> None:
        for host in ("", "evil.example", "evil.example:8080", "127.0.0.1.evil.example",
                     "localhost.evil.example:8080", "0.0.0.0:8080", "::1",
                     "localhost:", "localhost:abc", "localhost:65536", "localhost:" + "9" * 5000,
                     "[::1]junk", "[::1].", "[::1]:abc", "[::1]:8080:1", "127.0.0.1:80:80"):
            with self.subTest(host=host[:40]):
                self.assertFalse(gallery.host_allowed(host))


class TestSafeSubpath(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = os.path.join(self.tmp.name, "assets")
        self.sibling = os.path.join(self.tmp.name, "assets_secret")
        os.makedirs(os.path.join(self.base, "sub"), exist_ok=True)
        os.makedirs(self.sibling, exist_ok=True)
        with open(os.path.join(self.base, "ok.glb"), "wb") as f:
            f.write(b"ok")
        with open(os.path.join(self.base, "sub", "nested.glb"), "wb") as f:
            f.write(b"nested")
        with open(os.path.join(self.sibling, "loot.glb"), "wb") as f:
            f.write(b"loot")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_paths_resolve_inside_base(self) -> None:
        for rel in ("ok.glb", "sub/nested.glb", "COM10.glb", "my.CON"):
            with self.subTest(rel=rel):
                resolved = gallery.safe_subpath(self.base, rel)
                self.assertIsNotNone(resolved)
                self.assertEqual(
                    os.path.normcase(os.path.commonpath([os.path.realpath(self.base), resolved])),
                    os.path.normcase(os.path.realpath(self.base)),
                )

    def test_traversal_and_invalid_segments_are_refused(self) -> None:
        for rel in ("", "/", "\\", ".", "..", "sub", "sub/", "/ok.glb", "ok.glb/",
                    "../assets_secret/loot.glb", "..\\assets_secret\\loot.glb",
                    "sub/../../assets_secret/loot.glb", "sub//nested.glb", "./ok.glb",
                    "ok\x00.glb", "C:ok.glb", "ok.glb:stream"):
            with self.subTest(rel=rel):
                self.assertIsNone(gallery.safe_subpath(self.base, rel))

    def test_windows_device_names_refused_before_realpath(self) -> None:
        names = ("CON", "con.glb", "NUL", "NUL .glb", "COM1", "COM\u00b9.glb", "LPT\u00b2.png", "CONIN$", "conout$.bin")
        with mock.patch.object(gallery.os.path, "isreserved", None, create=True):
            with mock.patch.object(gallery.os.path, "realpath",
                                   side_effect=AssertionError("device path reached realpath")) as rp:
                for name in names:
                    with self.subTest(name=name):
                        self.assertIsNone(gallery.safe_subpath(self.base, name))
                        self.assertIsNone(gallery.safe_subpath(self.base, f"sub/{name}"))
                rp.assert_not_called()

    def test_symlink_escape_is_refused(self) -> None:
        link = os.path.join(self.base, "escape.glb")
        try:
            os.symlink(os.path.join(self.sibling, "loot.glb"), link)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"cannot create symlink here: {exc}")
        self.assertIsNone(gallery.safe_subpath(self.base, "escape.glb"))

    def test_realpath_escape_is_refused(self) -> None:
        outside = os.path.realpath(os.path.join(self.sibling, "loot.glb"))
        real_base = os.path.realpath(self.base)
        with mock.patch.object(gallery.os.path, "realpath",
                               side_effect=lambda p: real_base if p == self.base else outside):
            self.assertIsNone(gallery.safe_subpath(self.base, "escape.glb"))


class TestGalleryServer(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.models_dir = os.path.join(self.tmp.name, "models")
        self.models_secret = os.path.join(self.tmp.name, "models_secret")
        self.textures_dir = os.path.join(self.tmp.name, "textures")
        self.textures_secret = os.path.join(self.tmp.name, "textures_secret")
        self.viewer_dir = os.path.join(self.tmp.name, "viewer")
        self.viewer_secret = os.path.join(self.tmp.name, "viewer_secret")
        self.vendor_dir = os.path.join(self.viewer_dir, "vendor")
        for d in (self.models_dir, self.models_secret, self.textures_dir,
                  self.textures_secret, self.viewer_dir, self.viewer_secret, self.vendor_dir):
            os.makedirs(d, exist_ok=True)

        # Create dummy assets
        with open(os.path.join(self.models_dir, "test_car.glb"), "wb") as f:
            f.write(b"glTF\x02\x00\x00\x00\x0c\x00\x00\x00")

        with open(os.path.join(self.textures_dir, "test_tex.png"), "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

        with open(os.path.join(self.textures_dir, "inside.dds"), "wb") as f:
            f.write(_make_1x1_rgba_dds())

        with open(os.path.join(self.viewer_dir, "studio.html"), "w", encoding="utf-8", newline="\n") as f:
            f.write("<!DOCTYPE html><html><body><h1>Studio</h1></body></html>")

        with open(os.path.join(self.viewer_dir, "app.js"), "w", encoding="utf-8", newline="\n") as f:
            f.write("console.log('ok');\n")

        with open(os.path.join(self.vendor_dir, "three.min.js"), "w", encoding="utf-8", newline="\n") as f:
            f.write("// three\n")

        # Files outside the served roots that must never be served
        with open(os.path.join(self.tmp.name, "secret.txt"), "wb") as f:
            f.write(b"TOP_SECRET")
        with open(os.path.join(self.tmp.name, "outside.dds"), "wb") as f:
            f.write(_make_1x1_rgba_dds())
        with open(os.path.join(self.models_secret, "loot.glb"), "wb") as f:
            f.write(b"LOOT_GLB")
        with open(os.path.join(self.textures_secret, "loot.png"), "wb") as f:
            f.write(b"LOOT_PNG")
        with open(os.path.join(self.viewer_secret, "loot.txt"), "wb") as f:
            f.write(b"LOOT_VIEWER")

        # Configure handler
        gallery.StudioHandler.models_dir = self.models_dir
        gallery.StudioHandler.textures_dir = self.textures_dir
        gallery.StudioHandler.viewer_dir = self.viewer_dir

        # Start server on an open port
        addr = ("127.0.0.1", 0)
        self.httpd = gallery.socketserver.TCPServer(addr, gallery.StudioHandler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        time.sleep(0.1)

    def tearDown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def test_texture_catalog_labels_jpeg_and_webp(self) -> None:
        files = {"sky.JPG": "JPEG Image", "road.jpeg": "JPEG Image", "tree.webp": "WebP Image"}
        for name in files:
            with open(os.path.join(self.textures_dir, name), "wb") as f:
                f.write(b"synthetic")
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/textures", timeout=3) as r:
            textures = {row["name"]: row for row in json.load(r)["textures"]}
        for name, expected in files.items():
            with self.subTest(name=name):
                self.assertIn(name, textures)
                self.assertEqual(textures[name]["format"], expected)
                self.assertEqual(textures[name]["preview"], textures[name]["url"])

    def test_dds_selection_by_catalog_url(self) -> None:
        # Two RGBA8 layers with a 2x2 and 1x1 mip each.
        hdr = bytearray(_make_1x1_rgba_dds()[:128])
        struct.pack_into("<III", hdr, 8, 0x21007, 2, 2)
        struct.pack_into("<I", hdr, 28, 2)
        struct.pack_into("<I4s", hdr, 80, 4, b"DX10")
        data = hdr + struct.pack("<5I", 28, 3, 0, 2, 0)
        for value in (1, 3):
            data += bytes([value]) * 16 + bytes([value + 1]) * 4
        for name, contents in (("A #.DDS", data), ("A #.png", b"sidecar")):
            with open(os.path.join(self.textures_dir, name), "wb") as f:
                f.write(contents)

        def get(url):
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}" + url, timeout=3) as response:
                return response.read()

        row = next(item for item in json.loads(get("/api/textures"))["textures"]
                   if item["name"] == "A #.DDS")
        self.assertEqual(get(row["preview"]), b"sidecar")
        url = row["ddsPreview"]
        info = json.loads(get(url + "?info"))
        self.assertEqual([info[k] for k in ("width", "height", "mip_count", "array_size")], [2] * 4)
        for layer in range(2):
            for mip in range(2):
                size = 2 >> mip
                rgba = bytes([1 + layer * 2 + mip]) * (size * size * 4)
                expected = gallery.dds_png.encode_png(size, size, rgba)
                self.assertEqual(get(f"{url}?mip={mip}&layer={layer}"), expected)
                if mip == layer == 0:
                    self.assertEqual(get(url), expected)
        for query in ("layer=2", "mip=2", "mip=-1", "layer=1.5", "mip=", "mip=0&mip=1", "other=0"):
            with self.subTest(query=query), self.assertRaises(urllib.error.HTTPError) as caught:
                get(url + "?" + query)
            self.assertEqual(caught.exception.code, 400)

    def test_volume_z_selection(self) -> None:
        # Tight R16_FLOAT rows. Each 3x2 plane has a different pixel pattern.
        hdr = bytearray(_make_1x1_rgba_dds()[:128])
        struct.pack_into("<IIIII", hdr, 8, 0x80100F, 2, 3, 6, 3)
        struct.pack_into("<I", hdr, 28, 1)
        struct.pack_into("<I4s", hdr, 80, 4, b"DX10")
        struct.pack_into("<II", hdr, 108, 0x1008, 0x200000)
        data = hdr + struct.pack("<5I", 54, 4, 0, 1, 0)
        patterns = [(0, 1, 2, 3, 4, 0), (4, 3, 2, 1, 0, 4), (2, 0, 1, 4, 3, 1)]
        red = (0, 64, 128, 191, 255)
        for pattern in patterns:
            data.extend(struct.pack("<6e", *(v / 4 for v in pattern)))
        with open(os.path.join(self.textures_dir, "volume.dds"), "wb") as f:
            f.write(data)
        url = "/api/dds/volume.dds"
        for z, pattern in enumerate(patterns):
            rgba = bytes(c for v in pattern for c in (red[v], 0, 0, 255))
            expected = gallery.dds_png.encode_png(3, 2, rgba)
            status, _, png = _raw_get(self.port, f"{url}?mip=0&z={z}")
            self.assertEqual((status, png), (200, expected))
            status, _, sample = _raw_get(self.port, f"{url}?x=1&y=1&z={z}")
            self.assertEqual((status, json.loads(sample)), (200, {"R": str(pattern[4] / 4)}))
            if z == 0:
                self.assertEqual(_raw_get(self.port, url)[2], expected)
        for query in ("z=3", "layer=1&z=0", "z=-1", "z=1&z=2", "mip=1&z=0",
                      "x=0", "y=1", "x=3&y=0", "x=0&y=2"):
            with self.subTest(query=query):
                self.assertEqual(_raw_get(self.port, url + "?" + query)[0], 400)

    def test_rgba16_source_texel(self) -> None:
        hdr = bytearray(_make_1x1_rgba_dds()[:128])
        struct.pack_into("<IIII", hdr, 8, 0x2100F, 2, 4, 40)
        struct.pack_into("<I", hdr, 28, 2)
        struct.pack_into("<I4s", hdr, 80, 4, b"DX10")
        data = hdr + struct.pack("<5I", 10, 3, 0, 2, 0)
        # Each layer has two padded 40-byte rows then two tight mip-1 texels.
        data.extend(bytes(2 * (80 + 16)))
        struct.pack_into("<4e", data, 148 + 96 + 40 + 16, -3, 2.5, -0.0, float("inf"))
        struct.pack_into("<4H", data, len(data) - 8, 1, 0x7e03, 0xfc00, 0x3c00)
        path = os.path.join(self.textures_dir, "hdr.dds")
        with open(path, "wb") as f:
            f.write(data)
        url = "/api/dds/hdr.dds?"
        for query, expected in [
            ("layer=1&x=2&y=1", ["-3.0", "2.5", "-0.0", "inf"]),
            ("layer=1&mip=1&x=1&y=0", [str(2 ** -24), "nan", "-inf", "1.0"]),
        ]:
            status, _, sample = _raw_get(self.port, url + query)
            self.assertEqual((status, json.loads(sample)), (200, dict(zip("RGBA", expected))))
        with open(path, "rb+") as f:
            f.truncate(len(data) - 1)
        self.assertEqual(_raw_get(self.port, url + "layer=1&mip=1&x=1&y=0")[0], 400)
        self.assertEqual(_raw_get(self.port, "/api/dds/inside.dds?x=0&y=0")[0], 400)

    def test_endpoints(self) -> None:
        base = f"http://127.0.0.1:{self.port}"

        # 1. Studio HTML root
        with urllib.request.urlopen(f"{base}/", timeout=3) as r:
            self.assertEqual(r.status, 200)
            body = r.read().decode("utf-8")
            self.assertIn("<h1>Studio</h1>", body)

        # 2. Models API
        with urllib.request.urlopen(f"{base}/api/models", timeout=3) as r:
            self.assertEqual(r.status, 200)
            data = json.loads(r.read())
            self.assertEqual(data["count"], 1)
            self.assertEqual(data["models"][0]["name"], "test_car.glb")
            self.assertEqual(data["models"][0]["category"], "Транспорт (Vehicles)")

        # 3. Textures API
        with urllib.request.urlopen(f"{base}/api/textures", timeout=3) as r:
            self.assertEqual(r.status, 200)
            tdata = json.loads(r.read())
            self.assertEqual(tdata["count"], 2)
            self.assertEqual([t["name"] for t in tdata["textures"]], ["inside.dds", "test_tex.png"])

        # 4. Binary serving & DDS preview fallback inside textures_dir
        with urllib.request.urlopen(f"{base}/models/test_car.glb", timeout=3) as r:
            self.assertEqual(r.status, 200)
            self.assertTrue(r.read().startswith(b"glTF"))

        with urllib.request.urlopen(f"{base}/textures/test_tex.png", timeout=3) as r:
            self.assertEqual(r.status, 200)
            self.assertTrue(r.read().startswith(b"\x89PNG"))

        with urllib.request.urlopen(f"{base}/textures/inside.png", timeout=3) as r:
            self.assertEqual(r.status, 200)
            self.assertTrue(r.read().startswith(b"\x89PNG"))

        # 5. Viewer static & vendor assets
        with urllib.request.urlopen(f"{base}/vendor/three.min.js", timeout=3) as r:
            self.assertEqual(r.status, 200)
            self.assertIn(b"// three", r.read())

        with urllib.request.urlopen(f"{base}/app.js", timeout=3) as r:
            self.assertEqual(r.status, 200)
            self.assertIn(b"console.log", r.read())

    def _raw_get(self, path: str) -> bytes:
        """The whole response, status line and headers included, as the socket delivered it."""
        with socket.create_connection(("127.0.0.1", self.port), timeout=3) as sock:
            sock.sendall(f"GET {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n".encode())
            data = b""
            while chunk := sock.recv(65536):
                data += chunk
        return data

    def test_a_file_that_cannot_be_opened_is_one_500_response(self) -> None:
        # _send_file used to send 200 and Content-Length before opening the file, then wrote a second status line
        # (send_error's 500) into the body of the first.
        def refuse(path, *args, **kwargs):
            raise PermissionError(13, "Permission denied", path)

        gallery.open = refuse
        self.addCleanup(delattr, gallery, "open")
        response = self._raw_get("/textures/test_tex.png")
        self.assertEqual(response.count(b"HTTP/1."), 1, response)
        self.assertTrue(response.startswith(b"HTTP/1.0 500 "), response)
        self.assertNotIn(b"LOOT", response)

    def test_a_read_error_after_the_headers_ends_the_response_short(self) -> None:
        # Once 200 is out the status can't change: the body stops short and the connection closes, with no error
        # page appended to it.
        real_open = open

        class FailingRead(io.BufferedReader):
            def read(self, *args):
                raise OSError(5, "Input/output error")

        def failing(path, *args, **kwargs):
            return FailingRead(real_open(path, "rb", buffering=0))

        gallery.open = failing
        self.addCleanup(delattr, gallery, "open")
        response = self._raw_get("/textures/test_tex.png")
        head, _, body = response.partition(b"\r\n\r\n")
        self.assertEqual(response.count(b"HTTP/1."), 1, response)
        self.assertTrue(head.startswith(b"HTTP/1.0 200 "), head)
        size = os.path.getsize(os.path.join(self.textures_dir, "test_tex.png"))
        self.assertIn(f"Content-Length: {size}".encode(), head)
        self.assertEqual(body, b"")

    def test_a_read_error_after_the_headers_closes_a_keep_alive_connection(self) -> None:
        # The gallery answers HTTP/1.0, which closes after every response anyway. Under HTTP/1.1 the connection
        # would stay open, so a short body must close it, or the client would wait for the missing bytes.
        real_open = open

        class FailingRead(io.BufferedReader):
            def read(self, *args):
                raise OSError(5, "Input/output error")

        gallery.open = lambda path, *a, **k: FailingRead(real_open(path, "rb", buffering=0))
        self.addCleanup(delattr, gallery, "open")
        self.addCleanup(setattr, gallery.StudioHandler, "protocol_version", gallery.StudioHandler.protocol_version)
        gallery.StudioHandler.protocol_version = "HTTP/1.1"
        with socket.create_connection(("127.0.0.1", self.port), timeout=3) as sock:
            sock.sendall(b"GET /textures/test_tex.png HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n")
            data = b""
            while chunk := sock.recv(65536):   # a socket.timeout here means the connection was left open
                data += chunk
        head, _, body = data.partition(b"\r\n\r\n")
        self.assertTrue(head.startswith(b"HTTP/1.1 200 "), head)
        self.assertEqual(data.count(b"HTTP/1."), 1, data)
        self.assertEqual(body, b"")

    def test_traversal_and_sibling_prefix_escapes_are_forbidden(self) -> None:
        host = f"127.0.0.1:{self.port}"
        cases = (
            ("/vendor/../../secret.txt", b"TOP_SECRET"),
            ("/vendor/%2e%2e/%2e%2e/secret.txt", b"TOP_SECRET"),
            ("/models/../models_secret/loot.glb", b"LOOT_GLB"),
            ("/models/..%5cmodels_secret%5cloot.glb", b"LOOT_GLB"),
            ("/textures/../textures_secret/loot.png", b"LOOT_PNG"),
            ("/textures/../outside.png", b"\x89PNG"),
            ("/textures/%2e%2e/outside.png", b"\x89PNG"),
            ("/../viewer_secret/loot.txt", b"LOOT_VIEWER"),
            ("/%2e%2e/viewer_secret/loot.txt", b"LOOT_VIEWER"),
            ("/%2fviewer_secret/loot.txt", b"LOOT_VIEWER"),
            ("/models//test_car.glb", b"glTF"),
        )
        for target, secret in cases:
            with self.subTest(target=target):
                status, _, body = _raw_get(self.port, target, host=host)
                self.assertEqual(status, 403)
                self.assertNotIn(secret, body)

    def test_dds_symlink_escape_is_forbidden(self) -> None:
        link = os.path.join(self.textures_dir, "escaped.dds")
        try:
            os.symlink(os.path.join(self.tmp.name, "outside.dds"), link)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"cannot create symlink here: {exc}")
        status, _, body = _raw_get(self.port, "/textures/escaped.png", host=f"127.0.0.1:{self.port}")
        self.assertEqual(status, 403)
        self.assertNotIn(b"\x89PNG", body)

    _GUARDED_TARGETS = ("/", "/api/models", "/api/textures", "/models/test_car.glb", "/textures/test_tex.png")

    def test_host_header_guards(self) -> None:
        for target in self._GUARDED_TARGETS:
            with self.subTest(case="foreign", target=target):
                status, _, body = _raw_get(self.port, target, host="evil.example:8080")
                self.assertEqual(status, 403)
                self.assertNotIn(b"test_car.glb", body)
                self.assertNotIn(b"glTF", body)

        status, _, body = _raw_get(self.port, "/models/test_car.glb", host=None)
        self.assertEqual(status, 200)
        self.assertTrue(body.startswith(b"glTF"))

    def _check_duplicate_host_gets_400(self, get) -> None:
        loopback = f"127.0.0.1:{self.port}"
        for target in self._GUARDED_TARGETS:
            with self.subTest(target=target):
                status, _, body = get(target, host=loopback, extra_headers=("Host: evil.example",))
                self.assertEqual(status, 400)
                self.assertNotIn(b"test_car.glb", body)
                self.assertNotIn(b"glTF", body)

    def test_duplicate_host_gets_400(self) -> None:
        # The handler reads the request from memory, so the rule is checked on every machine.
        self._check_duplicate_host_gets_400(_memory_get)

    def test_duplicate_host_gets_400_over_a_socket(self) -> None:
        if not _loopback_keeps_a_repeated_host():
            self.skipTest("something on this machine removes a repeated Host field from loopback HTTP requests "
                          "(security software that filters HTTP)")
        self._check_duplicate_host_gets_400(functools.partial(_raw_get, self.port))

    def test_a_request_from_memory_is_answered_like_one_over_a_socket(self) -> None:
        # _memory_get must reach the same guards and files as _raw_get, or test_duplicate_host_gets_400 checks nothing.
        loopback = f"127.0.0.1:{self.port}"
        for target, host in (("/models/test_car.glb", loopback), ("/textures/test_tex.png", loopback),
                             ("/api/models", loopback), ("/models/test_car.glb", "evil.example"),
                             ("/models/../secret.txt", loopback), ("/models/test_car.glb", None)):
            with self.subTest(target=target, host=host):
                over_socket = _raw_get(self.port, target, host=host)
                from_memory = _memory_get(target, host=host)
                self.assertEqual(from_memory[0], over_socket[0])
                self.assertEqual(from_memory[2], over_socket[2])

    def test_windows_device_names_get_403_without_opening(self) -> None:
        host = f"127.0.0.1:{self.port}"
        targets = ("/models/COM%C2%B9.glb", "/textures/NUL%20.png", "/vendor/CONIN%24", "/AUX.js")
        with mock.patch.object(gallery.os.path, "isreserved", None, create=True):
            with mock.patch("builtins.open", side_effect=AssertionError("device path reached open")) as opened:
                for target in targets:
                    with self.subTest(target=target):
                        status, _, _ = _raw_get(self.port, target, host=host)
                        self.assertEqual(status, 403)
                opened.assert_not_called()

    def test_missing_valid_paths_return_404(self) -> None:
        host = f"127.0.0.1:{self.port}"
        for target in ("/vendor/missing.js", "/models/missing.glb", "/textures/missing.png", "/missing.js"):
            with self.subTest(target=target):
                status, _, _ = _raw_get(self.port, target, host=host)
                self.assertEqual(status, 404)


class TestGalleryCommandLine(unittest.TestCase):
    def test_rejects_out_of_range_ports_before_starting_server(self) -> None:
        for port in (-1, 65536):
            with (
                self.subTest(port=port),
                mock.patch.object(gallery, "start_server") as start_server,
                mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
            ):
                with self.assertRaises(SystemExit) as raised:
                    gallery.main(["--port", str(port), "--no-browser"])

                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--port must be between 0 and 65535", stderr.getvalue())
                start_server.assert_not_called()

    def test_passes_default_and_boundary_ports_to_server(self) -> None:
        for args, port in (([], 8080), (["--port", "0"], 0), (["--port", "65535"], 65535)):
            with self.subTest(port=port), mock.patch.object(gallery, "start_server") as start_server:
                self.assertEqual(gallery.main([*args, "--no-browser"]), 0)
                start_server.assert_called_once_with(
                    port=port, models_dir=None, textures_dir=None, open_browser=False,
                    retry_next_port=not args,
                )

    def test_reports_unavailable_listen_ports(self) -> None:
        cases = ((["--port", "65535"], [65535]), (["--port", "8080"], [8080]), ([], [8080, 8081]))
        for args, ports in cases:
            with (
                self.subTest(ports=ports),
                tempfile.TemporaryDirectory() as assets,
                mock.patch.object(gallery.http.server.HTTPServer, "__init__", side_effect=OSError("address in use")) as bind,
                mock.patch.object(sys, "stderr", new_callable=io.StringIO) as stderr,
            ):
                with self.assertRaises(SystemExit) as raised:
                    gallery.main([*args, "--models-dir", assets, "--textures-dir", assets, "--no-browser"])

                self.assertEqual(raised.exception.code, 2)
                self.assertIn(f"can't listen on port {ports[-1]}", stderr.getvalue())
                self.assertEqual([attempt.args[0] for attempt in bind.call_args_list],
                                 [("127.0.0.1", port) for port in ports])

    def test_implicit_port_can_use_the_next_port(self) -> None:
        with (
            tempfile.TemporaryDirectory() as assets,
            mock.patch.object(gallery.http.server.HTTPServer, "__init__",
                              side_effect=[OSError("address in use"), None]) as bind,
            mock.patch.object(gallery.http.server.HTTPServer, "server_port", 8081, create=True),
            mock.patch.object(gallery.http.server.HTTPServer, "serve_forever") as serve,
            mock.patch.object(sys, "stdout", new_callable=io.StringIO) as stdout,
        ):
            self.assertEqual(gallery.main(["--models-dir", assets, "--textures-dir", assets, "--no-browser"]), 0)
            self.assertEqual([attempt.args[0] for attempt in bind.call_args_list],
                             [("127.0.0.1", 8080), ("127.0.0.1", 8081)])
            self.assertIn("http://localhost:8081", stdout.getvalue())
            serve.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
