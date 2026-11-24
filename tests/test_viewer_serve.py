"""Tests for viewer/serve.py: the loopback Host check, /world/ path resolution and the HTTP layer.

    python -I tests/test_viewer_serve.py

Everything runs on a temporary world folder and a loopback server: no game data, no browser, no network beyond
127.0.0.1, standard library only.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.parse
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
VIEWER = ROOT / "viewer"
if str(VIEWER) not in sys.path:
    sys.path.insert(0, str(VIEWER))

import serve  # noqa: E402


class HostAllowedTest(unittest.TestCase):
    """Only a Host header naming this machine may be answered (DNS rebinding)."""

    def test_loopback_hosts_are_allowed(self):
        for host in ("127.0.0.1", "127.0.0.1:8765", "localhost", "localhost:1", "LOCALHOST:8765",
                     "localhost.", "[::1]", "[::1]:8765"):
            with self.subTest(host=host):
                self.assertTrue(serve.host_allowed(host))

    def test_other_hosts_are_refused(self):
        for host in ("", "evil.example", "evil.example:8765", "127.0.0.1.evil.example",
                     "0.0.0.0:8765", "::1", "localhost.evil.example:8765"):
            with self.subTest(host=host):
                self.assertFalse(serve.host_allowed(host))

    def test_a_request_without_a_host_header_is_allowed(self):
        # Browsers always send Host; a missing one is a local tool, not a web page.
        self.assertTrue(serve.host_allowed(None))

    def test_well_formed_edge_cases_are_allowed(self):
        for host in (" localhost ", "localhost:80 ", "localhost.:8765", "127.0.0.1.", "127.0.0.1:65535", "localhost:0",
                     "[::1]:0", "[::1]:00080"):
            with self.subTest(host=host):
                self.assertTrue(serve.host_allowed(host))

    def test_malformed_ports_are_refused(self):
        # Each of these used to be allowed: the port after the colon was never looked at.
        for host in ("localhost:", "localhost:abc", "localhost:-1", "localhost:+80", "localhost: 80",
                     "localhost:8 0", "localhost:65536", "localhost:99999", "localhost:123456",
                     "localhost:８０", "127.0.0.1:0x50", "localhost:80.", "localhost:" + "9" * 5000):
            with self.subTest(host=host[:40]):
                self.assertFalse(serve.host_allowed(host))

    def test_malformed_ipv6_literals_are_refused(self):
        # The first seven used to be allowed: everything after "]" was ignored.
        for host in ("[::1]junk", "[::1]x:8765", "[::1].", "[::1]]", "[::1]:", "[::1]:abc", "[::1]:8765:1",
                     "[::1", "::1]", "[[::1]", "[]", "[]:8765", "[::1%25eth0]", "[0:0:0:0:0:0:0:1]x"):
            with self.subTest(host=host):
                self.assertFalse(serve.host_allowed(host))

    def test_other_malformed_hosts_are_refused(self):
        for host in (":8765", ":", "localhost..", "localhost..:8765", "127.0.0.1:80:80", "localhost:8765:1",
                     "localhost\t:8765", "local host", "localhost\x00"):
            with self.subTest(host=host):
                self.assertFalse(serve.host_allowed(host))


class SafeWorldPathTest(unittest.TestCase):
    """safe_world_path resolves /world/... inside the world folder or refuses it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.world = self.root / "world"
        (self.world / "sub").mkdir(parents=True)
        (self.world / "ok.bin").write_bytes(b"world-bytes")
        (self.world / "sub" / "inner.bin").write_bytes(b"inner")
        # A file the world must never reach, next to the world folder.
        (self.root / "secret.txt").write_text("secret", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def resolve(self, request_path):
        return serve.safe_world_path(str(self.world), request_path)

    def test_a_world_file_resolves_to_its_real_path(self):
        resolved = self.resolve("/world/ok.bin")
        self.assertIsNotNone(resolved)
        self.assertEqual(os.path.normcase(os.path.realpath(resolved)),
                         os.path.normcase(os.path.realpath(self.world / "ok.bin")))

    def test_a_nested_world_file_resolves(self):
        self.assertIsNotNone(self.resolve("/world/sub/inner.bin"))

    def test_query_and_fragment_are_not_part_of_the_path(self):
        self.assertIsNotNone(self.resolve("/world/ok.bin?cache=1"))
        self.assertIsNotNone(self.resolve("/world/ok.bin#top"))

    def test_requests_outside_world_are_refused(self):
        for path in ("/ok.bin", "/", "/world", "/world/../secret.txt", "/world/sub/../../secret.txt",
                     "/world/%2e%2e/secret.txt", "/world/..%2fsecret.txt", "/world/..\\secret.txt",
                     "/world//ok.bin", "/world/./ok.bin", "/world/", "/world/sub/", "/world/sub"):
            with self.subTest(path=path):
                self.assertIsNone(self.resolve(path))

    def test_nul_bytes_and_drive_letters_are_refused(self):
        for path in ("/world/ok%00.bin", "/world/C:secret.txt", "/world/sub/inner.bin:stream"):
            with self.subTest(path=path):
                self.assertIsNone(self.resolve(path))

    def test_windows_reserved_device_names_are_refused(self):
        for name in ("CON", "con.txt", "NUL", "COM1", "LPT9.log", "aux."):
            with self.subTest(name=name):
                self.assertIsNone(self.resolve("/world/" + name))

    def test_windows_device_aliases_are_refused_without_isreserved(self):
        # Python 3.12 has no os.path.isreserved. Check the fallback on newer Pythons too,
        # before realpath or any other filesystem call can touch a console or serial device.
        names = ("COM\u00b9", "com\u00b2.glb", "COM\u00b3.json", "LPT\u00b9", "lpt\u00b2.glb",
                 "LPT\u00b3.json", "CONIN$", "conout$.bin",
                 "NUL .glb", "COM1 .glb", "LPT9  .json")
        with mock.patch.object(serve.os.path, "isreserved", None, create=True):
            with mock.patch.object(serve.os.path, "realpath",
                                   side_effect=AssertionError("device path reached realpath")) as realpath:
                for name in names:
                    for prefix in ("/world/", "/world/sub/"):
                        with self.subTest(name=name, prefix=prefix):
                            self.assertIsNone(self.resolve(prefix + urllib.parse.quote(name, safe="")))
                realpath.assert_not_called()

    def test_device_like_regular_filenames_still_resolve(self):
        for name in ("COM10.glb", "LPT10.json", "COM\u2074.glb", "COM1_mesh.glb", "console.glb",
                     "my.CON", "input.CONIN$"):
            with self.subTest(name=name):
                self.assertIsNotNone(self.resolve("/world/" + urllib.parse.quote(name, safe="")))

    def test_a_missing_world_or_bad_argument_is_refused(self):
        self.assertIsNone(serve.safe_world_path("", "/world/ok.bin"))
        self.assertIsNone(serve.safe_world_path(str(self.world), None))
        self.assertIsNone(serve.safe_world_path(str(self.world), b"/world/ok.bin"))

    def test_a_symlink_out_of_the_world_is_refused(self):
        link = self.world / "escape.bin"
        try:
            os.symlink(self.root / "secret.txt", link)
        except (OSError, NotImplementedError) as e:      # Windows without the privilege
            self.skipTest(f"cannot create a symlink here: {e}")
        self.assertIsNone(self.resolve("/world/escape.bin"))


def raw_request(port, target, host="127.0.0.1", method="GET", extra_headers=()):
    """Send a request without letting a client library normalise the target path or merge headers.
    `extra_headers` are raw "Name: value" lines sent after Host, e.g. a second Host field."""
    lines = [f"{method} {target} HTTP/1.1"]
    if host is not None:
        lines.append(f"Host: {host}")
    lines += list(extra_headers)
    lines += ["Connection: close", "", ""]
    with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
        sock.sendall("\r\n".join(lines).encode("latin-1"))
        chunks = []
        while True:
            data = sock.recv(65536)
            if not data:
                break
            chunks.append(data)
    raw = b"".join(chunks)
    head, _, body = raw.partition(b"\r\n\r\n")
    status = int(head.split(b"\r\n", 1)[0].split()[1])
    return status, head.decode("latin-1"), body


class HandlerTest(unittest.TestCase):
    """/world/ files are served with the guards in front of them; a foreign Host gets 403."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        world = root / "world"
        (world / "sub").mkdir(parents=True)
        (world / "ok.bin").write_bytes(b"world-bytes")
        (world / "model.glb").write_bytes(b"glTF")
        (root / "site").mkdir()
        (root / "site" / "index.html").write_text("<!doctype html><title>viewer</title>", encoding="utf-8")
        (root / "secret.txt").write_text("secret", encoding="utf-8")

        class QuietHandler(serve.Handler):
            def log_message(self, *args):
                pass

        QuietHandler.world = str(world)      # a class body cannot see the enclosing function's locals

        cls.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(root / "site")))
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(5)
        cls._tmp.cleanup()

    def test_a_world_file_is_served_without_caching(self):
        status, head, body = raw_request(self.port, "/world/ok.bin", host=f"127.0.0.1:{self.port}")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"world-bytes")
        self.assertIn("cache-control: no-store", head.lower())

    def test_the_glb_mime_type_is_registered(self):
        status, head, _ = raw_request(self.port, "/world/model.glb", host=f"localhost:{self.port}")
        self.assertEqual(status, 200)
        self.assertIn("content-type: model/gltf-binary", head.lower())

    def test_head_and_query_strings_work(self):
        status, _, body = raw_request(self.port, "/world/ok.bin?cache=1", host="127.0.0.1", method="HEAD")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"")

    def test_a_missing_host_header_is_served(self):
        status, _, body = raw_request(self.port, "/world/ok.bin", host=None)
        self.assertEqual(status, 200)
        self.assertEqual(body, b"world-bytes")

    def test_a_foreign_host_gets_403(self):
        for target in ("/world/ok.bin", "/index.html"):
            with self.subTest(target=target):
                status, head, _ = raw_request(self.port, target, host="evil.example:8765")
                self.assertEqual(status, 403)
                self.assertIn("cache-control: no-store", head.lower())

    def test_duplicate_host_headers_get_400(self):
        # headers.get("Host") returned only the first field, so a loopback Host followed by a foreign one was served.
        loopback = f"127.0.0.1:{self.port}"
        cases = {
            "loopback then foreign": (loopback, ["Host: evil.example"]),
            "foreign then loopback": ("evil.example", [f"Host: {loopback}"]),
            "the same loopback twice": (loopback, [f"Host: {loopback}"]),
            "a lowercase second field": (loopback, ["host: evil.example"]),
            "an empty second field": (loopback, ["Host:"]),
            "three fields": (loopback, [f"Host: localhost:{self.port}", "Host: [::1]"]),
        }
        for method in ("GET", "HEAD"):
            for target in ("/world/ok.bin", "/index.html"):
                for label, (host, extra) in cases.items():
                    with self.subTest(method=method, target=target, case=label):
                        status, head, body = raw_request(self.port, target, host=host, method=method,
                                                         extra_headers=extra)
                        self.assertEqual(status, 400)
                        self.assertIn("cache-control: no-store", head.lower())
                        self.assertNotIn(b"world-bytes", body)
                        self.assertNotIn(b"<title>viewer</title>", body)

    def test_a_malformed_host_gets_403(self):
        for host in (f"127.0.0.1:{self.port}junk", "localhost:abc", f"[::1]evil.example:{self.port}",
                     "localhost:" + "9" * 5000):
            with self.subTest(host=host[:40]):
                status, _, body = raw_request(self.port, "/world/ok.bin", host=host)
                self.assertEqual(status, 403)
                self.assertNotIn(b"world-bytes", body)

    def test_traversal_attempts_get_403(self):
        for target in ("/world/../secret.txt", "/world/%2e%2e/secret.txt",
                       "/world/sub/../../secret.txt", "/world/..%5csecret.txt"):
            with self.subTest(target=target):
                status, _, body = raw_request(self.port, target, host=f"127.0.0.1:{self.port}")
                self.assertEqual(status, 403)
                self.assertNotIn(b"secret", body)

    def test_windows_device_aliases_get_403_without_opening_them(self):
        targets = ("/world/COM%C2%B9.glb", "/world/sub/lpt%C2%B2.json", "/world/CONIN%24",
                   "/world/CONOUT%24.bin", "/world/NUL%20.glb")
        with mock.patch.object(serve.os.path, "isreserved", None, create=True):
            # If the path guard regresses, fail without ever opening an actual Windows device.
            with mock.patch("builtins.open", side_effect=AssertionError("device path reached open")) as opened:
                for method in ("GET", "HEAD"):
                    for target in targets:
                        with self.subTest(method=method, target=target):
                            status, _, _ = raw_request(self.port, target, method=method)
                            self.assertEqual(status, 403)
                opened.assert_not_called()

    def test_the_viewer_page_itself_is_served(self):
        # http.server itself folds a leading "//" into "/" before the handler sees the path.
        for target in ("/index.html", "/", "/?x=1", "/index.html?v=2", "/index.html#pos", "//index.html"):
            for method in ("GET", "HEAD"):
                with self.subTest(target=target, method=method):
                    status, head, body = raw_request(self.port, target, host=f"127.0.0.1:{self.port}",
                                                     method=method)
                    self.assertEqual(status, 200)
                    self.assertIn("content-type: text/html", head.lower())
                    if method == "GET":
                        self.assertIn(b"<title>viewer</title>", body)

    def test_nothing_else_outside_world_is_served(self):
        # These reached SimpleHTTPRequestHandler unguarded: on Windows /NUL opened the device (200, empty) and
        # the stream, trailing-dot and trailing-space forms returned the page or the server's own source.
        site = Path(self._tmp.name) / "site"
        (site / "other.txt").write_text("other-bytes", encoding="utf-8")
        (site / "sub").mkdir(exist_ok=True)
        (site / "sub" / "index.html").write_text("<title>nested</title>", encoding="utf-8")
        targets = ("/NUL", "/nul.html", "/CON", "/COM1", "/CONIN%24", "/index.html::$DATA", "/index.html::%24DATA",
                   "/index.html.", "/index.html%20", "/index.html/", "/INDEX.HTML", "/index%2Ehtml",
                   "/./index.html", "/sub/../index.html", "/other.txt", "/other.txt::$DATA",
                   "/sub/", "/sub/index.html", "/../secret.txt", "/%2e%2e/secret.txt", "/..%5csecret.txt",
                   "/world", "/worldx/ok.bin", "/favicon.ico")
        with mock.patch("builtins.open", side_effect=AssertionError("a refused path reached open")) as opened:
            for method in ("GET", "HEAD"):
                for target in targets:
                    with self.subTest(method=method, target=target):
                        status, head, body = raw_request(self.port, target, method=method)
                        self.assertEqual(status, 404)
                        self.assertIn("cache-control: no-store", head.lower())
                        for leaked in (b"<title>viewer", b"<title>nested", b"other-bytes", b"secret", b"world-bytes"):
                            self.assertNotIn(leaked, body)
            opened.assert_not_called()

    def test_translate_path_refuses_what_send_head_refuses(self):
        handler = serve.Handler.__new__(serve.Handler)      # no socket: only the path mapping is under test
        handler.directory = str(Path(self._tmp.name) / "site")
        handler.world = None
        for target in ("/NUL", "/index.html::$DATA", "/index.html.", "/other.txt", "/world/ok.bin"):
            with self.subTest(target=target):
                self.assertEqual(handler.translate_path(target), "")
        self.assertTrue(handler.translate_path("/index.html").endswith("index.html"))


class NoWorldHandlerTest(unittest.TestCase):
    """Without --world the page is still served and /world/ is simply absent (the viewer shows test terrain)."""

    def test_world_paths_are_404_and_the_page_is_served(self):
        with tempfile.TemporaryDirectory() as tmp:
            site = Path(tmp) / "site"
            (site / "world").mkdir(parents=True)
            (site / "index.html").write_text("<title>viewer</title>", encoding="utf-8")
            (site / "world" / "world.json").write_text("{}", encoding="utf-8")

            class QuietHandler(serve.Handler):
                world = None

                def log_message(self, *args):
                    pass

            server = http.server.ThreadingHTTPServer(
                ("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(site)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                port = server.server_address[1]
                status, _, body = raw_request(port, "/", host=f"127.0.0.1:{port}")
                self.assertEqual(status, 200)
                self.assertIn(b"<title>viewer</title>", body)
                # A world folder that happens to sit in the viewer folder is not served: /world/ exists only
                # with --world, behind safe_world_path.
                for target in ("/world/world.json", "/world/NUL", "/world/world.json::$DATA"):
                    with self.subTest(target=target):
                        status, _, body = raw_request(port, target, host=f"127.0.0.1:{port}")
                        self.assertEqual(status, 404)
                        self.assertNotIn(b"{}", body)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(5)


class WorldArgumentTest(unittest.TestCase):
    def test_explicit_world_must_be_a_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing"
            file = Path(tmp) / "world.json"
            file.write_text("{}", encoding="utf-8")
            for path in (missing, file):
                with self.subTest(path=path):
                    result = subprocess.run(
                        [sys.executable, "-I", str(VIEWER / "serve.py"), "--port", "0",
                         "--world", str(path)],
                        capture_output=True, text=True, errors="replace", timeout=5)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn("--world", result.stderr)
                    self.assertIn(str(path), result.stderr)
                    self.assertNotIn("Traceback", result.stderr)


class PortArgumentTest(unittest.TestCase):
    """--port is 0-65535; outside that it used to reach socketserver and raise OverflowError."""

    def test_valid_ports(self):
        self.assertEqual(serve.port_number("0"), 0)
        self.assertEqual(serve.port_number("8765"), 8765)
        self.assertEqual(serve.port_number("65535"), 65535)

    def test_invalid_ports_are_refused_by_argparse(self):
        for text in ("-1", "65536", "70000", "abc", ""):
            with self.subTest(text=text):
                with self.assertRaises(argparse.ArgumentTypeError):
                    serve.port_number(text)

    def test_the_cli_reports_a_bad_port_without_a_traceback(self):
        result = subprocess.run(
            [sys.executable, "-I", str(VIEWER / "serve.py"), "--port", "70000"],
            capture_output=True, text=True, errors="replace")
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be 0-65535", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_the_cli_reports_a_port_in_use_without_a_traceback(self):
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            port = taken.getsockname()[1]
            result = subprocess.run(
                [sys.executable, "-I", str(VIEWER / "serve.py"), "--port", str(port)],
                capture_output=True, text=True, errors="replace", timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn(f"can't listen on port {port}", result.stderr)
        self.assertIn("--port 0", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_a_second_viewer_on_the_same_port_is_refused(self):
        """Two real viewers. With SO_REUSEADDR set, Windows let the second one bind the port the first was serving:
        it printed its URL and never got a request. The test above holds the port with a plain socket, which
        Windows does refuse to share, so it did not see this."""
        first = subprocess.Popen([sys.executable, "-I", str(VIEWER / "serve.py"), "--port", "0"],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        second = None
        try:
            line = first.stdout.readline()                                        # it is serving now
            self.assertIn("viewer: http://127.0.0.1:", line)
            port = urllib.parse.urlsplit(line.split()[1]).port
            second = subprocess.Popen([sys.executable, "-I", str(VIEWER / "serve.py"), "--port", str(port)],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                out, err = second.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                self.fail(f"a second viewer is serving port {port} next to the first one")
        finally:
            for proc in (first, second):
                if proc is not None:
                    if proc.poll() is None:
                        proc.kill()
                    proc.communicate()
        self.assertEqual(second.returncode, 2, err)
        self.assertEqual(out, "")
        self.assertIn(f"can't listen on port {port}", err)
        self.assertNotIn("Traceback", err)

    def test_a_restart_binds_the_port_a_closed_viewer_just_served(self):
        handler = functools.partial(serve.Handler, directory=str(VIEWER))
        port = 0
        with mock.patch.object(serve.Handler, "log_message", lambda *args: None):
            for _ in range(3):
                server = serve.ViewerServer(("127.0.0.1", port), handler)
                port = server.server_address[1]
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
                        client.sendall(f"GET / HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode("ascii"))
                        self.assertTrue(client.recv(64).startswith(b"HTTP/1.0 200"))
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(10)
                self.assertFalse(thread.is_alive(), "the server thread is still running after shutdown()")

    @unittest.skipIf(os.name == "nt", "sends SIGINT, which a Windows child started this way does not get")
    def test_ctrl_c_stops_the_viewer_without_a_traceback(self):
        import signal
        # A test runner started in the background has SIGINT ignored, and a child inherits that: Python then
        # leaves it ignored and Ctrl+C would never arrive. Give the child the default disposition back.
        proc = subprocess.Popen([sys.executable, "-I", str(VIEWER / "serve.py"), "--port", "0"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL))
        try:
            self.assertIn("viewer: http://127.0.0.1:", proc.stdout.readline())   # it is serving now
            proc.send_signal(signal.SIGINT)
            out, err = proc.communicate(timeout=60)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
        self.assertEqual(proc.returncode, 0)
        self.assertIn("viewer: stopped", out)
        self.assertNotIn("Traceback", err)

    def test_ctrl_c_right_after_the_serving_line_still_stops_the_viewer(self):
        """The test above sends SIGINT as soon as it reads the serving line. When the signal was handled before main()
        was inside its try, the KeyboardInterrupt escaped (exit -2, a traceback): 24 of 300 runs with 12 at once
        under load. Here the interrupt comes from printing that line, every time."""
        import contextlib
        import io
        from unittest import mock

        def print_then_ctrl_c(*args, **kwargs):
            print(*args, **kwargs)
            if args and str(args[0]).startswith("viewer: http://127.0.0.1:"):
                raise KeyboardInterrupt

        out = io.StringIO()
        world = serve.Handler.world
        try:
            with mock.patch.object(sys, "argv", ["serve.py", "--port", "0"]), \
                    mock.patch.object(serve, "print", print_then_ctrl_c, create=True), \
                    contextlib.redirect_stdout(out):
                serve.main()
        except KeyboardInterrupt:
            self.fail("a Ctrl+C right after the serving line escaped main()")
        finally:
            serve.Handler.world = world
        self.assertIn("viewer: http://127.0.0.1:", out.getvalue())
        self.assertIn("viewer: stopped", out.getvalue())


if __name__ == "__main__":
    unittest.main()
