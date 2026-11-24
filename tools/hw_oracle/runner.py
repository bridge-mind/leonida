"""Find the tools, build the launcher, assemble kernels and run them on the GPU. Standard library only.

Two launchers: on Windows the HIP runtime that ships in the AMD driver (hw_oracle.c, this file), on Linux raw
dispatches through the kernel driver (hw_oracle_drm.c, drm.py, which also lists its own variables). open_oracle()
picks the one for the platform.

Tools are located through environment variables first, then by looking in the usual places:
    LEONIDA_AMD_CLANG   AMD clang (its folder must also hold ld.lld); default: amdclang / clang on PATH that lists
                        the amdgcn target (a launcher without ld.lld beside it is followed to the clang it is
                        installed as), %HIP_PATH% / %ROCM_PATH%, a .venv holding the `rocm-sdk-core` wheel
    LEONIDA_HIP_DLL     the HIP runtime (amdhip64_7.dll); default: the one in System32 (it ships with the driver)
    LEONIDA_HOST_CC     a C compiler for hw_oracle.c (MinGW gcc, or cl.exe from a Developer shell); default: gcc on
                        PATH, C:\\msys64\\ucrt64\\bin, then on Windows a cl on PATH or Visual Studio's (vswhere)
    LEONIDA_HW_ARCH     the GPU architecture to assemble for; default: what amdgpu-arch reports, else gfx1150
Anything missing raises Unavailable, which callers turn into exit 77 ("skipped").
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

try:
    from . import asm
except ImportError:   # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import asm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SENTINEL = 0xDEAD0000


class Unavailable(Exception):
    """A required tool or the GPU is not there."""


def _exe(name):
    return name + (".exe" if os.name == "nt" else "")


def _has_amdgcn(clang):
    try:
        out = subprocess.run([clang, "--print-targets"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return "amdgcn" in out


def _installed_clang(clang):
    """The clang a launcher stands for. The rocm-sdk wheels put `amdclang` in the Python environment's Scripts (bin)
    folder: it has the amdgcn target, but ld.lld and amdgpu-arch sit beside the real clang, in the folder `--version`
    names as InstalledDir. A clang with its ld.lld beside it, or one that names no such folder, is returned as it is."""
    if os.path.isfile(os.path.join(os.path.dirname(clang), _exe("ld.lld"))):
        return clang
    try:
        out = subprocess.run([clang, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return clang
    for line in out.splitlines():
        if line.startswith("InstalledDir:"):
            real = os.path.join(line.split(":", 1)[1].strip(), _exe("clang"))
            if os.path.isfile(real):
                return real
    return clang


def find_clang():
    env = os.environ.get("LEONIDA_AMD_CLANG")
    if env:
        if not os.path.isfile(env):
            raise Unavailable(f"LEONIDA_AMD_CLANG={env} does not exist")
        return env
    cands = [shutil.which(n) for n in ("amdclang", "clang")]
    for var in ("HIP_PATH", "ROCM_PATH"):
        root = os.environ.get(var)
        if root:
            cands += [os.path.join(root, "lib", "llvm", "bin", _exe("clang")), os.path.join(root, "bin", _exe("clang"))]
    tail = os.path.join("Lib", "site-packages", "_rocm_sdk_core", "lib", "llvm", "bin", _exe("clang"))
    cands.append(os.path.join(REPO, ".venv", tail))   # only the repo root's venv, never a glob under the cwd
    for c in cands:
        if c and os.path.isfile(c) and _has_amdgcn(c):
            return _installed_clang(c)
    raise Unavailable("no AMD clang (set LEONIDA_AMD_CLANG; `pip install rocm-sdk-core` from "
                      "https://stable.repo.amd.com/rocm/whl-next/ provides one)")


def find_hip_dll():
    env = os.environ.get("LEONIDA_HIP_DLL")
    if env:
        if not os.path.isfile(env):
            raise Unavailable(f"LEONIDA_HIP_DLL={env} does not exist")
        return env
    if os.name != "nt":
        raise Unavailable("the launcher targets the Windows AMD driver (amdhip64_7.dll)")
    for n in ("amdhip64_7.dll", "amdhip64_6.dll"):   # the launcher loads whatever path this returns
        p = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", n)
        if os.path.isfile(p):
            return p
    raise Unavailable("no amdhip64_7.dll in System32 (the AMD Windows driver provides it)")


def _is_msvc(cc):
    """cl.exe (MSVC or clang-cl) takes /Fe, not -o."""
    return os.path.basename(cc).lower() in ("cl.exe", "cl", "clang-cl.exe", "clang-cl")


def find_vcvars():
    """vcvars64.bat of the newest Visual Studio with the x64 C++ tools, as scripts/build_runtime.ps1 finds it."""
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = os.path.join(pf86, "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return None
    try:
        out = subprocess.run([vswhere, "-latest", "-products", "*", "-requires",
                              "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                             capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    bat = os.path.join(out[0].strip(), "VC", "Auxiliary", "Build", "vcvars64.bat") if out else ""
    return bat if os.path.isfile(bat) else None


def find_cc():
    """The host compiler for hw_oracle.c: ('gcc', path), ('msvc', cl path) or ('vcvars', vcvars64.bat path)."""
    env = os.environ.get("LEONIDA_HOST_CC")
    if env:
        if not os.path.isfile(env):
            raise Unavailable(f"LEONIDA_HOST_CC={env} does not exist")
        return ("msvc" if _is_msvc(env) else "gcc"), env
    for c in (shutil.which("gcc"), r"C:\msys64\ucrt64\bin\gcc.exe", r"C:\msys64\mingw64\bin\gcc.exe"):
        if c and os.path.isfile(c):
            return "gcc", c
    if os.name == "nt":
        # A cl already on PATH (a Developer shell) has its environment; otherwise set it up from vcvars64.bat.
        cl = shutil.which("cl")
        if cl:
            return "msvc", cl
        bat = find_vcvars()
        if bat:
            return "vcvars", bat
    raise Unavailable("no C compiler for hw_oracle.c: MinGW gcc or Visual Studio's C++ tools (set LEONIDA_HOST_CC)")


def build_command(kind, cc, src, out, workdir):
    """argv that compiles src into out; MSVC's objects and the vcvars script go to workdir, the cwd to run it in."""
    if kind == "gcc":
        return [cc, "-O1", src, "-o", out]
    cl = [cc if kind == "msvc" else "cl", "/nologo", "/O1", src, f"/Fe{out}", f"/Fo{os.path.join(workdir, '')}"]
    if kind == "msvc":
        return cl
    # cmd.exe strips the outer quotes of /c's argument, so the call goes through a script file.
    # vcvars64.bat runs vswhere itself, so its folder goes on PATH first (as scripts/build_llvm_mc.ps1 does).
    installer = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                             "Microsoft Visual Studio", "Installer")
    name = "build_hw_oracle.cmd"
    script = os.path.join(workdir, name)
    # Written as UTF-8, with chcp 65001 so cmd.exe reads it as UTF-8 on any console code page: the checkout, %TEMP%
    # and Visual Studio paths in here reach cl intact when one has a character outside ASCII (a user name, say).
    with open(script, "w", encoding="utf-8", newline="\n") as f:
        f.write(f'@echo off\r\nchcp 65001 >nul\r\nset "PATH={installer};%PATH%"\r\ncall "{cc}" >nul || exit /b 1\r\n'
                + subprocess.list2cmdline(cl) + "\r\n")
    # Relative to workdir, not the full path: cmd.exe /c loses a full path's quotes when it holds & or ^ (and
    # list2cmdline adds none without a space), and `.\` is still found with NoDefaultCurrentDirectoryInExePath set.
    return ["cmd.exe", "/d", "/c", "." + os.sep + name]


def detect_arch(clang):
    env = os.environ.get("LEONIDA_HW_ARCH")
    if env:
        return env
    tool = os.path.join(os.path.dirname(clang), _exe("amdgpu-arch"))
    if os.path.isfile(tool):
        try:
            out = subprocess.run([tool], capture_output=True, text=True, timeout=60).stdout.split()
            if out:
                return out[0].split(":")[0]   # amdgpu-arch prints 'gfx1010:xnack-': the kernels want the bare name
        except (OSError, subprocess.SubprocessError):
            pass
    return "gfx1150"


def _find_lld(clang):
    p = os.path.join(os.path.dirname(clang), _exe("ld.lld"))
    if os.path.isfile(p):
        return p
    found = shutil.which("ld.lld")
    if found:
        return found
    raise Unavailable("no ld.lld next to the AMD clang")


class Oracle:
    """Assembles batches of scalar programs for one architecture and runs them on device 0."""

    code_object = None   # None: the assembler's default version; 5 after an older HIP runtime refused that
    true16 = False       # True: this assembler wants v8.l for a 16-bit operand

    def __init__(self, arch=None, workdir=None, chunk=500):
        if not os.path.isfile(os.path.join(HERE, "hw_oracle.c")):
            raise Unavailable("hw_oracle.c is missing")
        self.clang = find_clang()
        self.lld = _find_lld(self.clang)
        self.dll = find_hip_dll()
        self.arch = arch or detect_arch(self.clang)
        try:
            asm.family(self.arch)
        except ValueError as e:   # a GPU the kernels have no spellings for is a skip, like a missing tool
            raise Unavailable(f"{e} (set LEONIDA_HW_ARCH)") from None
        self.chunk = chunk
        self.workdir = workdir or tempfile.mkdtemp(prefix="leonida-hw-oracle-")
        self.exe = self._build_launcher()
        self.device = self._probe()
        self.true16 = self._needs_true16()
        self._probe_waves()

    def _build_launcher(self):
        src = os.path.join(HERE, "hw_oracle.c")
        with open(src, "rb") as f:
            tag = hashlib.sha1(f.read()).hexdigest()[:10]
        exe = os.path.join(tempfile.gettempdir(), f"leonida_hw_oracle_{tag}{'.exe' if os.name == 'nt' else ''}")
        if not os.path.isfile(exe):
            # cl appends .exe to an /Fe name without one, so the temporary name keeps the extension last
            tmp = exe[:-4] + f".{os.getpid()}.exe" if os.name == "nt" else exe + f".{os.getpid()}"
            kind, cc = find_cc()
            # The compiler's messages are only quoted in the error below; one in another code page must not raise.
            r = subprocess.run(build_command(kind, cc, src, tmp, self.workdir), capture_output=True, text=True,
                               encoding="utf-8", errors="replace", cwd=self.workdir)
            if r.returncode:
                # cl reports compile errors on stdout
                raise Unavailable("hw_oracle.c did not build: " + (r.stderr.strip() or r.stdout.strip())[:300])
            os.replace(tmp, exe)
        return exe

    def _probe(self):
        """Run an empty kernel list: confirms the DLL loads and a device exists."""
        text, names, _ = asm.module_text([["v_mov_b32 v8, 0", "v_mov_b32 v9, 0", "v_mov_b32 v10, 0", "v_mov_b32 v11, 0"]],
                                         self.arch, prefix="probe")
        co, err = self._assemble(text)
        if co is None:
            raise Unavailable("probe kernel did not assemble: " + err[:300])
        rc, out, err = self._launch(co, names)
        if rc == 3 and any(e in err for e in ("hip error 209", "hip error 303")) and self.code_object is None:
            # LLVM 23 writes code object v6, which HIP runtimes from 2024 (amdhip64_6.dll) refuse as "no binary for
            # this GPU" (209) or "shared object init failed" (303). Ask for v5; an error that stays has another cause
            # and is reported below.
            self.code_object = 5
            co, why = self._assemble(text)
            if co is None:
                raise Unavailable("the HIP runtime refused the code object and v5 did not assemble: " + why[:300])
            rc, out, err = self._launch(co, names)
        if rc == 77:
            raise Unavailable(err.strip() or "no AMD device")
        if rc:
            raise Unavailable(f"probe launch failed (exit {rc}): {err.strip()[:300]}")
        head = out.splitlines()[0] if out else ""
        if "arch=" in head and self.arch not in head:
            raise Unavailable(f"device reports {head!r} but assembling for {self.arch} (set LEONIDA_HW_ARCH)")
        return head.lstrip("# ")

    def _needs_true16(self):
        """Whether this assembler rejects `v_cvt_f16_f32 v8, v1` and takes `v8.l` (LLVM 23 on GFX11)."""
        return asm.family(self.arch) != "gfx10" and asm.needs_true16(lambda text: self._assemble(text)[0] is not None)

    def _probe_waves(self):
        """Each wave size really runs as that size: a 64-thread group stores v_mbcnt lane ids. In a wave64 kernel
        thread 32 is lane 32 (mbcnt_hi adds 32); in a wave32 kernel it is lane 0 of the second wave. A runtime that
        ignored the wave size in the code object would fail here instead of producing plausible-looking data."""
        store = "global_store_b32" if asm.family(self.arch) == "gfx11" else "global_store_dword"
        tail = ["v_lshlrev_b32 v2, 2, v0", f"{store} v2, v1, s[2:3]", "s_endpgm"]
        lo = ["v_mbcnt_lo_u32_b32 v1, -1, 0"]
        k64 = asm.Kernel(lo + ["v_mbcnt_hi_u32_b32 v1, -1, v1"] + tail, 64, 64, 64, raw=True)
        k32 = asm.Kernel(lo + tail, 32, 64, 64, raw=True)
        w64, w32 = self.run([k64, k32], words=64)
        if w64 != list(range(64)):
            raise RuntimeError(f"wave64 probe: lane ids {w64!r}; the runtime did not run the kernel as wave64")
        if w32 != list(range(32)) * 2:
            raise RuntimeError(f"wave32 probe: lane ids {w32!r}; the runtime did not run the kernel as wave32")

    def _assemble(self, text, wave=32):
        s = os.path.join(self.workdir, "k.s")
        o = os.path.join(self.workdir, "k.o")
        co = os.path.join(self.workdir, "k.co")
        with open(s, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        flags = ["-mwavefrontsize64"] if wave == 64 else []
        if self.code_object:
            flags.append(f"-mcode-object-version={self.code_object}")
        r = subprocess.run([self.clang, "--target=amdgcn-amd-amdhsa", f"-mcpu={self.arch}", *flags, "-c", s, "-o", o],
                           capture_output=True, text=True)
        if r.returncode:
            return None, r.stderr
        # Some Windows LLD builds default to PE even when the input object is AMDGPU ELF.
        r = subprocess.run([self.lld, "-m", "elf64_amdgpu", "-shared", o, "-o", co],
                           capture_output=True, text=True)
        if r.returncode:
            return None, r.stderr
        return co, ""

    def _launch(self, co, names, words=4):
        try:
            r = subprocess.run([self.exe, co, str(words), self.dll], input="\n".join(names) + "\n", capture_output=True, text=True)
        except OSError as e:
            if getattr(e, "winerror", None) != 225:
                raise
            # Defender quarantined the launcher just built (a small unsigned exe that loads a DLL): not a crash
            raise Unavailable(f"Windows Defender blocked the launcher {self.exe} (WinError 225); add an exclusion for "
                              f"{os.path.dirname(self.exe)} (Windows Security, Virus & threat protection) and rerun") from None
        return r.returncode, r.stdout, r.stderr

    def run(self, programs, words=4):
        """Run each program (a list of lines). Returns one entry per program: a list of `words` ints, or a str
        explaining why it could not run (assembler rejection, or a MODE field this architecture cannot set)."""
        results = [None] * len(programs)
        for i, p in enumerate(programs):
            why = asm.mode_skip_reason(p.mode, self.arch) if isinstance(p, asm.Kernel) else None
            if why:
                results[i] = why            # never assembled: the directive would fail its whole chunk
        for wave in (32, 64):
            todo = [i for i, p in enumerate(programs) if results[i] is None and
                    (p.wave if isinstance(p, asm.Kernel) else 32) == wave]
            for lo in range(0, len(todo), self.chunk):
                try:
                    self._run_chunk(programs, todo[lo:lo + self.chunk], results, words, wave)
                except asm.ModeUnavailable as e:
                    raise Unavailable(str(e)) from None
        return results

    def _run_chunk(self, programs, idx, results, words, wave=32):
        live = list(idx)
        while live:
            text, names, starts = asm.module_text([programs[i] for i in live], self.arch, wave=wave)
            if self.true16:
                text = asm.true16_spelling(text)
            if asm.family(self.arch) != "gfx10":
                text = asm.opsel16_spelling(text)    # needed even where the true16 probe says no (asm.py)
            co, err = self._assemble(text, wave)
            if co is not None:
                break
            bad = asm.failing_kernels(err, starts)
            if not bad:
                raise RuntimeError("assembler failed with no kernel to blame:\n" + err[:600])
            first = err.strip().splitlines()[0] if err.strip() else "?"
            for k in bad:
                results[live[k]] = "assembler rejected: " + first.split("error:", 1)[-1].strip()
            live = [i for k, i in enumerate(live) if k not in bad]
        else:
            # every program in the chunk was rejected: that is a broken harness (a bad prologue, the wrong
            # architecture), not a thousand skips
            raise RuntimeError("the assembler rejected every program in a chunk: " +
                               (str(results[idx[0]]) if idx else "empty chunk"))
        tokens = []
        for n, i in zip(names, live):
            p = programs[i]
            tokens.append(f"{n}:{p.threads}:{p.words}" if isinstance(p, asm.Kernel) else n)
        top = max([4] + [programs[i].words for i in live if isinstance(programs[i], asm.Kernel)])
        rc, out, err = self._launch(co, tokens, top if top > 4 else words)
        if rc:
            raise RuntimeError(f"hw_oracle exited {rc}: {err.strip()[:300]}")
        got = {}
        for line in out.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name, *w = line.split()
            got[name] = [int(x, 16) for x in w]
        for n, i in zip(names, live):
            results[i] = got.get(n, "kernel produced no output")

    def close(self):
        shutil.rmtree(self.workdir, ignore_errors=True)


def open_oracle(**kw):
    """An Oracle for this platform, or raise Unavailable."""
    if sys.platform.startswith("linux"):
        try:
            from . import drm
        except ImportError:   # run as a script
            import drm
        return drm.DrmOracle(**kw)
    return Oracle(**kw)


def main():
    try:
        o = open_oracle()
    except Unavailable as e:
        print(f"hardware oracle unavailable: {e}")
        return 77
    print(f"hardware oracle ready: {o.device}; clang {o.clang}")
    o.close()
    return 0


if __name__ == "__main__":
    # Through the importable module: drm.py imports runner, and both must share one Unavailable and one Oracle.
    import runner as _runner
    sys.exit(_runner.main())
