"""The Linux backend of the hardware oracle: the same kernels, run as raw compute dispatches through the kernel
driver (libdrm_amdgpu) by hw_oracle_drm.c. Standard library only. No ROCm: an assembler with the amdgcn target
(clang from the distribution's LLVM), gcc and the libdrm headers are enough.

    LEONIDA_HW_DRM_NODE           the render node of the GPU to use, for example /dev/dri/renderD129. Required: there
                                  is no default, so that a raw command stream never goes to a GPU by accident.
                                  `ls -l /dev/dri/by-path` shows which node belongs to which PCI device.
    LEONIDA_HW_DRM_ALLOW_DISPLAY  set to 1 to use a GPU that has a display connected. A kernel that hangs makes the
                                  kernel driver reset the GPU, and with it everything shown on that display.
    LEONIDA_AMD_CLANG, LEONIDA_HOST_CC, LEONIDA_HW_ARCH   as in runner.py

The launcher supports compute engine version 10 only: GFX10, which is RDNA1 and RDNA2.
"""
import glob
import hashlib
import os
import shutil
import struct
import subprocess
import tempfile

try:
    from . import asm, runner
except ImportError:   # run as a script, or imported by runner.py run as a script
    import asm
    import runner

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "hw_oracle_drm.c")
Unavailable = runner.Unavailable


def connected_displays(node, sysfs="/sys/class/drm"):
    """The connectors of the GPU behind a render node that have a display attached, as sysfs names them."""
    found = []
    for status in sorted(glob.glob(os.path.join(sysfs, os.path.basename(node), "device", "drm", "card*", "card*-*",
                                                "status"))):
        try:
            with open(status, encoding="ascii") as f:
                if f.read().strip() == "connected":
                    found.append(os.path.basename(os.path.dirname(status)))
        except OSError:
            pass
    return found


def find_node(environ=None, sysfs="/sys/class/drm"):
    """The render node to use, or raise Unavailable."""
    environ = os.environ if environ is None else environ
    node = environ.get("LEONIDA_HW_DRM_NODE")
    if not node:
        raise Unavailable("set LEONIDA_HW_DRM_NODE to the render node of a GFX10 AMD GPU that drives no display "
                          "(`ls -l /dev/dri/by-path` lists them)")
    if not os.path.exists(node):
        raise Unavailable(f"LEONIDA_HW_DRM_NODE={node} does not exist")
    displays = connected_displays(node, sysfs)
    if displays and environ.get("LEONIDA_HW_DRM_ALLOW_DISPLAY") != "1":
        raise Unavailable(f"{node} drives a display ({', '.join(displays)}): a kernel that hangs resets that GPU. "
                          "Use a GPU without one, or set LEONIDA_HW_DRM_ALLOW_DISPLAY=1")
    return node


def _symbols(elf):
    """([(section name, address, file offset, size)], [(symbol, section number, value)]) of a little-endian ELF64."""
    if elf[:6] != b"\x7fELF\x02\x01":
        raise ValueError("not a little-endian ELF64 file")
    shoff, = struct.unpack_from("<Q", elf, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    sections = [struct.unpack_from("<IIQQQQIIQQ", elf, shoff + i * shentsize) for i in range(shnum)]

    def string(table, at):
        start = sections[table][4] + at
        return elf[start:elf.index(b"\0", start)].decode("ascii", "replace")

    symbols = []
    for s in sections:
        if s[1] != 2:   # SHT_SYMTAB
            continue
        for at in range(s[4], s[4] + s[5], 24):
            name, _info, _other, shndx, value, _size = struct.unpack_from("<IBBHQQ", elf, at)
            if name and 0 < shndx < len(sections):
                symbols.append((string(s[6], name), shndx, value))
    return [(string(shstrndx, s[0]), s[3], s[4], s[5]) for s in sections], symbols


def text_section(elf):
    """(the bytes of .text, {symbol: offset in .text}) of a little-endian ELF64 relocatable object."""
    sections, symbols = _symbols(elf)
    names = [s[0] for s in sections]
    if ".text" not in names:
        raise ValueError("no .text section")
    text = names.index(".text")
    _name, address, offset, size = sections[text]
    return elf[offset:offset + size], {name: value - address for name, shndx, value in symbols if shndx == text}


def kernel_registers(elf):
    """{kernel: (COMPUTE_PGM_RSRC1, COMPUTE_PGM_RSRC2, wave size)} from the kernel descriptors (the <name>.kd
    symbols) of an assembled module.

    The descriptor is where the assembler puts what a runtime must program before it starts the kernel: the
    register counts and the float mode (rounding, denormals, IEEE, DX10 clamp) in RSRC1, the user and system SGPRs
    and the workgroup's LDS in RSRC2, the wave size in the code properties. The HIP runtime follows it, and so does
    the launcher. What the launcher does not provide is refused here instead of being run differently."""
    sections, symbols = _symbols(elf)
    out = {}
    for name, shndx, value in symbols:
        if not name.endswith(".kd"):
            continue
        _section, address, offset, size = sections[shndx]
        if value - address + 64 > size:
            raise ValueError(f"{name}: the kernel descriptor runs past its section")
        at = offset + value - address
        group, private = struct.unpack_from("<II", elf, at)
        rsrc3, rsrc1, rsrc2, properties = struct.unpack_from("<IIIH", elf, at + 44)
        kernel = name[:-3]
        if properties & ~0x0408 or not properties & 0x0008:
            raise RuntimeError(f"{kernel}: kernel code properties {properties:#06x}; the launcher passes the kernarg "
                               "pointer in s[0:1] and nothing else")
        if private or rsrc3 or rsrc2 & 1:
            raise RuntimeError(f"{kernel}: scratch memory and COMPUTE_PGM_RSRC3 are not supported by the launcher")
        # The descriptor's group segment is the workgroup's LDS, and COMPUTE_PGM_RSRC2.LDS_SIZE (bits 15..23) is the
        # field the hardware reads, in 512-byte units. The assembler leaves that field 0 (measured on gfx1036:
        # 4096 bytes of LDS give group 4096 and rsrc2 0x84), so the launcher programs it from the group segment, as
        # the HIP runtime does on Windows. The hardware gives a workgroup at most 65536 bytes of LDS.
        if group > 65536:
            raise RuntimeError(f"{kernel}: {group} bytes of LDS; a workgroup gets at most 65536")
        rsrc2 = (rsrc2 & ~(0x1FF << 15)) | (((group + 511) // 512) << 15)
        if (rsrc2 >> 1) & 0x1F != 2:
            raise RuntimeError(f"{kernel}: {(rsrc2 >> 1) & 0x1F} user SGPRs; the launcher fills two")
        out[kernel] = (rsrc1, rsrc2, 32 if properties & 0x0400 else 64)
    return out


def launcher_tokens(tokens, offsets, registers):
    """name[:threads[:words]] -> name@offset/rsrc1/rsrc2/wave[:threads[:words]], the form hw_oracle_drm takes."""
    out = []
    for token in tokens:
        name, sep, rest = token.partition(":")
        if name not in offsets or name not in registers:
            raise RuntimeError(f"kernel {name} is not in the assembled module")
        rsrc1, rsrc2, wave = registers[name]
        out.append(f"{name}@{offsets[name]}/{rsrc1:#x}/{rsrc2:#x}/{wave}{sep}{rest}")
    return out


class DrmOracle(runner.Oracle):
    """runner.Oracle with the assembling and launching done for the kernel driver instead of the HIP runtime."""

    def __init__(self, arch=None, workdir=None, chunk=500):
        if not os.path.isfile(SOURCE):
            raise Unavailable("hw_oracle_drm.c is missing")
        self.node = find_node()
        self.clang = runner.find_clang()
        self.chunk = chunk
        self.workdir = workdir or tempfile.mkdtemp(prefix="leonida-hw-oracle-")
        self._offsets, self._registers = {}, {}
        self.exe = self._build_launcher()
        self.arch = arch or os.environ.get("LEONIDA_HW_ARCH") or self._device_arch()
        try:
            asm.family(self.arch)
        except ValueError as e:   # as in runner.Oracle: an architecture without spellings is a skip
            raise Unavailable(f"{e} (set LEONIDA_HW_ARCH)") from None
        self.device = self._probe()
        if hasattr(self, "_probe_waves"):   # runner.Oracle has it once it runs kernels of both wave sizes
            self._probe_waves()

    def _build_launcher(self):
        with open(SOURCE, "rb") as f:
            tag = hashlib.sha1(f.read()).hexdigest()[:10]
        exe = os.path.join(tempfile.gettempdir(), f"leonida_hw_oracle_drm_{tag}")
        if not os.path.isfile(exe):
            flags = ["-I/usr/include/libdrm", "-ldrm_amdgpu"]
            pkg_config = shutil.which("pkg-config")
            if pkg_config:
                r = subprocess.run([pkg_config, "--cflags", "--libs", "libdrm_amdgpu"], capture_output=True, text=True)
                if r.returncode == 0 and r.stdout.strip():
                    flags = r.stdout.split()
            tmp = exe + f".{os.getpid()}"
            kind, cc = runner.find_cc()
            if kind != "gcc":   # only an override named cl gives anything else on Linux
                raise Unavailable(f"hw_oracle_drm.c needs gcc or a compiler that takes its options, not {cc}")
            r = subprocess.run([cc, "-O1", SOURCE, "-o", tmp, *flags], capture_output=True, text=True)
            if r.returncode:
                raise Unavailable("hw_oracle_drm.c did not build (it needs the libdrm headers): " + r.stderr.strip()[:300])
            os.replace(tmp, exe)
        return exe

    def _device_arch(self):
        """What the launcher says the GPU is; it prints its device line before it reads any code."""
        r = subprocess.run([self.exe, os.devnull, "4", self.node], input="", capture_output=True, text=True)
        head = r.stdout.splitlines()[0] if r.stdout else ""
        if not head.startswith("#"):
            raise Unavailable((r.stderr.strip() or "the launcher printed no device line")[:300])
        arch = next((field[5:] for field in head.split() if field.startswith("arch=")), "")
        return "gfx1030" if arch in ("", "gfx10xx") else arch   # every GFX10 chip takes the same encodings

    def _assemble(self, text, wave=32):
        s = os.path.join(self.workdir, "k.s")
        o = os.path.join(self.workdir, "k.o")
        code = os.path.join(self.workdir, "k.bin")
        with open(s, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        r = subprocess.run([self.clang, "--target=amdgcn-amd-amdhsa", f"-mcpu={self.arch}",
                            *(["-mwavefrontsize64"] if wave == 64 else []), "-c", s, "-o", o],
                           capture_output=True, text=True)
        if r.returncode:
            return None, r.stderr
        with open(o, "rb") as f:
            elf = f.read()
        binary, self._offsets = text_section(elf)
        self._registers = kernel_registers(elf)
        with open(code, "wb") as f:
            f.write(binary)
        return code, ""

    def _launch(self, code, tokens, words=4):
        r = subprocess.run([self.exe, code, str(words), self.node],
                           input="\n".join(launcher_tokens(tokens, self._offsets, self._registers)) + "\n",
                           capture_output=True, text=True)
        return r.returncode, r.stdout, r.stderr
