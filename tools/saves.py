#!/usr/bin/env python3
"""saves.py: inspect, extract, rebuild and import the game's save containers (ASAV / JSAV / SLAYER1).

Standard library only. Run with `python -I tools/saves.py <command> ...`.
Reads files that are already plain on disk. It does no decryption and contains no keys; the packed SLAYER1 body
is reported, not decoded. Format notes: docs/saves.md.

Commands
  info    <file|dir>...          print headers (auto-detects ASAV, JSAV, SLAYER1 and save folders)
  extract <file> <outdir>        JSAV  -> index.json + blocks/NNN_<group>_<hash>.json + save.pretty.json
                                 ASAV  -> header.json + payload.bin (+ payload.json if it is text)
  pack    <extracted dir> <out>  rebuild a JSAV from an `extract` folder (blocks are taken verbatim)
                                 follows destination symlinks; destinations with multiple hard links are refused
  verify  <JSAV file>...         check each rebuild for byte identity, continuing after individual errors
  diff    <JSAV a> <JSAV b>      list blocks that were added, removed or changed between two saves
  import <save dir>... [--out DIR] [--map OLD=CONTAINER/BLOB]... [--with-packed] [--force] [--dry-run]
                                 copy the ASAV/JSAV payloads out of plain save folders into the on-disk layout
                                 of the runtime's offline XGameSave (runtime/shims/xgameruntime_gamesave.cpp):
                                 DIR/local-user/<container>/<blob>, names %-encoded the same way.
                                 DIR defaults to %LOCALAPPDATA%\\Leonida\\saves (the shim's LEONIDA_SAVE_DIR default).
                                 VERIFY: container/blob names are a guess until the executable shows them; by default
                                 container = slot name (from the ASAV header), blob = payload file name. Use --map.
"""
import json
import os
import stat
import struct
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402
from report_paths import same_report_path  # noqa: E402

# ---------------------------------------------------------------------------------------------- ASAV
ASAV_HDR = 0x170          # header size seen on every plain ASAV file
ASAV_TITLE = (0x28, 0x128)  # UTF-16LE display title, 128 code units
ASAV_NAME = (0x128, 0x160)  # ASCII slot name, 56 bytes


def ts(v):
    try:
        return datetime.fromtimestamp(v, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    except (OverflowError, OSError, ValueError):
        return None


def parse_asav(d):
    if d[:4] != b'ASAV':
        raise ValueError('not ASAV')
    # The fixed fields below are read at absolute offsets, so a short file must be rejected here instead of
    # letting struct.unpack_from raise a struct.error that says nothing about the save (the JSAV parser in #923).
    if len(d) < ASAV_HDR:
        raise ValueError(f'header is only {len(d)} bytes, an ASAV header is {ASAV_HDR}')
    f = struct.unpack_from('<IIIIIII', d, 4)
    h = {
        'magic': 'ASAV',
        'version': f[0],            # 1
        'kind': f[1],               # 1 = story (wraps compressed/encrypted JSAV), 2 = settings, 3 = player
        'unk_0c': f[2],             # 1 on plain files, 3 on the SLAYER1-wrapped one
        'field_10': f[3],           # 0 on plain files; 0x180 (= header size?) on the wrapped one
        'payload_offset': f[4],     # 0x170
        'field_18': f[5],           # 0 on plain files; looks like stored size on the wrapped one
        'field_1c': f[6],           # total file size on plain files; unpacked size on the wrapped one
        'timestamp': struct.unpack_from('<Q', d, 0x20)[0],
        'title': d[ASAV_TITLE[0]:ASAV_TITLE[1]].decode('utf-16le', 'replace').split('\0', 1)[0],
        'slot_name': d[ASAV_NAME[0]:ASAV_NAME[1]].split(b'\0', 1)[0].decode('ascii', 'replace'),
        'checksum_160': '%08x' % struct.unpack_from('<I', d, 0x160)[0],   # algorithm unknown
        'field_164': struct.unpack_from('<I', d, 0x164)[0],
        'size_168': struct.unpack_from('<Q', d, 0x168)[0],
        'file_size': len(d),
    }
    h['timestamp_utc'] = ts(h['timestamp'])
    if h['payload_offset'] > len(d):
        raise ValueError(f"payload offset {h['payload_offset']:#x} is past the end of a {len(d)}-byte file")
    return h


def asav_payload(d, h):
    end = h['field_1c'] if h['unk_0c'] == 1 and h['field_1c'] <= len(d) else len(d)
    return d[h['payload_offset']:end]


PROFILE_PAYLOAD_SIZE = 1644
PROFILE_HEADER_SIZE = 112


def parse_profile(pl):
    """Decode an ASAV profile record (kind 3) payload.

    The payload has a fixed size of 1,644 bytes:
    - 0x00..0x70 (112 bytes): active profile header starting with version=10, kind=3.
    - 0x70..0x66C (1,532 bytes): zero padding / reserved expansion buffer.
    """
    if len(pl) < 8:
        raise ValueError(f'profile payload too short ({len(pl)} bytes, expected at least 8)')
    ver, kind = struct.unpack_from('<II', pl, 0)
    if ver != 10:
        raise ValueError(f'unexpected profile version {ver} (expected 10)')
    if kind != 3:
        raise ValueError(f'unexpected profile record kind {kind} (expected 3)')

    active_len = min(len(pl), PROFILE_HEADER_SIZE)
    dwords = list(struct.unpack_from(f'<{active_len // 4}I', pl, 0))
    zero_tail = max(0, len(pl) - PROFILE_HEADER_SIZE)
    zero_clean = not any(pl[PROFILE_HEADER_SIZE:]) if len(pl) > PROFILE_HEADER_SIZE else True

    rec = {
        'version': ver,
        'kind': kind,
        'size': len(pl),
        'active_bytes': active_len,
        'zero_padding_bytes': zero_tail,
        'zero_padding_clean': zero_clean,
        'dwords': dwords,
    }
    if active_len >= 16:
        rec['subversion'] = dwords[2]
        rec['flags'] = dwords[3]
    if active_len >= 24:
        rec['account_id'] = struct.unpack_from('<Q', pl, 16)[0]
    return rec


def build_profile(dwords=None, account_id=0, total_size=PROFILE_PAYLOAD_SIZE):
    """Build a synthetic profile payload (kind 3, 1644 bytes)."""
    buf = bytearray(total_size)
    struct.pack_into('<II', buf, 0, 10, 3)
    if dwords:
        for i, val in enumerate(dwords[:PROFILE_HEADER_SIZE // 4]):
            struct.pack_into('<I', buf, i * 4, val)
        struct.pack_into('<II', buf, 0, 10, 3)
    if account_id:
        struct.pack_into('<Q', buf, 16, account_id)
    return bytes(buf)


# ---------------------------------------------------------------------------------------------- JSAV
JSAV_HDR = 16
JSAV_ENTRY = struct.Struct('<IIIII')  # hash, flags (1), reserved (0), absolute offset, size
JSAV_SEP = b',\r\n'


def parse_jsav(d):
    if d[:4] != b'JSAV':
        raise ValueError('not JSAV')
    if len(d) < JSAV_HDR:
        raise ValueError('JSAV header is only %d bytes' % len(d))
    ver, n_code, n_script = struct.unpack_from('<III', d, 4)
    n = n_code + n_script
    table_end = JSAV_HDR + n * JSAV_ENTRY.size
    if table_end > len(d):
        raise ValueError('JSAV header declares %d entries but the file has %d bytes' % (n, len(d)))
    entries = []
    for i in range(n):
        hsh, flags, rsv, off, size = JSAV_ENTRY.unpack_from(d, JSAV_HDR + i * JSAV_ENTRY.size)
        entries.append({'index': i, 'group': 'code' if i < n_code else 'script',
                        'hash': '%08x' % hsh, 'flags': flags, 'reserved': rsv, 'offset': off, 'size': size})
    try:
        body_off = d.index(b'{', table_end)
    except ValueError:
        raise ValueError('JSAV body marker not found after the entry table') from None
    try:
        body_text = d[body_off:].rstrip(b'\0').decode('utf-8')
        _, json_end = json.JSONDecoder().raw_decode(body_text)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f'JSAV body is not valid UTF-8 JSON: {exc}') from None
    # raw_decode returns a character offset, while the entry table uses byte offsets.
    # The last nonzero byte is not the JSON boundary: corrupt data can follow the document's padding.
    body_end = body_off + len(body_text[:json_end].encode('utf-8'))
    for entry in entries:
        start, end = entry['offset'], entry['offset'] + entry['size']
        if start < body_off or end > body_end:
            # Python slices silently truncate invalid ranges; never export or compare those as real blocks.
            raise ValueError(f"JSAV entry {entry['index']} range {start:#x}..{end:#x} is outside "
                             f"the JSON body {body_off:#x}..{body_end:#x}")
    if body_text[json_end:].strip(' \t\r\n'):
        raise ValueError('JSAV has non-padding data after the JSON body')
    return {'magic': 'JSAV', 'version': ver, 'code_blocks': n_code, 'script_blocks': n_script,
            'table_end': table_end, 'body_offset': body_off, 'body_end': body_end,
            'file_size': len(d)}, entries


def jsav_blocks(d, entries):
    return [d[e['offset']:e['offset'] + e['size']] for e in entries]


def align(n, a):
    return (n + a - 1) // a * a


def build_jsav(version, n_code, hashes, flags, blocks):
    """Rebuild the exact framing seen in retail saves."""
    n = len(blocks)
    body_off = align(JSAV_HDR + n * JSAV_ENTRY.size, 16)
    # Retail framing: '{"CodeSaveBlocks":[\r\n' b0 ',\r\n' b1 ... '],\r\n"ScriptSaveBlocks":[\r\n' ... ']\r\n}'
    # (the closing ']' follows the last block directly, no newline before it).
    body = bytearray(b'{"CodeSaveBlocks":[\r\n')
    offs = []
    for i, b in enumerate(blocks):
        if i == n_code:
            if n_code:
                body[-len(JSAV_SEP):] = b''
            body += b'],\r\n"ScriptSaveBlocks":[\r\n'
        offs.append(body_off + len(body))
        body += b + JSAV_SEP
    if blocks:
        body[-len(JSAV_SEP):] = b''
    if n_code == n:
        body += b'],\r\n"ScriptSaveBlocks":[\r\n'
    body += b']\r\n}'
    out = bytearray(struct.pack('<4sIII', b'JSAV', version, n_code, n - n_code))
    for h, f, o, b in zip(hashes, flags, offs, blocks):
        out += JSAV_ENTRY.pack(h, f, 0, o, len(b))
    out += b'\0' * (body_off - len(out))
    out += body
    out += b'\0' * (align(len(out), 16) - len(out))
    return bytes(out)


# ---------------------------------------------------------------------------------------------- SLAYER1
SLAYER1_HDR = 16           # "SLAYER1\0" + u32 version + u32 name length


def parse_slayer(d, path=None):
    if d[:8] != b'SLAYER1\0':
        raise ValueError('not SLAYER1')
    where = f' in {path}' if path else ''
    # The fixed preamble holds the magic, version and name length; a short file must be rejected here instead
    # of letting struct.unpack_from raise a struct.error that says nothing about the save (see parse_asav/#923).
    if len(d) < SLAYER1_HDR:
        raise ValueError(f'SLAYER1 header is only {len(d)} bytes{where}, a SLAYER1 header is at least '
                         f'{SLAYER1_HDR}')
    ver, nlen = struct.unpack_from('<II', d, 8)
    # The name and the 8-byte payload-size field that follows it must fit before either is read.
    size_off = SLAYER1_HDR + nlen
    if size_off + 8 > len(d):
        raise ValueError(f'SLAYER1{where} declares a {nlen}-byte name; with the 8-byte payload size the header '
                         f'needs {SLAYER1_HDR + nlen + 8} bytes, but the file has only {len(d)}')
    name = d[SLAYER1_HDR:SLAYER1_HDR + nlen].decode('ascii', 'replace')
    size = struct.unpack_from('<Q', d, size_off)[0]
    inner = size_off + 8
    return {'magic': 'SLAYER1', 'version': ver, 'name': name, 'payload_size': size,
            'payload_offset': inner, 'file_size': len(d)}


# ---------------------------------------------------------------------------------------------- helpers
def kind_of(d):
    if d[:4] == b'ASAV':
        return 'ASAV'
    if d[:4] == b'JSAV':
        return 'JSAV'
    if d[:8] == b'SLAYER1\0':
        return 'SLAYER1'
    return None


def read(p, n=-1):
    with open(p, 'rb') as f:
        return f.read(n)


def info_file(p, indent=''):
    d = read(p)
    k = kind_of(d)
    print(f'{indent}{p}: {k or "unknown"} ({len(d)} bytes)')
    if k == 'ASAV':
        h = parse_asav(d)
        for key, v in h.items():
            print(f'{indent}  {key:15} {v}')
        pl = asav_payload(d, h)
        print(f'{indent}  payload         {len(pl)} bytes, starts {pl[:16]!r}')
        if h['kind'] == 3 and len(pl) >= 8:
            try:
                prof = parse_profile(pl)
                print(f'{indent}  profile record  version={prof["version"]} kind={prof["kind"]} '
                      f'active={prof["active_bytes"]}B zero_pad={prof["zero_padding_bytes"]}B '
                      f'(clean={prof["zero_padding_clean"]})')
            except ValueError as ex:
                print(f'{indent}  profile record  failed to decode: {ex}')
    elif k == 'JSAV':
        h, es = parse_jsav(d)
        for key, v in h.items():
            print(f'{indent}  {key:15} {v}')
        groups = {}
        for e, b in zip(es, jsav_blocks(d, es)):
            g = b[2:15].decode('ascii', 'replace').strip('"')
            groups.setdefault((e['group'], g), []).append(e['size'])
        for (g, w), sizes in groups.items():
            print(f'{indent}  {g:6} wrapper {w}: {len(sizes)} blocks, {sum(sizes)} bytes, largest {max(sizes)}')
    elif k == 'SLAYER1':
        h = parse_slayer(d, p)
        for key, v in h.items():
            print(f'{indent}  {key:15} {v}')
        inner = d[h['payload_offset']:]
        if inner[:4] == b'ASAV':
            ih = parse_asav(inner)
            print(f'{indent}  inner ASAV: kind={ih["kind"]} title={ih["title"]!r} slot={ih["slot_name"]!r} '
                  f'time={ih["timestamp_utc"]} f10={ih["field_10"]:#x} f18={ih["field_18"]} f1c={ih["field_1c"]}')
            print(f'{indent}  inner body is high-entropy (compressed and/or encrypted); not decoded')


def info(paths):
    for p in paths:
        if os.path.isdir(p):
            cj = os.path.join(p, 'chives.json')
            if os.path.exists(cj):
                meta = json.loads(read(cj))
                print(f'{p}: save folder, dir {meta.get("save_directory")}')
            for name in sorted(os.listdir(p)):
                fp = os.path.join(p, name)
                if os.path.isfile(fp) and kind_of(read(fp, 8)):
                    info_file(fp, '  ')
        else:
            info_file(p)


def extract(p, out):
    d = read(p)
    k = kind_of(d)
    os.makedirs(out, exist_ok=True)
    if k == 'JSAV':
        h, es = parse_jsav(d)
        bd = os.path.join(out, 'blocks')
        os.makedirs(bd, exist_ok=True)
        blocks = jsav_blocks(d, es)
        for e, b in zip(es, blocks):
            e['file'] = f'blocks/{e["index"]:03d}_{e["group"]}_{e["hash"]}.json'
            with open(os.path.join(out, e['file']), 'wb') as f:
                f.write(b)
        with open(os.path.join(out, 'index.json'), 'w', encoding='utf-8', newline="\n") as f:
            json.dump({'header': h, 'entries': es}, f, indent=1)
        body = d[h['body_offset']:h['body_end']].decode('utf-8')
        with open(os.path.join(out, 'save.pretty.json'), 'w', encoding='utf-8', newline="\n") as f:
            json.dump(json.loads(body), f, indent=1)
        print(f'{len(es)} blocks -> {out}')
    elif k == 'ASAV':
        h = parse_asav(d)
        pl = asav_payload(d, h)
        with open(os.path.join(out, 'header.json'), 'w', encoding='utf-8', newline="\n") as f:
            json.dump(h, f, indent=1)
        with open(os.path.join(out, 'header.bin'), 'wb') as f:
            f.write(d[:h['payload_offset']])
        with open(os.path.join(out, 'payload.bin'), 'wb') as f:
            f.write(pl)
        if pl[:1] == b'{':
            try:
                obj = json.loads(pl.rstrip(b'\0').decode('utf-8'))
                with open(os.path.join(out, 'payload.json'), 'w', encoding='utf-8', newline="\n") as f:
                    json.dump(obj, f, indent=1)
            except ValueError as ex:
                print('payload looks like text but did not parse as JSON:', ex)
        elif h['kind'] == 3 and len(pl) >= 8:
            try:
                prof = parse_profile(pl)
                with open(os.path.join(out, 'profile.json'), 'w', encoding='utf-8', newline="\n") as f:
                    json.dump(prof, f, indent=1)
            except ValueError as ex:
                print('profile payload did not parse:', ex)
        print(f'ASAV kind {h["kind"]} ({h["slot_name"]}) -> {out}')
    else:
        sys.exit(f'{p}: extract supports JSAV and ASAV')


def pack_from_dir(src, refuse=None):
    """The JSAV bytes for an extracted folder. `refuse` is the --out path: a source file it names is refused."""
    index_path = os.path.join(src, 'index.json')

    def invalid(field, expected):
        raise ValueError(f'{index_path}: {field} must be {expected}')

    def source(name):
        path = os.path.join(src, name)
        if refuse is not None and same_report_path(path, refuse):
            raise ValueError(f'{path}: pack reads this file, and --out {refuse} would replace it')
        return read(path)

    text = source('index.json')
    try:
        idx = json.loads(text)
    except (ValueError, RecursionError) as exc:
        raise ValueError(f'{index_path}: invalid JSON ({exc})') from None
    if not isinstance(idx, dict):
        invalid('index', 'an object')
    h, es = idx.get('header'), idx.get('entries')
    if not isinstance(h, dict):
        invalid('header', 'an object')
    if not isinstance(es, list):
        invalid('entries', 'an array')
    for field in ('version', 'code_blocks'):
        value = h.get(field)
        if type(value) is not int or not 0 <= value <= 0xFFFFFFFF:
            invalid('header.' + field, 'a uint32 integer')
    if h['code_blocks'] > len(es):
        invalid('header.code_blocks', f'between 0 and the entry count ({len(es)})')
    hashes = []
    for i, entry in enumerate(es):
        prefix = f'entries[{i}]'
        if not isinstance(entry, dict):
            invalid(prefix, 'an object')
        if not isinstance(entry.get('file'), str) or not entry['file']:
            invalid(prefix + '.file', 'a nonempty path string')
        flags = entry.get('flags')
        if type(flags) is not int or not 0 <= flags <= 0xFFFFFFFF:
            invalid(prefix + '.flags', 'a uint32 integer')
        if not isinstance(entry.get('hash'), str):
            invalid(prefix + '.hash', 'a hexadecimal uint32 string')
        try:
            hashed = int(entry['hash'], 16)
        except ValueError:
            invalid(prefix + '.hash', 'a hexadecimal uint32 string')
        if not 0 <= hashed <= 0xFFFFFFFF:
            invalid(prefix + '.hash', 'a hexadecimal uint32 string')
        hashes.append(hashed)
    blocks = [source(e['file']) for e in es]
    return build_jsav(h['version'], h['code_blocks'], hashes, [e['flags'] for e in es], blocks)


def pack(src, out):
    # Publish at the symlink's target, as open(out, 'wb') did, without replacing the link itself.
    destination = os.path.realpath(out)
    try:
        existing = os.stat(destination)
    except FileNotFoundError:
        pass
    else:
        # Replacing one hard-link name cannot update the other aliases atomically.
        if stat.S_ISREG(existing.st_mode) and existing.st_nlink > 1:
            raise ValueError(f'{out}: destination has multiple hard links; choose an unlinked output path')
    data = pack_from_dir(src, destination)
    parent = os.path.dirname(destination)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile('wb', dir=parent, prefix='.saves-pack-', suffix='.tmp',
                                         delete=False) as f:
            tmp = f.name
            f.write(data)
        os.chmod(tmp, published_mode(destination))
        os.replace(tmp, destination)
    except BaseException:
        if tmp is not None:
            try:
                # A failed replacement may leave a read-only staging file on Windows.
                os.chmod(tmp, 0o600)
                os.remove(tmp)
            except OSError:
                pass
        raise
    print(f'wrote {out} ({len(data)} bytes)')


def verify(p):
    d = read(p)
    h, es = parse_jsav(d)
    blocks = jsav_blocks(d, es)
    for index, b in enumerate(blocks):
        try:
            json.loads(b)             # every block must be valid JSON on its own
        except RecursionError as e:
            raise ValueError(f'JSAV block {index} exceeds the JSON nesting limit') from e
    rebuilt = build_jsav(h['version'], h['code_blocks'], [int(e['hash'], 16) for e in es],
                         [e['flags'] for e in es], blocks)
    if rebuilt == d:
        print(f'{p}: OK, {len(es)} blocks parse and the rebuild is byte-identical')
        return True
    n = next((i for i, (a, b) in enumerate(zip(rebuilt, d)) if a != b), min(len(rebuilt), len(d)))
    print(f'{p}: MISMATCH at {n:#x} (rebuilt {len(rebuilt)} vs original {len(d)} bytes)')
    return False


def diff(pa, pb):
    da, db = read(pa), read(pb)
    (_, ea), (_, eb) = parse_jsav(da), parse_jsav(db)
    # A save can hold several blocks with the same (group, hash): keep them in order instead of letting
    # later ones overwrite earlier ones in a dict, which silently dropped them from the comparison.
    ma, mb = {}, {}
    for entries, data, table in ((ea, da, ma), (eb, db, mb)):
        for entry, block in zip(entries, jsav_blocks(data, entries)):
            table.setdefault((entry['group'], entry['hash']), []).append(block)
    for k in sorted(ma.keys() - mb.keys()):
        for block in ma[k]:
            print('only in A:', *k, len(block))
    for k in sorted(mb.keys() - ma.keys()):
        for block in mb[k]:
            print('only in B:', *k, len(block))
    same = 0
    for k in sorted(ma.keys() & mb.keys()):
        left, right = ma[k], mb[k]
        # Match identical blocks by content first: with duplicate keys, an unchanged block that moved
        # position should be reported as a removal/addition of the others, not as a false "changed".
        pending_right, matched_left = list(right), [False] * len(left)
        for i, block in enumerate(left):
            if block in pending_right:
                matched_left[i] = True
                pending_right.remove(block)
                same += 1
        left = [block for i, block in enumerate(left) if not matched_left[i]]
        for i in range(max(len(left), len(pending_right))):
            a = left[i] if i < len(left) else None
            b = pending_right[i] if i < len(pending_right) else None
            if a is None or b is None:
                label, block = ('only in B:', b) if a is None else ('only in A:', a)
                print(label, *k, len(block))
                continue
            if a == b:
                same += 1
                continue
            n = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            print(f'changed: {k[0]} {k[1]} {len(a)} -> {len(b)} bytes, first difference near: '
                  f'A {a[max(0, n - 40):n + 40]!r}  B {b[max(0, n - 40):n + 40]!r}')
    print(f'{same} blocks identical')


# ---------------------------------------------------------------------------------------------- import
def xgs_encode(name):
    """Same rule as encode_name() in runtime/shims/xgameruntime_gamesave.cpp: keep
    [A-Za-z0-9_-] and '.' except as the first character; everything else becomes %XX (uppercase hex) per UTF-8 byte."""
    out = []
    for c in name.encode('utf-8'):
        ch = chr(c)
        if (ch.isascii() and ch.isalnum()) or ch in '_-' or (ch == '.' and out):
            out.append(ch)
        else:
            out.append('%%%02X' % c)
    return ''.join(out)


def save_slot(folder):
    """Return (folder label, payload filename, slot name, title) for one save folder."""
    meta_path = os.path.join(folder, 'chives.json')
    meta = json.loads(read(meta_path)) if os.path.isfile(meta_path) else {}
    found = []
    for name in sorted(os.listdir(folder)):
        fp = os.path.join(folder, name)
        if os.path.isfile(fp) and name not in ('chives.json', 'chives.dat') and kind_of(read(fp, 8)) in ('ASAV', 'JSAV'):
            found.append(name)
    if not found:
        raise SystemExit(f'{folder}: no ASAV/JSAV payload found')
    if len(found) > 1:
        # One save per folder: refuse to guess (e.g. "slot1" next to an older "slot1.bak").
        raise SystemExit(f'{folder}: {len(found)} ASAV/JSAV payloads, expected one: {", ".join(found)}; '
                         'move the extra files out or import a folder per save')
    payload = found[0]
    slot, title = payload, None
    d = read(os.path.join(folder, payload))
    if kind_of(d) == 'ASAV':
        h = parse_asav(d)
        slot, title = h['slot_name'] or payload, h['title']
    packed = os.path.join(folder, 'chives.dat')
    if os.path.isfile(packed):
        p = read(packed)
        if kind_of(p) == 'SLAYER1':
            inner = p[parse_slayer(p, packed)['payload_offset']:]
            if inner[:4] == b'ASAV':
                ih = parse_asav(inner)
                slot, title = ih['slot_name'] or slot, ih['title']
    return meta.get('save_directory') or os.path.basename(os.path.normpath(folder)), payload, slot, title


CONTAINER_PATH_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_/')
CONTAINER_FILE_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-')


def valid_container_name(name):
    """XGameSaveCreateContainer's documented rule, as runtime/shims/xgameruntime_gamesave.cpp checks it: at most 256
    characters; a path part (up to the last '/') of [A-Za-z0-9_/], then a non-empty file part of [A-Za-z0-9_.-] that
    does not end in '.' nor hold '..'. The runtime lists any folder, but the game can only open one by a valid name."""
    if not name or len(name) > 256:
        return False
    path, _, file = name.rpartition('/')
    if path and not set(path + '/') <= CONTAINER_PATH_CHARS:
        return False
    return bool(file) and not file.endswith('.') and '..' not in file and set(file) <= CONTAINER_FILE_CHARS


def import_saves(args):
    flags = {'--out', '--map', '--with-packed', '--force', '--dry-run'}
    for option, following in zip(args, args[1:]):
        if option in ('--out', '--map') and following in flags:
            raise ValueError(f'import: {option} needs a value before {following}')

    import hashlib
    out, maps, with_packed, force, dry = None, {}, False, False, False
    folders = []
    it = iter(args)

    def value(flag):
        v = next(it, None)
        if v is None:
            raise ValueError(f'import: {flag} needs a value')
        return v
    for a in it:
        if a == '--out':
            out = value(a)
        elif a == '--map':
            spec = value(a)
            if '=' not in spec:
                raise ValueError(f'import: --map needs OLD=CONTAINER/BLOB, got {spec!r}')
            k, v = spec.split('=', 1)
            c, _, b = v.partition('/')
            maps[k] = (c, b or None)
        elif a == '--with-packed':
            with_packed = True
        elif a == '--force':
            force = True
        elif a == '--dry-run':
            dry = True
        else:
            folders.append(a)
    if not folders:
        raise SystemExit('import: give one or more save folders')
    if out is None:
        out = os.path.join(os.environ.get('LOCALAPPDATA', '.'), 'Leonida', 'saves')
    root = os.path.join(out, 'local-user')
    manifest = {'note': 'VERIFY: container/blob names are a guess until the executable shows them',
                'layout': '<LEONIDA_SAVE_DIR>/local-user/<container>/<blob>', 'entries': []}
    plan = []
    for folder in folders:
        sd, payload, slot, title = save_slot(folder)
        container, blob = slot, payload
        for key in (sd, payload, slot):
            if key in maps:
                container = maps[key][0] or container
                blob = maps[key][1] or blob
        files = [(payload, blob)]
        if with_packed and os.path.isfile(os.path.join(folder, 'chives.dat')):
            files.append(('chives.dat', 'chives.dat'))
        for src_name, blob_name in files:
            src = os.path.join(folder, src_name)
            dst = os.path.join(root, xgs_encode(container), xgs_encode(blob_name))
            plan.append((src, dst))
            data = read(src)
            manifest['entries'].append({'folder': sd, 'source': src_name, 'title': title, 'container': container,
                                        'blob': blob_name, 'size': len(data),
                                        'sha256': hashlib.sha256(data).hexdigest()})
            print(f'{sd:34} {src_name:24} -> container {container!r:12} blob {blob_name!r:24} {len(data):>9} bytes'
                  f'  ({title})')
            if not valid_container_name(container):
                print(f'warning: container {container!r} breaks XGameSaveCreateContainer\'s naming rule, so the game '
                      f'cannot open it by that name; give it another with --map {sd}=NAME', file=sys.stderr)
    dsts = [d.lower() for _, d in plan]
    if len(set(dsts)) != len(dsts):
        raise SystemExit('import: two payloads map to the same container/blob (names are case-insensitive on '
                         'Windows); use --map to separate them')
    if dry:
        print('dry run: nothing written')
        return
    for src, dst in plan:
        if os.path.exists(dst) and not force:
            raise SystemExit(f'{dst} exists; pass --force to overwrite')
    for src, dst in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = os.path.join(os.path.dirname(dst), '~' + os.path.basename(dst))   # the shim's temp-file convention
        with open(tmp, 'wb') as f:
            f.write(read(src))
        os.replace(tmp, dst)
    with open(os.path.join(out, 'import-manifest.json'), 'w', encoding='utf-8', newline="\n") as f:
        json.dump(manifest, f, indent=1)
    print(f'wrote {len(plan)} blobs under {root}')


def main(argv):
    if len(argv) < 2 or argv[1] in ('-h', '--help'):
        print(__doc__)
        return 0
    cmd, args = argv[1], argv[2:]
    try:
        if cmd == 'info' and args:
            info(args)
        elif cmd == 'extract' and len(args) == 2:
            extract(*args)
        elif cmd == 'pack' and len(args) == 2:
            pack(*args)
        elif cmd == 'verify' and args:
            status = 0
            for path in args:
                try:
                    if not verify(path):
                        status = max(status, 1)
                except (ValueError, struct.error) as e:
                    print(f'ERROR: {e} ({path})', file=sys.stderr)
                    status = 2
                except OSError as e:
                    print(f'ERROR: {e.strerror or e} ({path})', file=sys.stderr)
                    status = 2
            return status
        elif cmd == 'diff' and len(args) == 2:
            diff(*args)
        elif cmd == 'import':
            import_saves(args)
        else:
            print(__doc__)
            return 2
    except (ValueError, struct.error, OSError) as e:
        # A malformed save must report one clear line, not a traceback.
        print('ERROR: %s' % e, file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main(sys.argv))
