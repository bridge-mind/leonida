# XMA decoder third-party notice

The Rust package `xma` is consumed as an external Cargo dependency by `runtime/xma_decoder`. Cargo pins the source to commit `27122f7dfcceb9a899555362ee0dde2d530e033c` in [omonkeyz/xbox-emulator](https://github.com/omonkeyz/xbox-emulator/tree/27122f7dfcceb9a899555362ee0dde2d530e033c/crates/compat/xma). The package declares LGPL-2.1-or-later. Its license text is retained in `COPYING.LGPLv2.1`.

The upstream NOTICE identifies this package as a Rust port of FFmpeg's WMA Pro/XMA decoder from FFmpeg commit `76fa18983e3849595fdc332bf9fa583d5fe7a147`. Original notices:

> Wmapro compatible decoder
> Copyright (c) 2007 Baptiste Coudurier, Benjamin Larsson, Ulion
> Copyright (c) 2008 - 2011 Sascha Sommer, Benjamin Larsson
> This file is part of FFmpeg and is licensed under the GNU Lesser General Public License, version 2.1 or later.

Leonida's C ABI wrapper is independently written and licensed under GPL-3.0-or-later. No upstream implementation source is copied or vendored.
