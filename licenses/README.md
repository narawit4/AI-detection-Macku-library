# Release licensing checklist

The `licenses` directory is release material, not optional project
documentation. `manifest.json` records the exact package versions, provenance,
and SHA-256 hashes of every checked-in license, notice, and source archive.

For every executable release, publish beside `Jitter.exe`:

1. the repository-level `LICENSE` and `THIRD_PARTY_NOTICES.md`;
2. this entire `licenses` directory, unchanged; and
3. the complete corresponding Jitter source for that exact executable,
   including its build scripts and dependency pins.

The Jitter source alone does not satisfy the obligations for every bundled
component. In particular:

- `makcu` 2.3.1 is GPL-3.0. For an online release, offer the checked-in PyPI
  source archive with equivalent access from the same place as the binary as
  described by GPLv3 section 6(d). Other delivery methods must use an applicable
  option from GPLv3 section 6. If makcu is modified, replace or supplement the
  archive with the actual complete corresponding source and build scripts used.
- `pygame-ce` 2.5.6 is LGPL-2.1. Publish its checked-in source archive and any
  modifications. A distributor must also provide the materials and instructions
  needed for recipients to modify and relink/recombine the library when LGPL
  section 6 requires them. A source archive by itself is not a blanket claim of
  compliance for every one-file packaging method.
- NumPy, DXCam, pyserial, comtypes, and ONNX Runtime require preservation of
  their copyright and license notices. ONNX Runtime's exact
  `ThirdPartyNotices.txt` includes terms and source instructions for components
  bundled by Microsoft; those instructions remain applicable. pyserial's
  optional `hidapi` extra is not selected, and comtypes uses Windows system
  interfaces rather than redistributing COM, DXGI, or D3D system libraries.

For an online release, direct source archives next to the executable are the
preferred option. Equivalent-access links may be used only when the applicable
license permits them, must be clearly identified next to the binary, and must
remain available for the required period. A generic link to the Jitter
repository is not a substitute for exact third-party corresponding source.

Before publishing, audit the actual Nuitka output against the manifest. If the
packager includes a dependency or native library not recorded here, add its
exact notices and any required source/relinking materials before release. This
checklist records project release policy; distributors remain responsible for
compliance with the license texts themselves.
