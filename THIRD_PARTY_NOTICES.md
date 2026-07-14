# Third-party notices

This project currently integrates or declares the following components. The Windows build places their complete license texts in `_internal/licenses`.

- dnglab/rawler — LGPL-2.1; https://github.com/dnglab/dnglab
- LibRaw — LGPL-2.1 or CDDL-1.0; https://www.libraw.org/
- rawpy — MIT; https://github.com/letmaik/rawpy
- NumPy — BSD-3-Clause; https://numpy.org/
- tifffile — BSD-3-Clause; https://github.com/cgohlke/tifffile
- psutil — BSD-3-Clause; https://github.com/giampaolo/psutil
- PySide6 / Qt — LGPL-3.0 and other Qt licensing options; https://doc.qt.io/qtforpython-6/licenses.html
- PyInstaller — GPL-2.0-or-later with a special exception; https://pyinstaller.org/
- PMRID code and pretrained checkpoint – Apache-2.0; https://github.com/MegEngine/PMRID
- ONNX Runtime GPU – MIT; https://github.com/microsoft/onnxruntime
- NVIDIA CUDA runtime, cuBLAS, cuDNN, cuFFT and nvJitLink redistributable DLLs – NVIDIA SDK/product license terms; https://docs.nvidia.com/cuda/eula/
- 7-Zip standalone command line 26.02 — LGPL-2.1-or-later with unRAR restriction; used only to create and verify release archives; https://www.7-zip.org/

The PMRID checkpoint is pinned by upstream commit and SHA-256 in `vendor/pmrid/UPSTREAM.md`. It is a general Bayer RAW baseline trained in a mobile-camera data domain, not an A7C II-specific model.
