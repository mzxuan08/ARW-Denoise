# PMRID upstream record

- Project: Practical Mobile Raw Image Denoising (PMRID)
- Repository: https://github.com/MegEngine/PMRID
- Pinned commit: `8ebb9e8e96559881dee957f34243933c5beb77dd`
- License: Apache-2.0 (`LICENSE` in this directory)
- Original checkpoint: `models/torch_pretrained.ckp`
- Original checkpoint SHA-256: `9361614f3514d27351d81909f2215c0fdc38619c0288d936b7266485ac106c14`
- Original network definition: `models/net_torch.py`
- Original network definition SHA-256: `7f90bac455c3f1f29d948b8812e49664b43204be9ee319868120175116a670bd`

The upstream checkpoint is a general Bayer RAW baseline trained for the PMRID mobile-camera data domain. It is not an A7C II-specific model. Any converted ONNX artifact must record both the original checkpoint hash and its own hash in the model manifest.

