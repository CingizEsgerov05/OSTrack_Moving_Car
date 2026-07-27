# 🚗 OSTrack Moving Car

Single-object video tracking using **OSTrack** (One-Stream Tracking, *ECCV 2022*) — runs end-to-end in Google Colab on a free T4 GPU. Give it a starting bounding box on frame 0 (out of the box, tracking a moving car), and it tracks the target through the rest of the video, drawing a bounding box + motion trail and exporting a web-ready H.264 MP4. The same pipeline works on any single-object video, drone footage included.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CingizEsgerov05/OSTrack_Moving_Car/blob/main/OSTrack_Tracking.ipynb)
![Python](https://img.shields.io/badge/python-3.10-blue)
![GPU](https://img.shields.io/badge/GPU-T4%20(Colab)-76B900)
![License](https://img.shields.io/badge/license-MIT-green)

## Demo

<!-- Replace with an actual GIF or screenshot: assets/demo.gif -->
`assets/demo.gif`

## How it works

The notebook is split into 8 self-contained cells, run top to bottom:

| # | Cell | What it does |
|---|------|---------------|
| 1 | GPU check | Verifies a CUDA GPU is available and prints VRAM/CUDA version |
| 2 | Setup | Clones [`botaoye/OSTrack`](https://github.com/botaoye/OSTrack), installs dependencies, and applies two compatibility patches (`torch._six` removal, `loader.py`) so the 2022 codebase runs on modern PyTorch |
| 3 | Weights | Downloads the pretrained `vitb_256_mae_ce_32x4_ep300` checkpoint from Google Drive via `gdown` |
| 4 | **User config** | The only cell you edit — video path, initial bounding box, colors, trail settings |
| 5 | Load tracker | Instantiates the OSTrack model with the downloaded checkpoint |
| 6 | Track | Runs the tracker frame-by-frame, drawing the bounding box + fading motion trail, and writes an output video |
| 7 | Encode | Re-encodes the raw output to H.264 (`ffmpeg`) so it plays natively in browsers |
| 8 | Download | Triggers a browser download of the final MP4 |

## Quick start

1. Click **Open in Colab** above (or upload the `.ipynb` to Colab manually).
2. `Runtime → Change runtime type → T4 GPU`.
3. Upload your source video to the Colab session (left panel → **Files**).
4. Edit the **⚙️ USER CONFIG** cell:
   ```python
   VIDEO_PATH  = '/content/your_video.mp4'
   INIT_BBOX   = [x, y, width, height]   # bounding box on frame 0
   ```
5. `Runtime → Run all` (or `Ctrl+F9`). The tracked video downloads automatically at the end.

## Tech stack

- [OSTrack](https://github.com/botaoye/OSTrack) — Ye et al., *"Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework"*, ECCV 2022
- PyTorch, `timm==0.5.4`, OpenCV, `gdown`, `ffmpeg`

## Notes & known limitations

- Single-object tracking only — one bounding box, one target.
- Requires manually specifying the initial bounding box in pixel coordinates; no auto-detection.
- The setup cell patches `torch._six` and `lib/train/data/loader.py`, since the original 2022 codebase predates PyTorch 2.x.
- Model weights (~300 MB) are pulled from Google Drive at runtime rather than committed to the repo.

## Repository structure

```
.
├── OSTrack_Tracking.ipynb   # Main Colab notebook
├── assets/                  # Demo GIF / sample frames
├── README.md
└── LICENSE
```

## Credits

Built on top of [OSTrack](https://github.com/botaoye/OSTrack) by Botao Ye et al. This repo only adds the Colab tracking pipeline (setup patches, config, drawing/trail utilities, H.264 export); all tracking is done by the original OSTrack model and weights.

## License

MIT — see [LICENSE](LICENSE). Note that OSTrack itself has its own license/terms; check the [upstream repo](https://github.com/botaoye/OSTrack) before commercial use.
