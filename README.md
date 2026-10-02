# CrowdGuard AI

**From crowd counting to crowd risk forecasting.**
A privacy-preserving early-warning system that turns a single ordinary CCTV camera into a crowd-safety tool. It runs locally on a laptop or edge device, with no cloud and no stored faces.

> Built for **CURIOUSPARC 2026 – State Innovation Challenge**
> Team: `SUPERIORS` · Members: `[SUYASH MOON, SHREERANG MULAY, KAUSHAL AGRAWAL, UTKARSH MESHRAM]`
> Demo / video: `[LINK]` · Contact: `[EMAIL]`

---

## Why this exists

Crowd crushes at temples, railway stations, festivals and exam centres often start with a build-up that is visible on camera but missed by the people watching it. Existing crowd-counting systems report only a head count or a heatmap, and that is not enough to judge safety:

- A count of **500 is harmless in a stadium and dangerous in a narrow lane**.
- A count says nothing about **movement**, **location**, or **how fast** the situation is getting worse.

CrowdGuard AI answers the questions operators actually need answered: *Where is it dangerous? Is it getting worse? How many minutes do I have? What should I do?*

## What it does

| Stage | What happens | Built on |
|---|---|---|
| **1. Observe** | Estimates the crowd density map from a single camera frame | CLIP-EBC (CLIP-based density model) |
| **2. Calibrate** | Converts the estimate into **people per m²** using monocular depth and a one-time ground-plane calibration | Depth Anything V2 + operator-marked ground points |
| **3. Understand** | Detects opposing flows and stop-and-go waves; compares inflow and outflow at operator-marked choke points (gates, staircases) | Optical flow (OpenCV) |
| **4. Act** | Forecasts the minutes left before density reaches a critical threshold, then sends a specific recommendation | Time-series forecast + alert APIs |

**Example alert**

> "Gate 2 is at 5.2 people/m² and rising: close entry and open Gate 4."

Alerts go to a live dashboard and by SMS / WhatsApp. Every incident is logged for later review.

## Architecture

```
CCTV frame
   │
   ├─► CLIP-EBC ──────────► density map ─┐
   │                                     ├─► people/m² per zone ─┐
   ├─► Depth Anything V2 ─► ground plane ┘                       │
   │                                                             ▼
   └─► Optical flow ─► opposing flows, stop-and-go,        Risk engine
                       choke-point inflow/outflow  ───────►  (thresholds + forecast)
                                                                 │
                                    ┌────────────────────────────┼───────────────┐
                                    ▼                            ▼               ▼
                               Dashboard                  SMS / WhatsApp     Incident log
```

## Privacy by design

- **Local processing only:** frames never leave the device; no cloud dependency.
- **No stored faces:** the pipeline works on density and motion, not identity. Only numeric metrics and event logs are persisted.
- **No identification or tracking of individuals:** analysis is aggregate (zones, flows, counts).

## Getting started

> **Note:** commands and paths below describe the intended layout. Adjust them to match the code in this repo.

### Requirements

- Python 3.10+
- A CCTV / IP camera stream (RTSP) or a video file
- A laptop or edge device (a CUDA GPU is recommended for real-time speed; CPU works at a lower frame rate)

### Install

```bash
git clone https://github.com/<org>/crowdguard-ai.git
cd crowdguard-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download model weights (CLIP-EBC and Depth Anything V2) into `weights/`. See [`weights/README.md`](weights/README.md) for links and checksums.

### One-time calibration

1. Open a frame from the camera in the calibration tool.
2. Mark **4 or more ground points** with known real-world distances (e.g., floor tiles, lane markings, a measured rectangle).
3. Mark **choke points** (gates, staircases) and name them (e.g., `Gate 2`, `Gate 4`).
4. Save the result to `configs/<site>.yaml`.

```bash
python -m crowdguard.calibrate --source <video_or_rtsp_url> --out configs/site.yaml
```

### Run

```bash
python -m crowdguard.run --source <video_or_rtsp_url> --config configs/site.yaml
```

Then open the dashboard at `http://localhost:8000`.

### Configure alerts

Copy `.env.example` to `.env` and set your SMS / WhatsApp provider credentials. Alert thresholds live in the site config:

```yaml
thresholds:
  watch: 3.0       # people/m², heightened attention
  warning: 4.0     # people/m², prepare to act
  critical: 5.0    # people/m², act now
forecast:
  horizon_minutes: 10
```

> Threshold values are **configurable starting points**, not universal constants. Safe density depends on the venue, crowd behaviour and flow, so tune them per site.

## Project structure

```
crowdguard-ai/
├── crowdguard/
│   ├── density/        # CLIP-EBC wrapper
│   ├── depth/          # Depth Anything V2 + ground-plane calibration
│   ├── motion/         # optical flow, opposing-flow & stop-and-go detection
│   ├── chokepoints/    # inflow/outflow comparison
│   ├── risk/           # thresholds + time-to-critical forecast
│   ├── alerts/         # dashboard feed, SMS / WhatsApp, incident log
│   └── dashboard/      # operator UI
├── configs/            # per-site calibration & thresholds
├── eval/               # metrics scripts and dataset loaders
├── weights/            # model weights (not committed)
├── docs/               # slides, diagrams
└── README.md
```

## Validation plan

We evaluate on **public crowd datasets and simulation**. Real stampede footage is scarce and ethically sensitive, so the forecast is *not* validated on it.

| Metric | What it measures | Data |
|---|---|---|
| **MAE** (counting error) | Accuracy of the head-count estimate | Public crowd-counting datasets (e.g., ShanghaiTech, UCF-QNRF) |
| **Density error** | Estimated people/m² vs. a measured floor area | Calibrated scenes with known geometry |
| **Forecast lead time** | Minutes of warning before the critical threshold is reached | Crowd simulation and recorded build-up sequences |

Results will be published in [`eval/RESULTS.md`](eval/RESULTS.md) as they are produced.

## Project status

| Component | Status |
|---|---|
| Density estimation (CLIP-EBC) | `[ ] planned  [ ] in progress  [ ] done` |
| Depth + ground-plane calibration | `[ ] planned  [ ] in progress  [ ] done` |
| Motion & choke-point analysis | `[ ] planned  [ ] in progress  [ ] done` |
| Risk forecast | `[ ] planned  [ ] in progress  [ ] done` |
| Dashboard & alerts | `[ ] planned  [ ] in progress  [ ] done` |
| Evaluation on public datasets | `[ ] planned  [ ] in progress  [ ] done` |

*(Update this table to reflect what actually works today.)*

## Roadmap

1. **MVP:** core CCTV pipeline, density estimation, depth calibration, motion analysis, choke-point monitoring, dashboard.
2. **Testing:** public datasets and controlled simulations; measure MAE, density error and forecast lead time.
3. **Pilot:** controlled real-world deployment, operator feedback, site-specific threshold calibration.
4. **Scale:** multiple cameras, multi-zone monitoring, edge optimisation, integration with existing CCTV systems.

## Limitations

- **Decision support, not a guarantee.** CrowdGuard AI gives operators earlier warning. It does not prevent stampedes by itself.
- Monocular depth and single-camera geometry introduce error, especially at steep camera angles or in occluded scenes.
- Density thresholds and forecasts need per-site tuning and, ideally, a pilot before operational reliance.
- Accuracy drops in very dense scenes, poor lighting, heavy occlusion or camera motion.
- Forecast quality has been validated on public data and simulation only, not on real stampede footage.

## Acknowledgements

- [CLIP-EBC](https://github.com/Yiming-M/CLIP-EBC): CLIP-based crowd counting with enhanced blockwise classification
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2): monocular depth estimation
- Public crowd datasets used for evaluation (see `eval/`)
- **CURIOUSPARC 2026** organisers and partners: S2P Robotics, Curio Tinker, InternAcademy, Nextech Minds, Tech Surya, CredenceE, EKATTA, Praket Foods, Oceansphere, Real Motors, Christ University, VIT Pune

Please check each model's and dataset's own licence before reuse or deployment.

## Licence

`[Choose a licence, e.g., MIT or Apache-2.0]`. See [`LICENSE`](LICENSE).

## Team

| Name | Role |
|---|---|
| `[MEMBER 1]` | AI/ML & Computer Vision |
| `[MEMBER 2]` | Backend & System Architecture |
| `[MEMBER 3]` | Frontend & Dashboard |
| `[MEMBER 4]` | Research, Validation & Deployment |
