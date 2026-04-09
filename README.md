# NL Security Camera

![Status](https://img.shields.io/badge/status-in%20development-orange?style=flat-square)

A natural-language-controlled security camera system. Type plain English alert conditions ("notify me if someone leaves a bag unattended") and the pipeline compiles them into structured conditions, checks live camera frames against them using a Vision-Language Model, and fires alerts with snapshots when triggered.

---

## Project Structure

```
nl-security-cam/
├── backend/
│   ├── main.py          # FastAPI app + WebSocket stream
│   ├── camera.py        # OpenCV frame capture
│   ├── vlm.py           # LLaVA inference wrapper
│   ├── compiler.py      # NL → JSON condition compiler
│   └── requirements.txt
└── frontend/
    └── index.html       # Dashboard UI
```

---

## Prerequisites

### 1. Python 3.10+

```bash
python --version  # should be 3.10 or higher
```

### 2. Ollama

Ollama runs the LLaVA model locally. Install from [ollama.com](https://ollama.com), then pull the model:

```bash
ollama pull llava:7b
```

> **GPU note:** LLaVA-7B needs ~6GB VRAM. If you only have CPU, inference will be slow (~10–20s per frame). See [Faster Alternatives](#faster-alternatives) below.

---

## Installation

```bash
git clone https://github.com/yourname/nl-security-cam
cd nl-security-cam/backend

pip install -r requirements.txt
```

---

## Running

### Start the backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Open the dashboard

Open `frontend/index.html` directly in your browser, or serve it:

```bash
cd frontend
python -m http.server 5500
# then visit http://localhost:5500
```

---

## Usage

1. The dashboard connects to the WebSocket stream automatically, you should see your camera feed appear within a few seconds.
2. Type an alert condition in the input bar, e.g.:
   - `"alert me if a person enters the frame"`
   - `"notify when a bag is left unattended near the door"`
   - `"trigger if more than two people are present"`
3. Click **SET ALERT**. The compiler sends your query to LLaVA, which returns a structured JSON condition shown in the sidebar.
4. The pipeline now checks every sampled frame against that condition. When triggered, the feed flashes red, a banner appears with the reason, and a snapshot is logged in the alert sidebar.

---

## How It Works

```
Camera (OpenCV)
      │
      │  JPEG frame @ ~1fps
      ▼
  compiler.py          ← one-time: NL query → JSON condition via LLM
      │
      ▼
    vlm.py             ← per-frame: "does this condition apply?" → yes/no + reason
      │
      ▼
  main.py (FastAPI)    ← WebSocket: pushes frame + alert payload to browser
      │
      ▼
  index.html           ← renders feed, overlays alert, logs snapshots
```

The NL query is compiled once when you hit "SET ALERT", not on every frame. The per-frame VLM call uses a tightly constrained prompt that asks only for `TRIGGERED: yes/no` + a one sentence reason, keeping inference as fast as possible.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/compile` | POST | Compiles a NL query into a JSON condition. Body: `{"query": "..."}` |
| `/alerts` | GET | Returns the last 50 triggered alerts |
| `/ws/stream` | WebSocket | Streams `{frame: base64, alert: object\|null}` at ~1fps |

---

## Configuration

Key constants you may want to adjust, all in `main.py` and `vlm.py`:

| Variable | Default | Description |
|---|---|---|
| `Camera(source=0)` | `0` | Camera index. Change to `1`, `2`, etc. for external cameras, or pass an RTSP URL string for IP cameras |
| `asyncio.sleep(1.0)` | `1.0s` | Frame sampling interval. Lower = more responsive, higher VLM load |
| `model` in `VLMChecker` | `"llava:7b"` | The Ollama model used for frame checking |
| `model` in `QueryCompiler` | `"llava:7b"` | The Ollama model used for query compilation |
| `alert_log[-50:]` | `50` | Number of alerts retained in memory |

---

## Faster Alternatives

LLaVA-7B is accurate but slow on CPU. These are drop-in replacements in `vlm.py` and `compiler.py`:

| Model | Speed | Quality | Command |
|---|---|---|---|
| `moondream` | Fast (~2–4s CPU) | Good for simple conditions | `ollama pull moondream` |
| `llava:7b` | Medium (~8–15s CPU) | Recommended baseline | `ollama pull llava:7b` |
| `llava:13b` | Slow | Higher accuracy | `ollama pull llava:13b` |
| `qwen2-vl:7b` | Medium | Strong multilingual | `ollama pull qwen2-vl:7b` |

For prototyping, swap to `moondream` first, it's fast enough to feel interactive on CPU hardware.

---

## Known Limitations

- **Inference latency:** On CPU, frame checking takes 8–20s. The pipeline is designed for ~1fps event detection, not real-time video analysis.
- **JSON parsing:** The compiler uses a regex fallback to strip markdown fences from LLM output. Occasionally the LLM returns malformed JSON — the fallback wraps the raw query as `{"description": "..."}` which still works but is less structured.
- **Single condition:** Only one active condition is supported at a time. Setting a new condition overwrites the previous one.
- **No persistence:** Alert log is in-memory only. Restart the server and it resets.

---

## Extending the Project

Some directions explored in the project write-up:

**Multi-condition support** — extend `active_condition` to a list, run the VLM once per frame with all conditions batched into one prompt.

**Frame differencing gate** — before calling the VLM, compute pixel diff between the current and previous frame. Only invoke the VLM if the diff exceeds a threshold. Cuts VLM calls by 80–90% in static scenes.

**Replay and explain** — add a `/replay` endpoint that accepts a video file path and a query, extracts frames with ffmpeg, and returns timestamped alert events. Bridges into the Grounded Video Annotator project.

**Edge deployment** — export the fast-gate model (YOLO-NAS-S) to TensorRT for Jetson Orin Nano, keep the VLM call as a cloud fallback. See the edge deployment notes in `docs/edge.md` (to be written).

**SAM 2 integration** — when an alert fires, pass the snapshot to SAM 2 with the detected object description as a prompt. Return a pixel-level mask overlay instead of just a bounding box.

---

## Papers

The key work this project builds on:

- **LLaVA-1.5** — Improved Baselines with Visual Instruction Tuning. Liu et al., 2023.
- **Grounding DINO** — Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. Liu et al., 2023.
- **FastV** — An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models. Chen et al., 2024.
- **LLaVA-PruMerge** — Adaptive Token Reduction for Efficient Large Multimodal Models. Shang et al., 2024.
- **Minimalist Vision with Freeform Pixels** — ECCV 2024 Best Paper. (Relevant for the edge variant of this pipeline.)

---

## License

MIT