# 🛡️ Sentry: An Intelligent Violence Detection System

**Sentry** is an AI-powered surveillance system that automatically detects violent behavior in real time and offline video, using a hybrid pipeline of **pose estimation** and **transformer-based temporal modeling**. It goes beyond simple motion detection by understanding human posture and action over time, and pairs this with identity-aware face recognition, automated alerting, and forensic-grade PDF reporting.

> Main Project — MCA (S4), Department of Computer Applications, Government Engineering College, Thrissur, Kerala
> **Author:** Kalidasan R (TCR24MCA-2037)
> **Guide:** Dr. Sminesh C N, Professor & HOD

---

## ✨ Key Features

- **Dual-Mode Detection**
  - 🔴 **Live Mode** — Real-time, low-latency violence detection using pose-based rule analysis.
  - 🟢 **Offline Mode** — Deep, forensic-level analysis of uploaded video using transformer-based temporal modeling.
- **Pose-Based Violence Detection** — Extracts human skeletal keypoints (YOLOv11 Pose) and applies biomechanical rules (strike detection, kick detection, proximity, aggressive stance, etc.) to flag violent motion.
- **Transformer-Based Violence Detection** — VideoMAE processes video clips with self-attention to capture long-range temporal context and classify violent vs. non-violent behavior.
- **Multi-Person Tracking** — ByteTrack maintains stable identity tracking across frames, even under occlusion.
- **Identity-Aware Analysis** — Face detection (SCRFD / RetinaFace) + face recognition (ArcFace) to classify individuals as whitelist, blacklist, or unknown, and escalate severity accordingly.
- **Face Only Mode** — Identity-focused monitoring (access control / blacklist alerts) without action-based inference.
- **Severity Assignment Engine** — Weighted fusion of pose score, transformer score, and identity context into `Normal`, `Suspicious`, or `Danger` classifications.
- **Real-Time Alerts** — Multi-channel notifications (mobile + Telegram) with annotated screenshots, timestamps, and LLM-generated incident summaries (via Ollama + Mistral).
- **Automated PDF Reports** — Structured, multi-page reports with executive summaries, severity breakdowns, timelines, and visual evidence (via FPDF).
- **Multi-Camera Support** — Camera Manager handles single and multi-camera setups (USB, IP/RTSP).

---

## 🏗️ System Architecture

```
Video Input (Live Camera / Uploaded File)
        │
        ▼
 Camera Manager / Batch Frame Processor
        │
        ├── Face Recognition (SCRFD + ArcFace)
        ├── Pose Estimation (YOLOv11 Pose + ByteTrack)
        └── Transformer-Based Violence Detection (VideoMAE)
        │
        ▼
   Rule Engine → Severity Assignment
        │
        ├── Replay & Visualization
        ├── PDF Report Generator
        └── Alert Generator (Mobile / Telegram)
```

The system supports two decision paths:
- **Live camera input** → Pose-based real-time detection + face recognition → Severity assignment.
- **Uploaded video** → Violence detection check → optional high-accuracy transformer inference → Severity assignment.

---

## 🧰 Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.10 |
| Deep Learning | PyTorch, TorchVision |
| Pose Estimation | YOLOv11 Pose (Ultralytics) |
| Temporal Modeling | VideoMAE Transformer (HuggingFace Transformers) |
| Tracking | ByteTrack |
| Face Detection | SCRFD, RetinaFace |
| Face Recognition | ArcFace (ONNX Runtime) |
| Computer Vision | OpenCV |
| Backend Server | Uvicorn (ASGI) |
| Reporting | FPDF |
| Summarization | Ollama (local LLM) + Mistral |
| Frontend | React + TypeScript ([sentry-watch](https://github.com/KalidasanR117/sentry-watch)) |

---

## 💻 Hardware Requirements

| Component | Minimum |
|---|---|
| Processor | Intel Core i5 / AMD Ryzen 5 (x64) or higher |
| Memory | 16 GB RAM (32 GB recommended) |
| GPU | NVIDIA GPU with CUDA support, 6 GB VRAM |
| Storage | 20–50 GB free disk space |
| Camera | USB webcam / IP camera (RTSP) |
| Internet | Required for alerts and model updates |

---

## 🔗 Related Repositories

- **Frontend Dashboard:** [sentry-watch](https://github.com/KalidasanR117/sentry-watch) — the React + TypeScript web dashboard (Live Monitor, Video Analysis, Reports, Deep Face Lab) that connects to this backend.

This repository (`sentry`) contains the **backend** — detection pipelines, models, alerting, and reporting logic. The frontend UI lives in a separate repo linked above.

---

## 📂 Repository Structure

```
.
├── alerts/              # Alert generation and delivery logic
├── core/                # Core detection/severity pipeline
├── events/              # Event logging and management
├── facial_analysis/     # Face detection & recognition (SCRFD, ArcFace, RetinaFace)
├── llm/                 # LLM-based incident summarization (Ollama/Mistral)
├── models/              # Model weights / loaders
├── reports/             # PDF report generation
├── scripts/             # Utility scripts
├── tests/               # Test suite
├── utils/               # Shared utilities
├── webrtc/              # Camera / streaming UI integration
├── after_camera_main.py # Live camera pipeline entry point
├── app.py               # Application entry point
├── main.py               # Main runner
├── pose_face_main.py     # Combined pose + face pipeline
├── test_videomae.py      # VideoMAE offline inference testing
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10
- CUDA-capable GPU (recommended)
- Windows 11 or Linux (Ubuntu 20.04+)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/sentry.git
cd sentry

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
# Start the backend
python app.py

# Or run the live camera pipeline directly
python after_camera_main.py
```

To use the full web dashboard, clone and run the frontend separately:

```bash
git clone https://github.com/KalidasanR117/sentry-watch.git
cd sentry-watch
npm install
npm run dev
```

Then open the dashboard to:
- Start **Live Monitor** for real-time camera-based detection.
- Use **Video Analysis** to upload a file for pose-based or transformer-based (Deep Scan) offline analysis.
- View **Reports** for previously generated PDF reports.
- Use **Deep Face Lab** for identity/face-only analysis.

---

## 📊 Evaluation Results

| Method | Accuracy | Macro Precision | Macro Recall |
|---|---|---|---|
| Pose-Based Detection (Live) | 75.4% | 76.85% | 75.4% |
| Transformer-Based Detection (Offline, VideoMAE) | 86.3% | 86.3% | 86.3% |

- **Pose-based detection** favors real-time responsiveness with strong true-positive performance, at the cost of a higher false-positive rate.
- **Transformer-based detection** provides superior contextual accuracy for forensic/offline review, at higher computational cost.

---

## 🔮 Future Enhancements

- Integrate **ST-GCN** for improved joint-relationship modeling and reduced false positives.
- **Adaptive severity calibration** with dynamic thresholds for crowded/complex environments.
- **Edge deployment** via model compression/quantization (e.g., NVIDIA Jetson).
- **Dataset expansion** with diverse real-world scenarios for better generalization.

---

## 📖 References

Key literature that informed this project's design (see full bibliography in the project thesis):

1. Wastupranata et al., *"Deep learning for abnormal human behavior detection in surveillance videos—a survey,"* Electronics, 2024.
2. Pham et al., *"Video-based human action recognition using deep learning: a review,"* arXiv:2208.03775, 2022.
3. Beddiar et al., *"Vision-based human activity recognition: a survey,"* Multimedia Tools and Applications, 2020.
4. Zheng et al., *"Deep learning-based human pose estimation: A survey,"* ACM Computing Surveys, 2023.
5. Vaswani et al., *"Attention is all you need,"* NeurIPS, 2017.
6. Deng et al., *"ArcFace: Additive angular margin loss for deep face recognition,"* CVPR, 2019.

---

## 📄 License

This project was developed as an academic Main Project for the Master of Computer Applications degree (APJ Abdul Kalam Technological University). Please check with the author before reuse.

---

## 🙏 Acknowledgements

Developed under the guidance of **Dr. Sminesh C N**, Professor & HOD, Department of Computer Applications, Government Engineering College, Thrissur.
