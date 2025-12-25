# SafeProduction – Real-Time Human Detection & Tracking

SafeProduction is a **real-time computer vision application** for human detection and tracking using **YOLO (Ultralytics)**, **ByteTrack (Supervision)**, and **Streamlit WebRTC**.
The project is designed for safety-related use cases (e.g. APD/PPE monitoring) and runs directly from a webcam or video stream.

---

## 🚀 Features

* Real-time **human detection** (YOLO)
* Real-time **multi-object tracking** (ByteTrack)
* Stable video streaming using **Streamlit WebRTC**
* Modular project structure (model, tracker, handler, pipeline)
* Easy to extend for **PPE / APD detection**

---

## 📁 Project Structure

```
SafeProduction/
├── config/                            # Configuration files
├── src/
│ ├── app.py                           # Streamlit entry point
│ ├── handlers/
│ │   ├── pipeline_handler.py            # Main CV pipeline
│ │   └── human_handler.py               # Human detection & tracking handler
│ ├── models/                           # Trained YOLO models (.pt)
│ │   ├── downloader.py                 # Downloading models from gdrive
│ │   ├── human.pt     
| |   └── ppe_construction.pt          
│ ├── vision_models/
│ │   └── human/
│ │       ├── model.py                   # YOLO model wrapper
│ │       ├── predictor.py               # Detection & tracking logic
│ │       └── tracker.py                 # ByteTrack + visualization
│ ├── utils/                           # Helper utilities (future use)
│ └── schemas/                         # Data schemas (future use)
├── venv/                              # Python virtual environment
├── .env.example                       # Environment variables
├── .gitignore
├── requirements.txt
└── README.md
```
You can add another model logic in vision_models with new folder.

---

## 🧰 Requirements

* Python **3.9 – 3.11** (recommended: 3.10)
* Webcam (for real-time demo)
* OS: Windows / Linux / macOS

---

## 🐍 1. Create Virtual Environment (venv)

### Windows (PowerShell)

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 📦 2. Install Dependencies

Upgrade pip first:

```bash
pip install --upgrade pip
```

Install required packages:

```bash
pip install -r requirements.txt
```
---

## 📥 3. Download YOLO Model

Use file downloader.py to download the models, but dont forget to put the url drive inside the .env file.
---

## ▶️ 4. Run the Application

From the `src` directory:

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501
```

Allow webcam access when prompted.

---

## 🎥 How It Works (Pipeline Overview)

1. **Streamlit WebRTC** captures webcam frames
2. Each frame is sent to `video_frame_callback`
3. Frame passes through:
   * YOLO detection
   * ByteTrack tracking
   * Bounding box & ID annotation
4. Detecting each of ppe.
5. Annotated frame is returned to the browser with green if the ppe is completed and red if the ppe is not completed
6. Display the message for every id person and their ppe.

---

## 🧠 Important Notes

* Tracking IDs are **not permanent identities**
* IDs may change if:

  * the person leaves the frame
  * detection is lost temporarily
* This is **expected behavior** for ByteTrack

---

## 🛠️ Common Issues

### Video Freezes at Start

* Ensure `HumanTracker` does **NOT** instantiate `sv.Detections()` directly
* Always return a valid NumPy frame
* Exceptions inside WebRTC callbacks are silent — use try/except

---

## 📄 License

This project is for academic and research purposes.
You are free to modify and extend it.


