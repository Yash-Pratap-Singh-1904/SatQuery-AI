# SatQuery AI — Explainable Multi-Modal Aerial & Satellite Disaster Intelligence System

[![Course](https://img.shields.io/badge/Course-CSE471%20Deep%20Learning%20for%20Computer%20Vision-1B365D.svg)](#)
[![Test Accuracy](https://img.shields.io/badge/ResNet--18%20Test%20Accuracy-97.80%25-10B981.svg)](#)
[![Frontend](https://img.shields.io/badge/Frontend-React%20%2B%20TypeScript%20%2B%20Vite-38BDF8.svg)](#)
[![Backend](https://img.shields.io/badge/Backend-FastAPI%20%2B%20PyTorch%20%2B%20YOLOv8-F59E0B.svg)](#)

**SatQuery AI** is an end-to-end, explainable multi-modal geospatial intelligence platform built for rapid post-disaster damage assessment and rescue triage. It integrates a fine-tuned **ResNet-18 Convolutional Neural Network (CNN)** with **Grad-CAM** visual saliency heatmaps, **YOLOv8x** human/vehicle rescue localization, and **Qwen2-VL-2B** conversational vision-language reasoning inside an interactive web dashboard.

---

## Important Links

* **Live Deployed Web Application:** [https://sat-query-ai-xi.vercel.app](https://sat-query-ai-xi.vercel.app)
* **Public GitHub Repository:** [https://github.com/Yash-Pratap-Singh-1904/SatQuery-AI](https://github.com/Yash-Pratap-Singh-1904/SatQuery-AI)
* **Colab Backend Inference Endpoint:** `https://arrival-viewless-spill.ngrok-free.dev/api/analyze` *(Requires active Google Colab T4 GPU runtime)*
* **Benchmark Dataset (Kaggle):** [NOAA Hurricane Harvey Satellite Images](https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage)

---

## Project Team

* **Yash Pratap Singh** — Roll No.: `47` | Reg. No.: `12408053`
* **Heman Rajkumar** — Roll No.: `16` | Reg. No.: `12408560`
* **Department:** School of Computer Science and Engineering, Lovely Professional University (LPU), Jalandhar, Punjab

---

---

## End-to-End System Workflow

SatQuery AI uses a decoupled client-server architecture that separates lightweight browser rendering from GPU-accelerated deep learning inference:

    [1. Client Interface: React + TypeScript + Tailwind UI]
           │  User uploads an aerial/satellite tile (up to 400MP) & enters a query
           ▼  HTTPS POST /api/analyze (multipart/form-data + 3-turn chat history)
    [2. Cloud Inference Server: FastAPI + Uvicorn (Google Colab NVIDIA T4 GPU)]
           │
           ├──► Step A: Image Safety & Resampling
           │      • Bypasses PIL decompression limits for ultra-high-res satellite imagery
           │      • Downscales oversized tiles (>1280px) via Lanczos resampling
           │
           ├──► Step B: Custom ResNet-18 CNN & Grad-CAM Visual Grounding
           │      • Forward pass predicts DAMAGE vs. NO_DAMAGE softmax probabilities
           │      • Backward hooks on layer4[-1] generate a JET Grad-CAM heatmap
           │      • OpenCV contour thresholding (cam_norm > 0.65) draws primary anomaly BBox
           │
           ├──► Step C: YOLOv8x Rescue & Object Triage (1024px)
           │      • Detects humans, rescue workers, cars/trucks, boats, and aircraft
           │      • Checks coordinate overlap with Grad-CAM hazard zone (cam_norm >= 0.45)
           │      • Draws crimson boxes for humans in danger; cyan boxes for safe perimeter
           │
           └──► Step D: Qwen2-VL-2B Domain Calibration & Multi-Turn Q&A
                  • Calibrates out-of-domain macro coastal scenes to prevent false flood alerts
                  • Synthesizes structured natural language answers conditioned on CNN + YOLO telemetry
           │
           ▼  Returns JSON (Base64 Grad-CAM overlay, probabilities, BBox, & chat response)
    [3. Interactive Visualization: Split Original/Grad-CAM Viewer & Persistent Chat Feed]

---

## Technology Stack & Component Roles

| Layer / Subsystem | Technologies Used | Specific Role in SatQuery AI |
| :--- | :--- | :--- |
| **Classification Backbone** | PyTorch, Torchvision (`ResNet-18`) | Performs supervised binary classification (`damage` vs. `no_damage`) via transfer learning (`layer4` + custom regularized head). |
| **Visual Explainability** | PyTorch Forward/Backward Hooks, OpenCV (`cv2`), NumPy | Extracts spatial feature gradients from `layer4[-1]`, generates color-mapped Grad-CAM overlays, and computes contour bounding boxes. |
| **Rescue Object Triage** | Ultralytics `YOLOv8x` (`imgsz=1024`) | Localizes small humans, rescue personnel, ground vehicles, boats, and aircraft, flagging individuals inside high-risk Grad-CAM hazard zones. |
| **Conversational VLM** | Hugging Face `Qwen2-VL-2B-Instruct` (FP16), `transformers` | Provides natural language visual question answering, maintains multi-turn context, and prevents domain-shift false positives on coastal airports/harbors. |
| **Backend API & Tunnel** | FastAPI, Uvicorn, Pillow (`PIL`), `pyngrok`, `nest-asyncio` | Exposes the asynchronous `POST /api/analyze` REST endpoint on Google Colab T4 GPU and bridges it to the web via TLS-encrypted tunneling. |
| **Frontend Dashboard** | React, TypeScript, Vite, Tailwind CSS, Lucide Icons | Renders the glassmorphic UI featuring Split View / Grad-CAM Only / Original toggles, confidence telemetry meters, quick prompts, and persistent chat. |

---

## Dataset Description & Preprocessing Pipeline

### 1. Dataset Source & Partitioning
The supervised CNN was trained and evaluated on the **Satellite Images of Hurricane Damage** dataset curated from **NOAA** aerial imagery following Hurricane Harvey (2017):
* **Training Set (`train_another`):** `10,000` images (`5,000` damage, `5,000` no_damage)
* **Validation Set (`validation_another`):** `2,000` images (`1,000` damage, `1,000` no_damage)
* **Unseen Test Set (`test_another`):** `9,000` images (`8,000` damage, `1,000` no_damage)
* **Native Format:** `128 × 128` pixel 3-channel RGB JPEG aerial patches.

### 2. Data Preprocessing & Augmentation
To ensure rotational invariance and prevent overfitting during training, every input tile passes through the following preprocessing pipeline:
1. **Spatial Resizing:** Bilinear resampling from `128 × 128` to `224 × 224` pixels to match ResNet-18 receptive field dimensions.
2. **Geometric Augmentations (Training Only):**
   * Random Horizontal Flip (`p = 0.5`)
   * Random Vertical Flip (`p = 0.2`)
   * Random Rotation (`±15°`)
3. **Photometric Augmentation (Training Only):** Color Jitter (`brightness = 0.1`, `contrast = 0.1`) to simulate varying atmospheric lighting.
4. **Channel Normalization:** Tensor scaling to `[0, 1]` followed by ImageNet channel standardization (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`).

### 3. Model Performance Summary
* **Training Accuracy (Epoch 5):** `98.87%` (Loss: `0.0327`)
* **Validation Accuracy (Epoch 5):** `98.65%` (Loss: `0.0356`)
* **Unseen Test Accuracy (9,000 tiles):** `97.80%` (`0.9875` F1-score on `damage`; `7,811` TP, `991` TN, `189` FN, `9` FP)

---

## Repository Structure

    SatQuery-AI/
    ├── SatQuery_AI_CNN_Pipeline.ipynb   # Complete Colab notebook: dataset setup, ResNet-18 training, Grad-CAM, YOLOv8 & FastAPI server
    ├── src/
    │   ├── App.tsx                      # Main React dashboard, split heatmap viewer, telemetry cards & multi-turn chat logic
    │   ├── App.css                      # Custom styling rules
    │   ├── index.css                    # Tailwind CSS configuration
    │   └── main.tsx                     # React DOM entry point
    ├── public/                          # Static assets & icons
    ├── index.html                       # HTML entry template
    ├── package.json                     # Node.js / React / Vite dependencies
    ├── requirements.txt                 # Python / PyTorch / FastAPI / Ultralytics dependencies
    └── README.md                        # Project documentation & setup guide

---

## How to Run Locally After Downloading the ZIP File

Follow these step-by-step instructions to run both the **GPU Inference Backend** and the **React Web Dashboard** locally after extracting the project ZIP archive:

### Prerequisites
* **Node.js** (v18 or higher) and **npm** installed on your computer
* A free **Google Colab** account (with `T4 GPU` runtime enabled)
* A free **ngrok** account and authtoken from [dashboard.ngrok.com](https://dashboard.ngrok.com)

---

### Part 1: Start the Deep Learning Backend (Google Colab T4 GPU)

1. Extract the downloaded ZIP file and locate `SatQuery_AI_CNN_Pipeline.ipynb`.
2. Open [Google Colab](https://colab.research.google.com), click **File > Upload notebook**, and upload `SatQuery_AI_CNN_Pipeline.ipynb`.
3. In Colab, go to **Runtime > Change runtime type**, select **T4 GPU**, and click **Save**.
4. Run Cells 1 through 5 sequentially to download the Hurricane Harvey dataset, train/evaluate the ResNet-18 model, and verify the confusion matrix.
5. In the final server cell (**Cell 6**), replace `"YOUR_NGROK_AUTHTOKEN"` with your personal ngrok authtoken and run the cell.
6. Once initialized, Colab will print your active public API URL in the console:

       🚀 PUBLIC API URL: NgrokTunnel: "https://your-tunnel-url.ngrok-free.dev" -> "http://localhost:8000"
       INFO: Uvicorn running on http://0.0.0.0:8000

   Copy this `https://...ngrok-free.dev` URL.

*(Optional Local Python Environment Setup: If you have a local NVIDIA CUDA GPU and wish to install Python packages locally instead of Colab, run `pip install -r requirements.txt`.)*

---

### Part 2: Run the React Frontend Locally

1. Open your terminal / command prompt and navigate into the extracted project directory:

       cd path/to/SatQuery-AI

2. Install all required Node.js dependencies:

       npm install

3. Open `src/App.tsx` in your code editor and update the `API_ENDPOINT` constant at the top of the file with the ngrok URL copied from Part 1:

       const API_ENDPOINT = "https://your-tunnel-url.ngrok-free.dev/api/analyze";

4. Start the local Vite development server:

       npm run dev

5. Open your browser and navigate to **`http://localhost:5173`**.
6. Click **Upload Aerial Tile**, select any satellite or drone disaster image, and click any preset question or type your own custom prompt to test real-time classification, Grad-CAM heatmaps, human hazard detection, and multi-turn chat.

