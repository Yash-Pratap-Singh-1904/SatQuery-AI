# SatQuery AI — Multi-Modal Disaster Aerial Imagery Analyzer (CSE471)

**Team Members:**
- **Yash Pratap Singh** (Roll No: 47 | Reg No: 12408053)
- **Heman Rajkumar** (Roll No: 16 | Reg No: 12408560)
- **Institution:** School of Computer Science and Engineering, Lovely Professional University

## Overview
SatQuery AI combines a fine-tuned **ResNet-18 Convolutional Neural Network** (97.80% test accuracy on 9,000 NOAA Hurricane Harvey satellite tiles) with **Grad-CAM** visual saliency heatmaps, **YOLOv8x** rescue object localization, and **Qwen2-VL-2B** multi-turn conversational reasoning.

## Setup & Execution
1. **Backend (Google Colab T4 GPU):** Open `SatQuery_AI_CNN_Pipeline.ipynb`, add your ngrok token, and run all cells to train ResNet-18 and launch the FastAPI server.
2. **Frontend (React + Vite):**
   ```bash
   npm install
   npm run dev