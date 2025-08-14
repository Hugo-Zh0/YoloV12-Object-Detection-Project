# 🚀 YOLOv12 Object Detection Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Anaconda](https://img.shields.io/badge/Anaconda-Navigator-green.svg)](https://www.anaconda.com/download)
[![VS Code](https://img.shields.io/badge/Editor-VS%20Code-blue.svg)](https://code.visualstudio.com/)
[![Ultralytics](https://img.shields.io/badge/YOLOv12-Ultralytics-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---
<br>

## 📌 Overview
A **collaborative group project** by Swinburne University students in partnership with **CSIRO**.  
This repository contains the setup, configuration, and workflow for training and running **YOLOv12** object detection models.

**👨‍💻 Team Members:** Harron, Feng, Bunmi, Huss, Hugo.

---

<br>

## 📦 Prerequisites

Before you begin, make sure you have the following installed:

| Tool | Purpose | Download | Tutorial |
|------|---------|----------|----------|
| **Python 3.10+** | Required to run YOLOv12 | [🔗 Download](https://www.python.org/downloads/) | [🎞️YT-Link](https://www.youtube.com/watch?v=C3bOxcILGu4&pp=ygUXaW5zdGFsbCBweXRob24gd2luZG93cyA%3D)
| **Anaconda Navigator** | Virtual environment management | [🔗 Download](https://www.anaconda.com/download) | [🎞️YT-Link](https://www.youtube.com/watch?v=vrA6Xv0k4a4&ab_channel=DevOpsCamp)
| **Visual Studio Code** | Code editing & Python integration | [🔗 Download](https://code.visualstudio.com/) | [🎞️YT-Link](https://www.youtube.com/watch?v=mIVB-SNycKI&ab_channel=GeekyScript) 
| **Git** | Clone and manage repositories | [🔗 Download](https://git-scm.com/downloads) | [🎞️YT-Link](https://www.youtube.com/watch?v=iYkLrXobBbA&ab_channel=CodeBear)

---

<br>

## ⚙️ Setting Up the Codebase

1️⃣ **Clone this Repository**

https://github.com/user-attachments/assets/69724dcf-b7e4-44f8-8742-8cf390b4e77c

<br>

2️⃣ **Create a Virtual Environment** (via Anaconda Navigator or terminal)

https://github.com/user-attachments/assets/9e9edece-bb32-4995-b8f2-379e8e60a3ad

```bash
conda create -n YoloV12-Object-Detection-Project python=3.12 -y
conda activate YoloV12-Object-Detection-Project
```

<br>

3️⃣ **Install Dependencies**

https://github.com/user-attachments/assets/14b34deb-45d9-4864-9cb0-27e5bd16f204
https://github.com/user-attachments/assets/5d072ef0-dbbf-4f3f-af07-6d4a34e2af12

```bash
pip install ultralytics

# For GPU users (NVIDIA ONLY):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CPU users:
pip install torch torchvision torchaudio
```

<br>

4️⃣ **Open in VS Code**
```bash 
1. Inside VSC do the command Ctrl+Shift+P
2. Search for Python Select Interpreter
3. Select your created conda enviroment
4. You will now see it load and activate in the terminal below VSC
```

https://github.com/user-attachments/assets/ab91682d-b073-4d9d-83cd-9ae0c548f5b2

---

<br>

## 📂 Project Structure
```
📦 YOLOv12-Object-Detection-Project
 ┣ 📂 data              # Training and validation datasets
 ┣ 📂 models            # YOLOv12 model configuration files
 ┣ 📂 scripts           # Custom scripts for training/inference
 ┣ 📜 requirements.txt  # Project dependencies
 ┗ 📜 README.md         # Project documentation
```

---

<br>

## 📜 License
This project is licensed under the [MIT License](LICENSE).

## ⭐ Acknowledgements
- **[Ultralytics YOLOv12](https://github.com/ultralytics/ultralytics)**
- **[Roboflow](https://roboflow.com/)**
- **CSIRO** – Project client & industry guidance
