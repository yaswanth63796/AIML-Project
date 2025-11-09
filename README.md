# 🧠 Child Labour Detection System

## 📘 Overview
The **Child Labour Detection System** is a Deep Learning-based Computer Vision project designed to automatically identify instances of child labour from video footage or surveillance data. Traditional monitoring methods rely on manual inspections, which are time-consuming and prone to errors.  
This system utilizes **Convolutional Neural Networks (CNNs)** such as **VGG16** and **ResNet** to detect the presence of children in industrial or working environments, helping NGOs and authorities monitor and prevent child exploitation efficiently.

---

## 🎯 Objective
To develop an **intelligent, automated, and scalable system** that detects and classifies child labour activities from real-world video surveillance using deep learning models.

---

## 🧠 Methodology

1. **Dataset Preparation**
   - Dataset includes videos categorized as *Child Labour* and *Adult Labour*.
   - Frames are extracted, resized, and normalized.
   - Each frame is labeled and prepared for CNN training.

2. **Model Training**
   - CNN architectures like **VGG16**, **ResNet**, or **YOLOv3** are used.
   - Models learn to distinguish between adult and child workers.

3. **Detection Process**
   - Each frame is analyzed individually.
   - A **voting mechanism** determines if a video segment involves child labour.

4. **Evaluation**
   - Performance is measured using **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

---

## 🧰 Technologies Used

| Category | Tools / Frameworks |
|-----------|--------------------|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Computer Vision | OpenCV |
| Model Architectures | VGG16, ResNet, YOLOv3 |
| Machine Learning Library | Scikit-learn |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |

---

## 📁 Project Structure

Child-Labour-Detection-System/
│
├── Data_frames/ # Dataset (train/test/validation)
├── models/ # Saved model files (.pth)
├── scripts/ # Python scripts
│ ├── train_model.py
│ ├── evaluate_model.py
│ └── preprocess_videos.py
├── static/ # Frontend static files (optional)
├── templates/ # HTML templates (optional, if using Flask)
├── app.py # Flask web application (optional)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # License file (optional)

yaml
Copy code

---

## ⚙️ Installation and Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR-USERNAME/Child-Labour-Detection-System.git
   cd Child-Labour-Detection-System
Create and activate a virtual environment

bash
Copy code
python -m venv venv
venv\Scripts\activate      # For Windows
source venv/bin/activate   # For Linux/Mac
Install required dependencies

bash
Copy code
pip install -r requirements.txt
Run model training

bash
Copy code
python scripts/train_model.py
Evaluate the trained model

bash
Copy code
python scripts/evaluate_model.py
(Optional) Run the web app

bash
Copy code
python app.py
📊 Evaluation Metrics
Metric	Description
Accuracy	Measures overall prediction correctness
Precision	How many predicted child labour cases were correct
Recall	How many actual child labour cases were detected
F1-Score	Harmonic mean of Precision and Recall
