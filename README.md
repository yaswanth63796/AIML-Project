<u>Child Labour Detection System</u>
<u>Overview</u>

The Child Labour Detection System is a deep learning-based computer vision project developed to automatically detect potential instances of child labour from surveillance footage or images. Manual inspection for identifying child labour is inefficient and error-prone. This project leverages Convolutional Neural Networks (CNNs) such as VGG16, ResNet, and YOLOv3 to automate this detection process, supporting organizations and authorities in monitoring and preventing child exploitation.

<u>Objective</u>

The primary objective of this project is to design and implement an intelligent, automated, and scalable system capable of identifying and classifying child labour activities in real-world environments using deep learning techniques.

<u>Methodology</u>
1. Dataset Preparation

The dataset contains video samples categorized as Child Labour and Adult Labour.

Video frames are extracted, resized, normalized, and labeled.

The prepared frames are used to train CNN-based models.

2. Model Training

CNN architectures such as VGG16, ResNet, and YOLOv3 are implemented for classification tasks.

The models learn to differentiate between adult and child workers using supervised learning.

3. Detection Process

Video frames are analyzed individually through the trained model.

A voting mechanism aggregates frame-level predictions to decide whether a video segment indicates child labour.

4. Evaluation

Model performance is assessed using key metrics such as Accuracy, Precision, Recall, and F1-Score.

<u>Technologies Used</u>
Category	Tools / Frameworks
Programming Language	Python
Deep Learning Framework	PyTorch
Computer Vision	OpenCV
Model Architectures	VGG16, ResNet, YOLOv3
Machine Learning Utilities	Scikit-learn
Data Handling	NumPy, Pandas
Visualization	Matplotlib, Seaborn
<u>Project Structure</u>
Child-Labour-Detection-System/
│
├── Data_frames/                # Dataset (train/test/validation)
├── models/                     # Trained model files (.pth)
├── scripts/                    # Python scripts for processing and training
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── preprocess_videos.py
├── static/                     # Static files (if web interface used)
├── templates/                  # HTML templates (if using Flask)
├── app.py                      # Flask web application entry point
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── LICENSE                      # License file (optional)

<u>Installation and Setup</u>

Follow the steps below to set up and run the project locally:

Step 1: Clone the repository
git clone https://github.com/YOUR-USERNAME/Child-Labour-Detection-System.git
cd Child-Labour-Detection-System

Step 2: Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # For Windows
source venv/bin/activate   # For Linux/Mac

Step 3: Install dependencies
pip install -r requirements.txt

Step 4: Train the model
python scripts/train_model.py

Step 5: Evaluate the model
python scripts/evaluate_model.py

Step 6: (Optional) Run the web application
python app.py

<u>Evaluation Metrics</u>
Metric	Description
Accuracy	Measures overall correctness of predictions.
Precision	Indicates how many of the detected child labour instances are actually correct.
Recall	Reflects how well the model identifies actual cases of child labour.
F1-Score	Harmonic mean of precision and recall, providing a balanced performance measure.
<u>Future Enhancements</u>

Integration of real-time video stream processing.

Model optimization for deployment on edge devices (e.g., Raspberry Pi).

Implementation of an alert system for live detection.

Expanding dataset diversity for improved accuracy.

<u>Contributors</u>

Yaswanth V — Project Developer & Researcher

<u>License</u>

This project is licensed under the MIT License.
Refer to the LICENSE
 file for more information.
