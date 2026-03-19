 Multi-Class Image Classification System (134 Classes)

 
 Project Description

This project is a deep learning–based image classification system that can classify images into 134 different categories. It uses a Convolutional Neural Network (CNN) with transfer learning to achieve efficient and accurate predictions. The model is deployed using a Streamlit web application for real-time image classification.

 Problem Statement

Classifying images into multiple categories is a challenging task, especially when the number of classes is large. This project aims to build a scalable and efficient model that can accurately classify images into 134 classes and make it accessible through a simple web interface.

 Features

Multi-class classification (134 categories)
Transfer learning using MobileNetV2
Data augmentation for better generalization
Real-time prediction via web app
Displays predicted class with confidence score
Model evaluation using confusion matrix

Technologies Used

Python
TensorFlow / Keras
NumPy
Matplotlib
Pillow
Streamlit

 Project Architecture

Data Preprocessing
Images resized to 224×224
Pixel normalization (0–1 scaling)
Data augmentation (rotation, zoom, flip)
Model
Pretrained MobileNetV2 (feature extractor)
Frozen base layers
Custom layers:
GlobalAveragePooling
Dense (256, ReLU)
Dropout (0.5)
Output layer (Softmax for 134 classes)
Training
Optimizer: Adam (learning rate = 0.0001)
Loss: Categorical Crossentropy
Callbacks: EarlyStopping, ModelCheckpoint
Deployment
Streamlit web app for user interaction
 Folder Structure
```
CNN_PROJECT/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   ├── cnn_model.keras
│   ├── best_model.keras
│   └── class_names.json
│
├── notebook/
│
├── train.py
├── split.py
├── predict.py
├── evaluate.py
├── app.py
├── requirements.txt
└── README.md
```

 Installation
 
git clone https://github.com/your-username/cnn-134-class-image-classifier.git
cd cnn-134-class-image-classifier
pip install -r requirements.txt
- How to Run
Train Model
 
python train.py
Run Web App
 
streamlit run app.py
 
 Example Output

Input: Uploaded image
Output:
Predicted Class: Rice Brown Spot
Confidence Score: 0.36

 Future Improvements

Apply fine-tuning on pretrained layers
Use advanced models like EfficientNet
Improve dataset quality and balance
Deploy on cloud (AWS / GCP)
Add top-3 predictions for better usability

 Author

Jahnavi Besabathini
B.Tech CSE (AI & ML)
Aspiring Generative AI Engineer
