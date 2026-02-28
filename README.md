# Project Statement
Image classification becomes challenging when the number of categories increases significantly. Traditional approaches struggle to learn complex visual patterns across many classes.
This project aims to design a deep learning system capable of automatically classifying images into 134 distinct categories using a Convolutional Neural Network (CNN).
The goal is to build an end-to-end pipeline including data preprocessing, model training, evaluation, and deployment through a simple web interface.
# Project Idea
The idea behind this project is to apply deep learning techniques to solve a large-scale multi-class classification problem. Instead of manually identifying image features, the CNN automatically learns hierarchical visual patterns such as edges, textures, and object structures.
To make the system practical and user-friendly, a Streamlit web application is developed to allow real-time predictions.
# Objectives
Build a CNN model for high multi-class image classification
Improve generalization using data augmentation
Evaluate model performance using validation metrics
Deploy the trained model through an interactive interface
Demonstrate an end-to-end deep learning workflow
# Model Architecture
Convolutional Neural Network (CNN)
Multiple Convolution + Pooling layers
Data Augmentation for improved robustness
Fully Connected Dense layers
Softmax activation for multi-class prediction
Training Configuration
Loss Function: Categorical Crossentropy
Optimizer: Adam
Output Layer: 134 classes
Evaluation Metric: Accuracy
# Training Details
Parameter
Value
Total Classes :134
Epochs :5
Training Accuracy :~55%
Validation Accuracy:~63%
Training Device:CPU
Due to computational limitations, training was performed on CPU and stopped after achieving stable validation performance.
# Key Observations
Validation accuracy exceeded training accuracy due to strong data augmentation.
Handling 134 classes makes prediction inherently complex.
Achieved performance shows good generalization ability.
Confidence scores are reasonable for large multi-class problems.
# Web Application (Streamlit)
The project includes a simple web interface for real-time predictions.
## Features
Upload an image
Predict image class
Display prediction confidence
Interactive and easy-to-use UI
Run the application
Bash
Copy code
streamlit run app.py
## How to Run the Project
1️ Install dependencies
pip install -r requirements.txt
2️ Train the Model
python train.py
This will:
Load dataset
Train CNN model
Save trained model (model.h5)
3️ Make Prediction
python predict.py
4️ Launch Web App
streamlit run app.py
# Technologies Used
Python
TensorFlow / Keras
NumPy
OpenCV / PIL
Streamlit
Deep Learning (CNN)
# Future Improvements
Train using GPU for higher accuracy
Apply Transfer Learning (ResNet / EfficientNet)
Add Top-3 predictions visualization
Deploy application on cloud platforms
HuggingFace Spaces
Render
Streamlit Cloud
# Skills Demonstrated
Deep Learning Model Development
CNN Architecture Design
Multi-Class Classification
Data Augmentation Techniques
Model Evaluation
Model Deployment using Streamlit
End-to-End ML Pipeline Development
<img width="6000" height="5400" alt="confusion_matrix_full" src="https://github.com/user-attachments/assets/d333e450-d483-4a25-9002-4044652c7a18" />
<img width="3600" height="3000" alt="confusion_matrix_top20" src="https://github.com/user-attachments/assets/18462ebf-f284-48d0-9a0d-8c92713c5484" />
<img width="1920" height="1080" alt="demo" src="https://github.com/user-attachments/assets/a08dbb5c-acb4-40f0-b924-0f1919036b46" />
<img width="1920" height="1080" alt="prediction_output" src="https://github.com/user-attachments/assets/2a7faede-9d0f-43b5-8419-799384f6c168" />
