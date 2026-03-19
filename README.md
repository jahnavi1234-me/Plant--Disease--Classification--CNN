 

 AI-Powered Plant Disease Detection using CNN

 Project Description

This project is a deep learning-based application designed to automatically detect plant diseases from leaf images. It uses a Convolutional Neural Network (CNN) with transfer learning to classify plant leaf images into 134 different disease categories. The system is deployed using a Streamlit web application, allowing users to upload images and receive real-time predictions.


---

 Project Statement

Plant diseases significantly impact agricultural productivity and crop quality. Traditional disease identification methods rely on manual inspection, which can be time-consuming, error-prone, and dependent on expert knowledge. This becomes increasingly challenging when dealing with a large number of disease categories with similar visual patterns.
This project aims to develop an automated deep learning-based system capable of accurately classifying plant leaf images into 134 disease categories, enabling faster and more reliable disease detection.


---

 Features

Classifies plant leaf images into 134 disease categories

Uses MobileNetV2 transfer learning for efficient training

Applies data augmentation to improve model generalization

Provides real-time predictions through a Streamlit web app

Displays predicted class with confidence score

Includes model evaluation using confusion matrix and classification report

<img width="6000" height="5400" alt="confusion_matrix_full" src="https://github.com/user-attachments/assets/8062d4da-9f04-4abc-87f7-8238b285317a" />

<img width="3600" height="3000" alt="confusion_matrix_top20" src="https://github.com/user-attachments/assets/1e634623-582c-4556-a54a-54bfe1ad6cde" />

---

 Technologies Used

Python
TensorFlow / Keras
NumPy
OpenCV / PIL
Matplotlib / Seaborn
Scikit-learn
Streamlit


---

 Project Architecture

Data Collection → Data Preprocessing → Model Building → Training → Evaluation → Deployment

1. Data Collection
   - Plant leaf image dataset with 134 classes

2. Data Preprocessing
   - Image resizing (224x224)
   - Normalization
   - Data augmentation

3. Model Building
   - MobileNetV2 (pre-trained)
   - Custom dense layers + dropout

4. Training
   - Optimizer: Adam
   - Loss: Categorical Crossentropy

5. Evaluation
   - Classification report
   - Confusion matrix
   - Top confused classes analysis

6. Deployment
   - Streamlit web application
   - Real-time image prediction


---
```
 Folder Structure

CNN_PROJECT/

├── data/
│   ├── raw/
│   ├── processed/

├── models/
│   ├── cnn_model.keras
│   ├── best_model.keras
│   ├── class_names.json
│   ├── confusion_matrix_full.png
│   ├── confusion_matrix_top20.png
│   ├── evaluation_report.txt

├── src/
│   ├── train_data.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── load_data.py

├── app/
│   └── app.py

├── outputs/
│   └── sample_images/

├── requirements.txt
└── README.md


---
```
Installation

Clone the repository and install dependencies using pip install -r requirements.txt.


---

 How to Run

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/train_data.py

# Evaluate the model
python src/evaluate.py

# Run prediction script
python src/predict.py

# Launch Streamlit app
streamlit run app/app.py

---

 Example Output

Predicted Disease: strawberry_rotten
Confidence Score: 0.47

The system takes a leaf  or fruit or vegetable image as input and outputs the predicted disease class along with a confidence score.

<img width="1920" height="1080" alt="demo" src="https://github.com/user-attachments/assets/b27d470e-b2ca-49af-8a44-380574aa86f7" />
<img width="1920" height="1080" alt="prediction_output" src="https://github.com/user-attachments/assets/c6927598-68ce-4c37-a989-c702ffeba80d" />

---

 Future Improvements

Train the model using GPU for higher accuracy
Fine-tune upper layers of MobileNetV2
Use advanced architectures like EfficientNet or ResNet
Add Top-3 predictions visualization
Deploy on cloud platforms such as Streamlit Cloud or Hugging Face Spaces


---

 Author

Jahnavi Besabathini
Aspiring Generative AI Engineer
 


---

If you want, I can next make this even stronger with GitHub badges + demo section, which makes recruiters notice your project immediately 🔥
