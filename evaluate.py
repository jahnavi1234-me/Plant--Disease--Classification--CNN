import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import os

# --------------------
# CONFIG
# --------------------
IMG_SIZE = (224, 224)   # must match training
BATCH_SIZE = 32
MODEL_PATH = "models/cnn_model.keras"
TEST_DIR = "data/processed/test"

# --------------------
# LOAD MODEL
# --------------------
model = tf.keras.models.load_model(MODEL_PATH)

# --------------------
# LOAD TEST DATA
# --------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# --------------------
# PREDICTIONS
# --------------------
pred_probs = model.predict(test_generator)
y_pred = np.argmax(pred_probs, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# --------------------
# CLASSIFICATION REPORT
# --------------------
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n")
print(report)

# Save report to file
with open("models/evaluation_report.txt", "w") as f:
    f.write(report)

# --------------------
# CONFUSION MATRIX (Full)
# --------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(20, 18))
sns.heatmap(
    cm,
    cmap="pinks",
    xticklabels=class_names,
    yticklabels=class_names,
    fmt="d"
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Full)")
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig("models/confusion_matrix_full.png", dpi=300)
plt.show()

# --------------------
# CONFUSION MATRIX (Top Confused Classes)
# --------------------
# Find top 20 most confused classes
errors = cm.copy()
np.fill_diagonal(errors, 0)
top_errors_idx = errors.sum(axis=0).argsort()[-20:]  # top 20 errors
cm_small = cm[np.ix_(top_errors_idx, top_errors_idx)]
small_classes = [class_names[i] for i in top_errors_idx]

plt.figure(figsize=(12,10))
sns.heatmap(cm_small, cmap="pink", xticklabels=small_classes, yticklabels=small_classes, fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Top 20 Most Confused Classes")
plt.xticks(rotation=90, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig("models/confusion_matrix_top20.png", dpi=300)
plt.show()
