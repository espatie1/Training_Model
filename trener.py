import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import matplotlib.pyplot as plt

# === 1. Data Loading ===
data = pd.read_csv("balanced_color_data.csv")  # Loading balanced data

# Shuffle the data for random order
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Create additional features based on rules
epsilon = 1e-6  # Minimum value to prevent division by zero
data["R_G_ratio"] = data["R"] / (data["G"] + epsilon)  # Avoid division by zero
data["R_minus_G"] = data["R"] - data["G"]
data["Dominant_G"] = data["G"]

# Separate input features and labels
X = data[["R", "G", "B", "R_G_ratio", "R_minus_G", "Dominant_G"]].values
y = data["Label"].values

# === 2. Check data for invalid values ===
# Ensure no inf or nan values
if np.any(np.isinf(X)) or np.any(np.isnan(X)):
    raise ValueError("Data contains inf or nan. Check the data sources.")

# Encode labels into numerical format
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)  # Red -> 0, Green -> 1, Blue -> 2, None -> 3

# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# === 3. Create and train the model ===
# Create a model with two hidden layers of 64 neurons each
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(6,)),    # Input layer: 6 features (R, G, B, R/G, R-G, G)
    tf.keras.layers.Dense(8, activation='relu'),  # First hidden layer
    tf.keras.layers.Dense(8, activation='relu'),  # Second hidden layer
    tf.keras.layers.Dense(4, activation='softmax')  # Output layer: 4 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Add callbacks to improve training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy", mode="max"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)
]

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=200,  # Increase number of epochs
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# Load the best model (with the correct extension)
model = tf.keras.models.load_model("best_model.keras")

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# === 4. Save the model and preprocessors ===
# Save the Keras model
model.save("color_classifier.h5")

# Save the label encoder and scaler
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and preprocessors saved.")

# === 5. Convert the model to TensorFlow Lite ===
# Load the trained model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model in TFLite format
with open("color_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite and saved as color_classifier.tflite")

# === 6. Display plots ===
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
