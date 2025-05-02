# Food Recognition Project

A real-time food item recognition system powered by a custom-trained Keras model and a simple Flask web app.

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/food-recognition.git
cd food-recognition
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Contents of requirements.txt should include:

```plaintext
tensorflow>=2.6
opencv-python
numpy
flask
```

## 3. Train & Save the Model (optional)
If you want to retrain the model or tweak hyperparameters:

Open food_classification_model.ipynb
Follow the cells to preprocess your data, build the model, and train.
At the end of training, save your model:

```python
model.save('food_recognition_model.keras')
```

## 4. Run the Web Application

```python
python app.py
```

- The app will start on http://0.0.0.0:4000 by default.
- Open your browser and navigate to http://localhost:4000.

## 🖥️ Usage

The Flask app exposes two routes:

- GET /
    Renders the live camera feed UI.
- POST /predict
    Accepts a JSON payload:

```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ..."
}
```

Returns:

```json
{
  "label": "Samosa",
  "confidence": 0.92
}
```

## 🔧 Configuration in app.py

1. Imports — ensure you have:

```python
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, request, jsonify
```

2. Load the model:

```python
model = load_model('food_recognition_model.keras')
```

3. Define class labels mapping (index → label):

```python
class_labels = {
    0: 'Bhaji Pav',
    1: 'Dabeli',
    2: 'DoubleCheesePizza',
    3: 'Paneer Tikka Sandwich',
    4: 'Samosa',
    5: 'Vada Pav',
    6: 'Wheat Sandwich',
    7: 'puff'
}
```


4. Confidence threshold (adjust as needed):

```python
CONFIDENCE_THRESHOLD = 0.8
```

## ⚠️ Important
The order of class_labels must exactly match the order your model was trained on—i.e. the same ordering as

```python
list(train_generator.class_indices.keys())
```

in your training notebook. If these differ, your model will predict correct indices but map them to the wrong names.


