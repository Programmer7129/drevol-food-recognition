# import cv2
# import numpy as np
# import base64
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# from flask import Flask, render_template, Response, jsonify

# # Load the trained Keras model
# model = load_model('food_recognition_model.keras')

# # Define class labels
# class_labels = {
#     0: 'Bhaji Pav', 
#     1: 'Dabeli', 
#     2: 'DoubleCheesePizza',
#     3: 'Paneer Tikka Sandwich',
#     4: 'Samosa',
#     5: 'Vada Pav',
#     6: 'Wheat Sandwich',
#     7: 'puff'
# }

# # Initialize Flask app
# app = Flask(__name__)

# # Initialize camera
# camera = cv2.VideoCapture(0)

# def process_frame():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         # Resize and preprocess frame for the model
#         img = cv2.resize(frame, (224, 224))  # Match model input size
#         img_array = img_to_array(img)
#         img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
#         img_array = img_array / 255.0  # Normalize pixel values

#         # Predict the class
#         predictions = model.predict(img_array)
#         predicted_class_index = np.argmax(predictions[0])
#         predicted_label = class_labels.get(predicted_class_index, "Unknown")

#         # Display prediction on frame
#         cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Encode frame to JPEG format for streaming
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video_feed')
# def video_feed():
#     return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(debug=True, threaded=True)


import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask, render_template, Response, request, jsonify

# Load trained Keras model
model = load_model('food_recognition_model.keras')

# Define class labels
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

# Confidence threshold (adjust if needed)
CONFIDENCE_THRESHOLD = 0.8  # 80% confidence required for valid detection

app = Flask(__name__)

camera = None  # Initialize camera variable
is_streaming = False  # State variable to control streaming

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def process_frame():
    global camera
    while is_streaming:
        success, frame = camera.read()
        if not success:
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        # Resize and preprocess frame for ML model
        img = cv2.resize(frame, (224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

        # Predict food item
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])  # Get confidence score

        # Determine label
        if len(faces) > 0:  # If a face is detected, label as "Unknown"
            predicted_label = "Unknown"
        elif confidence < CONFIDENCE_THRESHOLD:  # If confidence is too low, label as "Unknown"
            predicted_label = "Unknown"
        else:
            predicted_label = class_labels.get(predicted_class_index, "Unknown")

        # Display detected food on frame
        cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 255) if predicted_label == "Unknown" else (0, 255, 0), -1)  # Red for unknown, Green for detected
        cv2.putText(frame, predicted_label, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_camera():
    global camera, is_streaming
    if not is_streaming:
        camera = cv2.VideoCapture(0)
        is_streaming = True
    return jsonify({'status': 'started'})

@app.route('/stop', methods=['POST'])
def stop_camera():
    global camera, is_streaming
    if is_streaming:
        is_streaming = False
        camera.release()
        camera = None
    return jsonify({'status': 'stopped'})

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
