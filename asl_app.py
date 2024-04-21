from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import mediapipe as mp
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)
CORS(app)

# Load ASL model architecture from JSON file
with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the ASL model architecture
ASL_model = model_from_json(loaded_model_json)

# Load ASL model weights
ASL_model.load_weights('model_checkpoint.h5')

# Initialize MediaPipe for hand tracking
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Create hands object for hand tracking
hands = mp_hands.Hands()
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Landmarks 
def draw_styled_landmarks(image, results):
    # Check if hand landmarks are detected in the results
    if results.multi_hand_landmarks:
      # Iterate over each detected hand's landmarks  
      for hand_landmarks in results.multi_hand_landmarks:
         # Draw landmarks and connections on the image 
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

# Extract keypoints
def extract_keypoints(results):
    keypoints_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if hand_landmarks:  # Check if hand landmarks are present
                keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark])
                # Resize keypoints to match the model's input shape
                keypoints_resized = cv2.resize(keypoints, (100, 100), interpolation=cv2.INTER_AREA)
                keypoints_resized_with_channels = np.expand_dims(keypoints_resized, axis=-1)
                keypoints_resized_with_channels = np.repeat(keypoints_resized_with_channels, 3, axis=-1)
                keypoints_list.append(keypoints_resized_with_channels)
            else:
                keypoints_list.append(np.zeros((100, 100, 3)))
    return np.array(keypoints_list)



actions = np.array([ 'Z', 'nothing', 'V', 'W', 'Y',
        'U', 'T', 'F', 'R', 'Q',
        'P', 'J', 'N', 'M','L',
        'K', 'O', 'I', 'H', 'G', 
        'S', 'E', 'C', 'D', 'B', 
        'X','space', 'A', 'del'])

# New detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(3, 480)  # Width (Portrait)
    cap.set(4, 640)  # Height (Portrait)

    sequence = []
    sentence = []
    accuracy = []
    predictions = []
    threshold = 0.8

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
 
            image, results = mediapipe_detection(frame, hands)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            try:
                if len(sequence) == 30 and len(keypoints) > 0:
                    res = ASL_model.predict(keypoints)[0]
                    predicted_class = np.argmax(res)
                    predictions.append(predicted_class)

                    if np.unique(predictions[-10:])[0] == predicted_class and res[predicted_class] > threshold:
                        if len(sentence) > 0:
                            if actions[predicted_class] != sentence[-1]:
                                sentence.append(actions[predicted_class])
                                accuracy.append(str(res[predicted_class] * 100))
                        else:
                            sentence.append(actions[predicted_class])
                            accuracy.append(str(res[predicted_class] * 100))

                    if len(sentence) > 1:
                        sentence = sentence[-1:]
                        accuracy = accuracy[-1:]
                else:
                    sentence = ['nothing']
                    accuracy = ['99.0']
            except Exception as e:
                pass

            prediction_label = ' '.join(sentence) + ' =' + ' '.join(accuracy)+ '%'
            cv2.putText(image, f"Sign: {prediction_label}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', image)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/camera_feed_with_predictions')
def camera_feed_with_predictions():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
