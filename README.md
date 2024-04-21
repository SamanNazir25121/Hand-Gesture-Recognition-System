# Hand Gesture Recognition System

## Overview

This project implements a real-time hand gesture recognition system using Python, TensorFlow/Keras, MediaPipe, and Flask. It detects and tracks hand landmarks in video frames, extracts features from the detected landmarks, and uses a pre-trained machine learning model to predict American Sign Language (ASL) gestures. The system provides accurate gesture recognition and streams the results as a real-time video feed over HTTP using the Flask web framework.

## Technologies Used

- **Python**: Core programming language for implementing the hand gesture recognition system.
- **TensorFlow/Keras**: Machine learning library used to load, compile, and make predictions with pre-trained ASL recognition models.
- **MediaPipe**: Library for hand tracking and landmark detection in video frames.
- **Flask**: Web framework for serving the hand gesture recognition functionality as a real-time video stream.
- **Java (Frontend)**: Utilized for user interaction and frontend development.

## Components

### Python Backend
- **Hand Tracking**: Utilizes the MediaPipe library to detect and track hand landmarks in real-time video frames.
- **Feature Extraction**: Extracts hand landmarks and formats them as input for the machine learning model.
- **Machine Learning Model**: Loads a pre-trained ASL recognition model and makes predictions about the sign language gestures being performed.
- **Visualization**: Visualizes hand landmarks and predicted gestures on video frames using OpenCV.

### Flask Web Server
- **Real-Time Video Streaming**: Serves the hand gesture recognition functionality as a real-time video stream over HTTP.
- **Endpoint Definition**: Defines an endpoint (`/camera_feed_with_predictions`) that continuously streams video frames with predicted gestures to clients.
- **Integration with Frontend**: Can be integrated with a Java frontend for user interaction and visualization.

### Java Frontend
- **User Interaction**: Provides a user-friendly interface for interacting with the hand gesture recognition system.
- **Integration with Backend**: Communicates with the Flask backend to receive real-time video streams and display gesture predictions.

## Setup and Usage

1. **Install Dependencies**: Install the required Python libraries (TensorFlow, MediaPipe, Flask) and Java dependencies for the frontend.
2. **Download Pre-trained Model**: Download the pre-trained ASL recognition model model.json`, `model_checkpoint.h5`).
3. **Run Backend**: Start the Flask web server to serve the hand gesture recognition functionality.
4. **Run Frontend**: Launch the Java frontend application for user interaction and visualization.

  
