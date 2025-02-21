# HandBright-LED-Control
A Python project using MediaPipe and PyFirmata to control LED brightness with hand gestures. The distance between the index finger and thumb adjusts the LED intensity via PWM signals sent to an Arduino.
Features:
🖐️ Hand tracking with MediaPipe
💡 LED brightness control with Arduino (PWM)
🎥 Real-time gesture recognition using OpenCV
🔌 Simple hardware setup with an LED & resistor
Requirements:
Arduino board 
Python (mediapipe, opencv-python, numpy, pyfirmata)
Webcam for hand tracking

Requirements
Arduino Board (e.g., Arduino Uno)
LED + Resistor (220Ω)
Python Libraries: mediapipe, cv2, numpy, pyfirmata
Arduino connected via USB
Wiring
Arduino Pin	Component
9 (PWM)	LED (Positive Leg)
GND	LED (Negative via 220Ω Resistor)
Install Required Libraries
Run the following commands:

pip install mediapipe opencv-python numpy pyfirmata
