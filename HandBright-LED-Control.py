
import cv2
import mediapipe as mp
import numpy as np
import pyfirmata
import time

# Initialize Arduino
board = pyfirmata.Arduino('COM3')  # Change 'COM3' to your Arduino port
led_pin = board.get_pin('d:9:p')  # PWM pin 9 for LED
it = pyfirmata.util.Iterator(board)
it.start()

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    height, width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get landmarks for thumb tip (4) and index tip (8)
            thumb = hand_landmarks.landmark[4]
            index = hand_landmarks.landmark[8]

            # Convert to pixel coordinates
            thumb_x, thumb_y = int(thumb.x * width), int(thumb.y * height)
            index_x, index_y = int(index.x * width), int(index.y * height)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Calculate distance
            distance = np.hypot(index_x - thumb_x, index_y - thumb_y)

            # Normalize brightness (distance range: ~0 to 150 pixels)
            brightness = np.clip(distance / 150, 0, 1)
            led_pin.write(brightness)  # Send PWM signal

            # Display distance & brightness level
            cv2.putText(frame, f'Brightness: {int(brightness * 100)}%', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking LED Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
board.exit()
