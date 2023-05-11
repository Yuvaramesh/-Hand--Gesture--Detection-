import cv2
import mediapipe as mp

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils 

# Start capturing frames from webcam
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # Convert the frame to RGB color space
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with mediapipe hands module
    with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5) as hands:
        results = hands.process(frame)
        
        # Draw hand landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                # Get the hand posture points
                posture_points = []
                for point in hand_landmarks.landmark:
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])
                    posture_points.append((x, y))
                    
                # Determine hand pose
                thumb = posture_points[4]
                index = posture_points[8]
                middle = posture_points[12]
                ring = posture_points[16]
                pinky = posture_points[20]
                
                # Measure distances between thumb and fingers
                thumb_index_dist = ((thumb[0]-index[0])**2 + (thumb[1]-index[1])**2)**0.5
                thumb_middle_dist = ((thumb[0]-middle[0])**2 + (thumb[1]-middle[1])**2)**0.5
                thumb_ring_dist = ((thumb[0]-ring[0])**2 + (thumb[1]-ring[1])**2)**0.5
                thumb_pinky_dist = ((thumb[0]-pinky[0])**2 + (thumb[1]-pinky[1])**2)**0.5
                
                if thumb_index_dist < 50 and thumb_middle_dist < 50 and thumb_ring_dist < 50 and thumb_pinky_dist < 50:
                    print("Fist")
                elif thumb_index_dist > 100 and thumb_middle_dist > 100 and thumb_ring_dist > 100 and thumb_pinky_dist > 100:
                    print("Open hand")
                elif thumb[0] > index[0] and thumb[0] > middle[0] and thumb[0] > ring[0] and thumb[0] > pinky[0]:
                    print("Thumbs up")
    
    # Show the frame on the screen
    cv2.imshow('Hand Pose Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
