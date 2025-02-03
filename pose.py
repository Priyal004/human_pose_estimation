import cv2
import mediapipe as mp
#Initialize
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
#load
image_path = "C:/Users/LENOVO/Desktop/pose.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#perform
results = pose.process(image_rgb)
#Draw
if results.pose_landmarks:
    print("Pose landwark detected!")
    #Extract
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z:{landmark.z})")
    for landmark in results.pose_landmarks.landmark:
        #Get image
        h, w, c = image.shape
        #COnvert
        cx , cy = int(landmark.x  * w), int(landmark.y * h)
        #Draw
        cv2.circle(image, (cx, cy) , 5, (0, 255, 0), -1)
        #optional
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow("Pose landmarks",image)
    cv2.imshow("Pose drawing", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
