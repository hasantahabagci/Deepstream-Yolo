import cv2

cap = cv2.VideoCapture("output.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:   # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
