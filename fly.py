import cv2
cap = cv2.VideoCapture('rtsp://10.42.0.10:8554/video')
#cap = cv2.VideoCapture('rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4')
print(cap)
ret, frame = cap.read()
while ret:
    ret, frame = cap.read()
    cv2.imshow("current frame", frame)
    cv2.imwrite('frame.jpg', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()