import cv2


cap=cv2.VideoCapture(1)
while True:
    ret,frame=cap.read()
    cv2.imshow('capture',frame)
    if cv2.waitKey(1)==ord('q'):
        print('quit!')
        break

cap.release()
cv2.destroyAllWindows()
