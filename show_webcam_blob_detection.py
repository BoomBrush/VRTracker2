# Standard imports
import cv2, time
import numpy as np
 
params = cv2.SimpleBlobDetector_Params()
params.blobColor = 255
params.minArea = 75
#params.maxArea = xxx
#params.filterByCircularity = 1
#params.filterByInertia = 1
detector = cv2.SimpleBlobDetector_create(params)

cap = cv2.VideoCapture(0)

#cap.set(3,320)
#cap.set(4,240)
#cap.set(5,120)
cap.set(3,640)
cap.set(4,480)
cap.set(5,60)

def get_ir_position():
    before = time.time()
    ret, im = cap.read()

    keypoints = detector.detect(im)
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    if len(keypoints) == 1:
        point = keypoints[0]
        return True, point.pt[0], point.pt[1], point.size, im_with_keypoints
    else:
        return False, 0, 0, 0, im_with_keypoints

# Capture the first blob, keep looping until you see it
while True:
    one_blob, x, y, size, frame = get_ir_position()

    if one_blob:
        break

    cv2.imshow("Keypoints", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

x_offset = x
y_offset = y
size_offset = size

while True:
    one_blob, x, y, size, frame = get_ir_position()

    if one_blob:
        print(x, y, size, x_offset, y_offset)

    cv2.imshow("Keypoints", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
