import cv2
def dispkp(img,kp):
    img = cv2.drawKeypoints(img, kp, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT) 
    cv2.namedWindow("Trackers", cv2.WINDOW_NORMAL) 
    cv2.imshow('Trackers', img) 
    cv2.waitKey(1)


def disp(img):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL) 
    cv2.imshow('Image', img) 
    cv2.waitKey(1)