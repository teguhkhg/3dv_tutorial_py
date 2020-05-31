import cv2
import numpy as np

def main(show_rectify = True):
    K = np.array([[432.7390364738057, 0, 476.0614994349778],
                  [0, 431.2395555913084, 288.7602152621297],
                  [0, 0, 1]])
    dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 
        0.0001149909868437517, -0.01803978785585194])
    
    video = cv2.VideoCapture("../bin/data/chessboard.avi")

    map1, map2 = None, None
    while True:
        ret, image = video.read()
        if not ret:
            break

        info = "Original"

        height, width = image.shape[:2]
        if show_rectify:
            if map1 is None or map2 is None:
                map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeff, None, None, (width, height), cv2.CV_32FC1)
            image = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
            info = "Rectified"
        cv2.putText(image, info, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

        cv2.imshow("3DV Tutorial: Distortion Correction", image)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_rectify = not show_rectify
        elif key == ord('p'):
            key = cv2.waitKey()
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_rectify = not show_rectify

    video.release()

if __name__ == "__main__":
    main()
