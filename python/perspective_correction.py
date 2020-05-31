import cv2
import numpy as np

def mouseEventHandler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points_src = param[0]
        points_src.append((x, y))
        print("A point (index: %zd) is selected at (%d, %d).\n", len(points_src)-1, x, y)

def main():
    card_size = (450, 250)
    original = cv2.imread("../bin/data/sunglok_desk.jpg")

    points_dst = np.array([[0, 0],
                           [card_size[0], 0],
                           [0, card_size[1]],
                           [card_size[0], card_size[1]]])
    
    points_src = []
    param = [points_src]
    window_name = "3DV Tutorial: Perspective Correction"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouseEventHandler, param)

    while len(points_src) < 4:
        display = original.copy()
        display = cv2.rectangle(display, (10, 10), card_size, (0, 0, 255), 2)
        idx = len(points_src)
        cv2.circle(display, (points_dst[idx, 0]+10, points_dst[idx, 1]+10), 5, (0, 255, 0), -1)
        cv2.imshow(window_name, display)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    if len(points_src) < 4:
        return

    H, mask = cv2.findHomography(np.asarray(points_src), points_dst)
    rectify = cv2.warpPerspective(original, H, card_size)

    cv2.imshow(window_name, rectify)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()