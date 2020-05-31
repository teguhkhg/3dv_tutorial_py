import cv2
import numpy as np

from collections import namedtuple
from math import cos, sin

point3d = namedtuple("world", "x y z")
point2d = namedtuple("image", "x y")

class MouseDrag:
    def __init__(self):
        self.dragged = False
        self.start = point2d(0, 0)
        self.end = point2d(0, 0)

def mouseEventHandler(event, x, y, flags, param):
    if param is None:
        return

    drag = param[0]
    if event == cv2.EVENT_LBUTTONDOWN:
        drag.dragged = True
        drag.start = point2d(x, y)
        drag.end = point2d(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drag.dragged:
            drag.end = point2d(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        if drag.dragged:
            drag.dragged = False
            drag.end = point2d(x, y)

def DEG2RAD(v):
    return v*np.pi/180.0

def Rx(rx):
    return np.array([[1, 0, 0],
                     [0, cos(rx), -sin(rx)],
                     [0, sin(rx), cos(rx)]])

def Ry(ry):
    return np.array([[cos(ry), 0, sin(ry)],
                     [0, 1, 0],
                     [-sin(ry), 0, cos(ry)]])

def Rz(rz):
    return np.array([[cos(rz), -sin(rz), 0],
                     [sin(rz), cos(rz), 0],
                     [0, 0, 1]])

def main():
    f = 810.5
    cx = 480
    cy = 270
    L = 3.31
    window_name = "3DV Tutorial: Object Localization and Measurement"

    cam_ori = point3d(DEG2RAD(-18.7), DEG2RAD(-8.2), DEG2RAD(2.0))
    grid_x = range(-2, 3)
    grid_z = range(5, 35)

    image = cv2.imread("../bin/data/daejeon_station.png")
    drag = MouseDrag()
    param = [drag]
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouseEventHandler, param)

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]])
    Rc = Rz(cam_ori.z).dot(Ry(cam_ori.y)).dot(Rx(cam_ori.x))
    R = Rc.T
    tc = np.array([[0, -L, 0]]).T
    t = -Rc.T.dot(tc)

    for z in grid_z:
        p = K.dot(R.dot(np.array([[grid_x[0], 0, z]]).T) + t)
        q = K.dot(R.dot(np.array([[grid_x[-1], 0, z]]).T) + t)
        p = p/p[2]
        q = q/q[2]
        cv2.line(image, (p[0], p[1]), (q[0], q[1]), (64, 128, 64), 1)

    for x in grid_x:
        p = K.dot(R.dot(np.array([[x, 0, grid_z[0]]]).T) + t)
        q = K.dot(R.dot(np.array([[x, 0, grid_z[-1]]]).T) + t)
        p = p/p[2]
        q = q/q[2]
        cv2.line(image, (p[0], p[1]), (q[0], q[1]), (64, 128, 64), 1)

    while True:
        image_copy = image.copy()
        if drag.end.x > 0 and drag.end.y > 0:
            c = R.T.dot(np.array([[drag.start.x - cx, drag.start.y - cy, f]]).T)
            if c[1] < np.finfo(float).eps:
                continue
            h = R.T.dot(np.array([[drag.end.x - cx, drag.end.y - cy, f]]).T)
            Z = c[2]/c[1]*L
            X = c[0]/c[1]*L
            H = (c[1]/c[2] - h[1]/h[2])*Z

            cv2.line(image_copy, drag.start, drag.end, (0, 0, 255), 2)
            cv2.circle(image_copy, drag.end, 4, (255, 0, 0), -1)
            cv2.circle(image_copy, drag.start, 4, (0, 255, 0), -1)
            cv2.putText(image_copy, "X:%.2f, Z:%.2f, H:%.2f" % (X, Z, H), 
                (drag.start.x-20, drag.start.y+20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        
        cv2.imshow(window_name, image_copy)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

if __name__ == "__main__": 
    main()