import cv2
import numpy as np

def main():
    video = cv2.VideoCapture("../bin/data/traffic.avi")

    ret, gray_ref = video.read()
    if not ret:
        video.release()
        return
    if gray_ref.shape[2] > 1:
        gray_ref = cv2.cvtColor(gray_ref, cv2.COLOR_BGR2GRAY)

    point_ref = cv2.goodFeaturesToTrack(gray_ref, 2000, 0.01, 10)
    if len(point_ref) < 4:
        video.release()
        return

    while True:
        ret, image = video.read()
        if not ret:
            break
        if image.shape[2] > 1:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        point, m_status, err = cv2.calcOpticalFlowPyrLK(gray_ref, gray, point_ref, None)
        H, mask = cv2.findHomography(np.asarray(point, dtype=np.float32), np.asarray(point, dtype=np.float32), cv2.RANSAC)
        warp = cv2.warpPerspective(image, H, (image.shape[1], image.shape[0]))

        p1 = np.squeeze(point_ref, axis=1)
        p2 = np.squeeze(point, axis=1)
        for i in range(len(point_ref)):
            if mask[i][0]:
                cv2.line(image, (p1[i][0], p1[i][1]), (p2[i][0], p2[i][1]), (0, 0, 255))
            else:
                cv2.line(image, (p1[i][0], p1[i][1]), (p2[i][0], p2[i][1]), (0, 127, 0))

        image = np.hstack((image, warp))
        cv2.imshow("3DV Tutorial: Video Stabilization", image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

if __name__ == "__main__":
    main()