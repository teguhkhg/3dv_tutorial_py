import cv2
import numpy as np

def main():
    K = np.array([[432.7390364738057, 0, 476.0614994349778],
                  [0, 431.2395555913084, 288.7602152621297],
                  [0, 0, 1]])
    dist_coeff = np.array([-0.2852754904152874, 0.1016466459919075, -0.0004420196146339175, 
        0.0001149909868437517, -0.01803978785585194])

    board_pattern = (10, 7)
    board_cellsize = 0.025
    video = cv2.VideoCapture("../bin/data/chessboard.avi")

    box_lower = np.array([[4*board_cellsize, 2*board_cellsize, 0],
                          [5*board_cellsize, 2*board_cellsize, 0],
                          [5*board_cellsize, 4*board_cellsize, 0],
                          [4*board_cellsize, 4*board_cellsize, 0]])
    box_upper = np.array([[4*board_cellsize, 2*board_cellsize, -board_cellsize],
                          [5*board_cellsize, 2*board_cellsize, -board_cellsize],
                          [5*board_cellsize, 4*board_cellsize, -board_cellsize],
                          [4*board_cellsize, 4*board_cellsize,-board_cellsize]])

    obj_points = np.zeros((board_pattern[0]*board_pattern[1], 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2)
    obj_points *= board_cellsize

    while True:
        ret, image = video.read()
        if not ret:
            break

        success, img_points = cv2.findChessboardCorners(image, board_pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK)
        if success:
            _, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist_coeff)

            line_lower, _ = cv2.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv2.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
            line_lower = np.squeeze(line_lower, axis=1)
            line_upper = np.squeeze(line_upper, axis=1)
            cv2.polylines(image, [np.int32(line_lower)], True, (255, 0, 0), 2)
            for i in range(line_lower.shape[0]):
                cv2.line(image, 
                    (int(line_lower[i][0]), int(line_lower[i][1])), 
                    (int(line_upper[i][0]), int(line_upper[i][1])), 
                    (0, 255, 0), 2)
            cv2.polylines(image, [np.int32(line_upper)], True, (0, 0, 255), 2)
            
            R, _ = cv2.Rodrigues(rvec)
            p = -R.T.dot(tvec)
            info = "XYZ: [%.3f, %.3f, %.3f]" % (p[0], p[1], p[2])
            cv2.putText(image, info, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        
        cv2.imshow("3DV Tutorial: Pose Estimation (Chessboard)", image)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    video.release()

if __name__ == "__main__":
    main()