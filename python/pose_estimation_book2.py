import cv2
import numpy as np

def main():
    f = 1000.0
    cx = 320.0
    cy = 240.0
    min_inlier_num = 100

    video = cv2.VideoCapture("../bin/data/blais.mp4")
    cover = cv2.imread("../bin/data/blais.jpg")

    fdetector = cv2.ORB_create()   
    fmatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    obj_keypoint, obj_descriptor = fdetector.detectAndCompute(cover, None)

    box_lower = np.array([[30, 145, 0],
                          [30, 200, 0],
                          [200, 200, 0],
                          [200, 145, 0]], dtype=np.float64)
    box_upper = np.array([[30, 145, -50],
                          [30, 200, -50],
                          [200, 200, -50],
                          [200, 145, -50]], dtype=np.float64)

    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1.0]])
    dist_coeff = np.zeros((5, 1))

    while True:
        ret, image = video.read()
        if not ret:
            break
        
        img_keypoint, img_descriptor = fdetector.detectAndCompute(image, None)
        if img_keypoint is None or img_descriptor is None:
            continue
        matches = fmatcher.match(img_descriptor, obj_descriptor)
        if len(matches) < min_inlier_num:
            continue
        
        obj_points = []
        obj_project = []
        img_points = []
        for m in matches:
            obj_points.append(obj_keypoint[m.trainIdx].pt + (1.0,))
            obj_project.append(obj_keypoint[m.trainIdx].pt)
            img_points.append(img_keypoint[m.queryIdx].pt)

        try:
            ret, rvec, tvec, inlier = cv2.solvePnPRansac(
                np.array(obj_points), np.array(img_points), K, dist_coeff, None, None, False, 500, 2, 0.99, None)
        except:
            continue

        inlier_mask = [0]*len(matches)
        if inlier is not None:
            for i in inlier:
                inlier_mask[i[0]] = 1
        else:
            continue
        image_result = cv2.drawMatches(image, img_keypoint, cover, obj_keypoint, 
            matches, None, (0, 0, 255), (0, 127, 0), inlier_mask)

        if len(inlier) > min_inlier_num:
            obj_inlier = []
            img_inlier = []
            for i in inlier:
                if inlier_mask[i[0]]:
                    obj_inlier.append(obj_points[i[0]])
                    img_inlier.append(img_points[i[0]])

            obj_inlier = [np.asarray(obj_inlier, dtype=np.float32)]
            img_inlier = [np.asarray(img_inlier, dtype=np.float32)]
            flags = cv2.CALIB_FIX_ASPECT_RATIO|\
                cv2.CALIB_FIX_PRINCIPAL_POINT|\
                cv2.CALIB_ZERO_TANGENT_DIST|\
                cv2.CALIB_USE_INTRINSIC_GUESS
            h, w = image.shape[:2]
            ret, K, dist_coeff, rvecs, tvecs = cv2.calibrateCamera(
                obj_inlier, img_inlier, (w, h), K, dist_coeff, flags=flags)
            rvec = rvecs[0]
            tvec = tvecs[0]

            line_lower, _ = cv2.projectPoints(box_lower, rvec, tvec, K, dist_coeff)
            line_upper, _ = cv2.projectPoints(box_upper, rvec, tvec, K, dist_coeff)
            line_lower = np.squeeze(line_lower, axis=1)
            line_upper = np.squeeze(line_upper, axis=1)
            cv2.polylines(image_result, [np.int32(line_lower)], True, (255, 0, 0), 2)
            for i in range(line_lower.shape[0]):
                cv2.line(image_result, 
                    (int(line_lower[i][0]), int(line_lower[i][1])), 
                    (int(line_upper[i][0]), int(line_upper[i][1])), 
                    (0, 255, 0), 2)
            cv2.polylines(image_result, [np.int32(line_upper)], True, (0, 0, 255), 2)

        info = "Inliers: %d (%d%%), Focal Length: %.0f" % (
            len(inlier), 100*len(inlier)/len(matches), K[0, 0])
        cv2.putText(image_result, info, (5, 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
        cv2.imshow("3DV Tutorial: Pose Estimation (Book)", image_result)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    
    video.release()

if __name__ == "__main__":
    main()