import cv2
import numpy as np

from math import sqrt

def convertLine(line):
    return np.array([line[0], -line[1], -line[0]*line[2] + line[1]*line[3]])

def main():
    ransac_trial = 50
    ransac_n_sample = 2
    ransac_thresh = 3.0
    data_num = 1000
    data_inlier_ratio = 0.5
    data_inlier_noise = 1.0

    truth = np.array([1.0/sqrt(2.0), 1.0/sqrt(2.0), -240.0])
    data = []
    for i in range(data_num):
        if np.random.uniform(0.0, 1.0) < data_inlier_ratio:
            x = np.random.uniform(0.0, 480.0)
            y = (truth[0]*x + truth[2])/-truth[1]
            x += np.random.normal(data_inlier_noise)
            y += np.random.normal(data_inlier_noise)
            data.append((x, y))
        else:
            data.append((np.random.uniform(0.0, 640.0), np.random.uniform(0.0, 480.0)))

    best_score = -1
    for i in range(ransac_trial):
        sample = []
        for j in range(ransac_n_sample):
            index = int(np.random.uniform(0, len(data)))
            sample.append(data[index])
        nnxy = cv2.fitLine(np.asarray(sample), cv2.DIST_L2, 0, 0.01, 0.01)
        line = convertLine(nnxy)

        score = 0
        for j in range(len(data)):
            error = abs(line[0]*data[j][0] + line[1]*data[j][1] + line[2])
            if error < ransac_thresh:
                score += 1

        if score > best_score:
            best_score = score
            best_line = line

    nnxy = cv2.fitLine(np.asarray(data), cv2.DIST_L2, 0, 0.01, 0.01)
    lsm_line = convertLine(nnxy)

    print("* The Truth: %.3f, %.3f, %.3f\n" % (truth[0], truth[1], truth[2]))
    print("* Estimate (RANSAC): %.3f, %.3f, %.3f (Score: %d)\n" % (
        best_line[0], best_line[1], best_line[2], best_score))
    print("* Estimate (LSM): %.3f, %.3f, %.3f\n" % (lsm_line[0], lsm_line[1], lsm_line[2]))

if __name__ == "__main__":
    main()