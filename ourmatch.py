import cv2
import numpy as np
class myBFMatcher:
    def knnMatch(self, des1, des2, k=2):
        matches = []
        for i in range(len(des1)):
            distances = np.linalg.norm(des1[i] - des2, axis=1)
            indices = np.argsort(distances)[:k]
            match1 = cv2.DMatch(_queryIdx=i, _trainIdx=indices[0], _distance=distances[indices[0]])
            match2 = cv2.DMatch(_queryIdx=i, _trainIdx=indices[1], _distance=distances[indices[1]])
            matches.append([match1, match2])
        return matches

def our_RANSAC(good_matches,kp1,kp2):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # RANSAC
    best_H = None
    inliers = 0

    for k in range(1000):
        # 随机选择四个匹配点
        indices = np.random.choice(len(good_matches), 4, replace=False)
        src_sample = src_pts[indices]
        dst_sample = dst_pts[indices]

        # 计算单应性矩阵
        H, k = cv2.findHomography(src_sample, dst_sample)

        # 计算内点数量
        transformed_pts = cv2.perspectiveTransform(src_pts, H)
        distance = np.sqrt(np.sum((transformed_pts - dst_pts) ** 2, axis=2))
        inlier_count = np.sum(distance < 5.0)

        # 保存最好的单应性矩阵和内点数量
        if inlier_count > inliers:
            best_H = H
            inliers = inlier_count

    # 使用最好的单应性矩阵进行筛选
    transformed_pts = cv2.perspectiveTransform(src_pts, best_H)
    distance = np.sqrt(np.sum((transformed_pts - dst_pts) ** 2, axis=2))
    matches_mask = distance < 5.0
    return matches_mask

