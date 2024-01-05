import cv2
import numpy as np
import time
import sift as sf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
# 加载图像

time1=time.time()

img1 = cv2.imread('img/port1.jpg')  # 查询图像
img2 = cv2.imread('img/port2.jpg')  # 训练图像
width = 800  # 新宽度
height = 600  # 新高度
# 调整图像大小
img1 = cv2.resize(img1, (width, height))
img2 = cv2.resize(img2, (width, 600))
time2=time.time()

print('图片载入时间:%fs'%(time2-time1))

# gauss_img ,dog_img= sf.ret_gauss(img1)
# plt.figure(1)
# plt.title('高斯金字塔')
# plt.xticks([])
# plt.yticks([])
# for i in range(len(gauss_img[0])):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(gauss_img[0][i].astype(np.uint8),cmap='gray')
#     plt.xticks([])
#     plt.yticks([])
#
# plt.figure(2)
# plt.title('高斯差分金字塔')
# plt.xticks([])
# plt.yticks([])
# for i in range(len(dog_img[0])):
#     plt.subplot(2, 3, i + 1)
#     plt.imshow(dog_img[0][i].astype(np.uint8), cmap='gray')
#     plt.xticks([])
#     plt.yticks([])
# plt.show()
# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 计算两幅图像的关键点和描述符
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
time3=time.time()
print('特征提取时间:%fs'%(time3-time2))
# img_sift = cv2.drawKeypoints(img1, kp1, None, color=(0,255,0))
# cv2.imshow('12',img_sift)
# 创建暴力匹配器
bf = cv2.BFMatcher()
#bf = cv2.FlannBasedMatcher()
# 进行特征匹配
matches = bf.knnMatch(des1, des2, k=2)
time4=time.time()
print('特征匹配时间:%fs'%(time4-time3))


# 使用Lowe's ratio测试来筛选好的匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
time5=time.time()
print('Lowes ratio筛选时间:%fs'%(time5-time4))

    # 使用RANSAC算法去除误匹配点
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
matches_mask = mask.ravel().tolist()

# 根据RANSAC结果筛选匹配点

ans_matches=[match for i,match in enumerate(good_matches) if matches_mask[i]]
time6=time.time()
print('RANSAC算法时间:%fs'%(time6-time5))

# 绘制匹配点
#img3 = cv2.drawMatches(img1, kp1, img2, kp2, ans_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

img4 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

time7=time.time()
print('结果展示时间:%fs'%(time7-time6))

# 显示结果图像
#cv2.imshow('Matches', img3)
cv2.imshow('M', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
# print('运行完毕')