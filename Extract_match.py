import cv2
import numpy as np
import sift as sf
import matplotlib.pyplot as plt
import ourmatch
plt.rcParams['font.sans-serif'] = ['SimHei']

class EM_Sys():
    def __init__(self):
        self.img1=None
        self.img2=None
        self.kp1=None
        self.kp2=None
        self.des1=None
        self.des2=None
    def update_img(self,img1,img2):
        self.img1=np.array(img1)
        self.img2 = np.array(img2)

    def extract(self,ex_method='sift'):
        if ex_method=='sift':
            ex_obj = cv2.SIFT_create()
        else:
            return
        self.kp1, self.des1 = ex_obj.detectAndCompute(self.img1, None)
        self.kp2, self.des2 = ex_obj.detectAndCompute(self.img2, None)

    def match(self,ma_method='BF',lowe_ra=0.75,Use_RANSANC=True):
        M=np.array([])
        good_matches = []
        if ma_method=='BF':
            ma_obj = cv2.BFMatcher()
            #ma_obj = ourmatch.myBFMatcher()
        elif ma_method=='FLANN':
            ma_obj=cv2.FlannBasedMatcher()
        else:
            return M,good_matches
        matches = ma_obj.knnMatch(self.des1, self.des2, k=2)
        # 使用Lowe's ratio测试来筛选好的匹配点
        for m, n in matches:
            if m.distance < lowe_ra * n.distance:
                good_matches.append(m)
        if Use_RANSANC:
            src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matches_mask = mask.ravel().tolist()
            #matches_mask = ourmatch.our_RANSAC(good_matches,self.kp1,self.kp2)
            # 根据RANSAC结果筛选匹配点
            good_matches = [match for i, match in enumerate(good_matches) if matches_mask[i]]
        return M,good_matches

    def Draw_img_ans(self,good_matches,flag=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS):
        img3 = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # 显示结果图像
        cv2.imshow('Matches', img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def show_gaussi(self):
        print(self.img1)
        gauss_img, dog_img = sf.ret_gauss(self.img1)
        plt.figure(1)
        plt.title('高斯金字塔')
        plt.xticks([])
        plt.yticks([])
        for i in range(len(gauss_img[0])):
            plt.subplot(2, 3, i + 1)
            plt.imshow(gauss_img[0][i].astype(np.uint8), cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.figure(2)
        plt.title('高斯差分金字塔')
        plt.xticks([])
        plt.yticks([])
        for i in range(len(dog_img[0])):
            plt.subplot(2, 3, i + 1)
            plt.imshow(dog_img[0][i].astype(np.uint8), cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.show()