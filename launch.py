# launch.py
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from flask_cors import CORS


img1 = None
img2 = None

def process_image(image_file, scale=1.0, rotate=0, dx=0, dy=0, illu=0,save_path=None):
    # # 读取图像

    img_data = np.frombuffer(image_file.read(), np.uint8)
    # # 加载图像

    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    # 缩放

    img = cv2.resize(img, (int(350 * float(scale)),int(350*float(scale)) ))
    # 旋转

    rows, cols, _ = img.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), float(rotate), 1)
    img = cv2.warpAffine(img, rotation_matrix, (cols, rows))

    # 平移
    translation_matrix = np.float32([[1, 0, float(dx)], [0, 1, float(dy)]])
    img = cv2.warpAffine(img, translation_matrix, (cols, rows))

    # 光照变换
    img = cv2.convertScaleAbs(img, alpha=1, beta=int(illu))

    if save_path:
        cv2.imwrite(save_path, img)

    return img

app = Flask(__name__)
CORS(app)


@app.route('/')
def index():
    return render_template('index.html')
params = {}

@app.route('/upload_params', methods=['POST'])
def upload_params():
    global params
    params = request.json  # 参数是以JSON格式发送的
    return jsonify({'status': 'success'})

measures = {}
@app.route('/upload_measures', methods=['POST'])
def upload_measures():
    global measures
    measures = request.json
    return jsonify({'status': 'success'})

@app.route('/match_images', methods=['POST'])
def match_images():
    # 获取参数
    global params
    img1_file = request.files.get('image1')
    img2_file = request.files.get('image2')
    print(params)
    # 调用图像处理方法
    global img1,img2
    img1 = process_image(img1_file, scale=params.get('param1','1.0'), rotate=params.get('param2','0'),
                                  dx=params.get('param3','0'), dy=params.get('param4','0'), illu=params.get('param5','0'),save_path='static/img1_processed.jpg')
    img2 = process_image(img2_file, scale=params.get('param6','1.0'), rotate=params.get('param7','0'),
                                  dx=params.get('param8','0'), dy=params.get('param9','0'), illu=params.get('param10','0'),save_path='static/img2_processed.jpg')

    processed_img1_path = 'static/img1_processed.jpg'
    processed_img2_path = 'static/img2_processed.jpg'

    global measures

    # 图像特征提取及匹配
    if measures.get('matchingMethod1','SIFT') == 'SIFT':
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

    if measures.get('matchingMethod2','BF') == 'BF':
        ma_obj = cv2.BFMatcher()
        print(measures.get('matchingMethod2','BF'))
    elif measures.get('matchingMethod2','BF') == 'FLANN':
        ma_obj = cv2.FlannBasedMatcher()
        print(measures.get('matchingMethod2', 'BF'))

    matches = ma_obj.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < float(measures.get('parameter','0.75')) * n.distance:
            good_matches.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    matches_mask = mask.ravel().tolist()

    ans_matches = [match for i, match in enumerate(good_matches) if matches_mask[i]]

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, ans_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 保存匹配后的图像
    result_image_path = 'static/matched_image.jpg'  # 保存路径，可以根据需要修改
    cv2.imwrite(result_image_path, img3)

    # 返回匹配结果图像路径
    return jsonify({'img1_path': processed_img1_path,
                    'img2_path': processed_img2_path,
                    'result_image_path': result_image_path})

if __name__ == '__main__':
    app.run(debug=True)
