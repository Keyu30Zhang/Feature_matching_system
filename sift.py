from numpy import  array,  log, sqrt, zeros,round
from cv2 import resize, GaussianBlur, subtract, INTER_LINEAR, INTER_NEAREST,cvtColor,COLOR_BGR2GRAY

def generateBaseImage(image, sigma, assumed_blur):
    #高斯滤波
    image = resize(image, (0, 0), fx=2, fy=2, interpolation=INTER_LINEAR)
    sigma_diff = sqrt(max((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)  # the image blur is now sigma instead of assumed_blur


#计算图像金字塔层数,计算公式为log2(min(M,N))-1;M,N为图像宽和长
def computeNumberOfOctaves(image_shape):
    return int(round(log(min(image_shape)) / log(2) - 1))

def generateGaussianKernels(sigma, num_intervals):

    num_images_per_octave = num_intervals + 3
    k = 2 ** (1. / num_intervals)
    gaussian_kernels = zeros(num_images_per_octave)
    gaussian_kernels[0] = sigma

    for image_index in range(1, num_images_per_octave):
        sigma_previous = (k ** (image_index - 1)) * sigma
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels

def generateGaussianImages(image, num_octaves, gaussian_kernels):
    gaussian_images = []

    for octave_index in range(num_octaves):
        gaussian_images_in_octave = []
        gaussian_images_in_octave.append(image)  # first image in octave already has the correct blur
        for gaussian_kernel in gaussian_kernels[1:]:
            image = GaussianBlur(image, (0, 0), sigmaX=gaussian_kernel, sigmaY=gaussian_kernel)
            gaussian_images_in_octave.append(image)
        gaussian_images.append(gaussian_images_in_octave)
        octave_base = gaussian_images_in_octave[-3]
        image = resize(octave_base, (int(octave_base.shape[1] / 2), int(octave_base.shape[0] / 2)), interpolation=INTER_NEAREST)
    return gaussian_images

def generateDoGImages(gaussian_images):
    dog_images = []

    for gaussian_images_in_octave in gaussian_images:
        dog_images_in_octave = []
        for first_image, second_image in zip(gaussian_images_in_octave, gaussian_images_in_octave[1:]):
            dog_images_in_octave.append(subtract(second_image, first_image))  # ordinary subtraction will not work because the images are unsigned integers
        dog_images.append(dog_images_in_octave)
    return dog_images


def ret_gauss(image, sigma=1.6, num_intervals=3, assumed_blur=0.5):
    image = cvtColor(image, COLOR_BGR2GRAY)
    image = image.astype('float32')
    base_image = generateBaseImage(image, sigma, assumed_blur)  #高斯模糊,去噪
    num_octaves = computeNumberOfOctaves(base_image.shape)   #计算图像金字塔层数
    gaussian_kernels = generateGaussianKernels(sigma, num_intervals) #创建高斯核
    gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)#模糊和降采样
    dog_images = generateDoGImages(gaussian_images)#形成高斯差分金字塔
    return gaussian_images, dog_images
