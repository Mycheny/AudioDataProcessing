# -*- coding: utf-8 -*- 
# @File cv2_to_pil.py
# @Time 2020/11/10 15:01
# @Author wcy
# @Software: PyCharm
# @Site
import random
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from scipy.fftpack import fft, ifft, fft2, ifft2


def decode(ori_path, img_path, res_path, alpha=5.0):
    ori = cv2.imread(ori_path)
    img = cv2.imread(img_path)
    ori_f = np.fft.fft2(ori)
    img_f = np.fft.fft2(img)
    height, width = ori.shape[0], ori.shape[1]
    watermark = (ori_f - img_f) / alpha
    watermark = np.real(watermark)
    res = np.zeros(watermark.shape)
    random.seed(height + width)
    x = list(range(int(height / 2)))
    y = list(range(width))
    random.shuffle(x)
    random.shuffle(y)
    for i in range(int(height / 2)):
        for j in range(width):
            res[x[i]][y[j]] = watermark[i][j]
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def encode(img_path, wm_path, res_path, alpha=5.0):
    img = cv2.imread(img_path, 0)
    img = np.tile(np.expand_dims(np.tile(np.arange(0, 100, 10), (1, )), axis=1), (1, 100))
    img_f = np.fft.fft2(img)
    # cv2.imshow("img1", img_f.astype(np.uint8))
    # cv2.imshow("img2", img_f.real)
    # cv2.imshow("img3", img_f.imag)
    # cv2.imshow("img4", img)
    # a = img_f.astype(np.uint8)
    # b = img_f.real
    # c = img_f.imag
    # cv2.waitKey(0)

    height, width, channel = np.shape(img)
    watermark = cv2.imread(wm_path)
    wm_height, wm_width = watermark.shape[0], watermark.shape[1]
    x, y = list(range(int(height / 2))), list(range(width))
    random.seed(height + width)
    random.shuffle(x)
    random.shuffle(y)
    tmp = np.zeros(img.shape)
    for i in range(int(height / 2)):
        for j in range(width):
            if x[i] < wm_height and y[j] < wm_width:
                tmp[i][j] = watermark[x[i]][y[j]]
                tmp[height - 1 - i][width - 1 - j] = tmp[i][j]

    res_f = img_f + alpha * tmp
    res = np.fft.ifft2(res_f)
    res = np.real(res)
    cv2.imwrite(res_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def show(x, y):
    y_min_index = np.argmin(y)
    y_max_index = np.argmax(y)
    y_min = x[y_min_index]
    y_max = x[y_max_index]
    show_min = f"{y_min_index}  {str(float(y[y_min_index])).zfill(5)}"
    show_max = f"{y_max_index}  {str(float(y[y_max_index])).zfill(5)}"
    # 以●绘制最大值点和最小值点的位置
    plt.plot(y_min, y[y_min_index], 'ko')
    plt.plot(y_max, y[y_max_index], 'ko')
    plt.annotate(show_min, xy=(y_min, y[y_min_index]), xytext=(y_min, y[y_min_index]))
    plt.annotate(show_max, xy=(y_max, y[y_max_index]), xytext=(y_max, y[y_max_index]))
    plt.plot(x, y)
    plt.show()


def test():
    single_sample = 16  # 一个周期采样数单个
    show_T = 1  # 显示多少个周期
    sample = single_sample * show_T  # 采样数
    frequency = 1
    angular_frequency = 2 * np.pi * frequency  # 角频率
    initial_phase = 0  # 初相
    amplitude = 1  # 振幅
    t = np.linspace(0, 1.0, sample)  # 时间数组
    t = np.tile(np.expand_dims(t, axis=1), (1, t.size))
    y = amplitude * np.sin(angular_frequency * t + initial_phase)
    f = np.fft.fft(y)
    real_f = f.real
    imag_f = f.imag
    show(t, y)
    show(np.arange(0, f.shape[0]), real_f)
    show(np.arange(0, f.shape[0]), imag_f)
    print()


def test2():
    def get_image(frequency):
        """
        返回对应频率的二维波形
        :param frequency: 频率
        :return:
        """
        sampling_rate = 8000  # 一个周期采样数（采样率）
        show_T = 2  # 显示多少秒对应频率的波形
        sample = sampling_rate * show_T  # 总采样数
        angular_frequency = 2 * np.pi * frequency  # 角频率
        initial_phase = 0  # 初相
        amplitude = 1  # 振幅
        t = np.linspace(0, show_T, sample)  # 时间数组
        t = t[:-1]  # t[-1] 是另一个周期的起点需要去掉
        t = np.tile(np.expand_dims(t, axis=1), (1, t.size))
        y = amplitude * np.sin(angular_frequency * t + initial_phase)
        return y

    # https://www.zhihu.com/question/25523672
    y_2 = get_image(2)
    y_4 = get_image(4)
    y_6 = get_image(6)
    y = (y_2+y_4.T+y_6)/3
    y = y_2
    f = np.fft.fft2(y)
    f1 = fft2(y)
    real_f = f.real  # 实部就代表所有的偶函数（余弦函数）的成分
    imag_f = f.imag  # 虚部就代表所有奇函数（正弦函数）的成分
    real_f1 = f1.real
    imag_f1 = f1.imag
    fshift = np.fft.fftshift(f)  # 将低频翻转到高频频谱到图像中央
    real_fshift = fshift.real
    imag_fshift = fshift.imag
    print()


if __name__ == '__main__':
    test2()

    filename = "/home/web_site/src/static/upload/admin/12558925.png"
    # img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))
    # img.save("res.png")
    encode("src.jpg", "watermark.png", "res.png")
    # decode("src.png", "res.png", "res1.png")