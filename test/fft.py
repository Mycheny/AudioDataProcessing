# -*- coding: utf-8 -*- 
# @File fft.py
# @Time 2020/11/11 13:52
# @Author wcy
# @Software: PyCharm
# @Site
import cv2
import numpy as np
from matplotlib import pyplot as plt

def main1():
    img = cv2.imread('src.jpg', 0)
    f = np.fft.fft2(img)  # 傅里叶变换得到频谱，一般来说，低频分量模值最大
    fshift = np.fft.fftshift(f)  # 平移频谱到图像中央
    fshift = np.fft.ifftshift(fshift)  # 平移频谱到图像中央
    fshift = fshift + np.random.random(fshift.shape)
    # 将频谱转换成db
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    plt.subplot(321)
    plt.imshow(img, cmap='gray')
    plt.title('Input Image')
    plt.xticks([0, magnitude_spectrum.shape[1]], [0, magnitude_spectrum.shape[1]])
    plt.yticks([magnitude_spectrum.shape[0], 0], [0, magnitude_spectrum.shape[0]])
    plt.subplot(322)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.xticks([0, magnitude_spectrum.shape[1]], [0, magnitude_spectrum.shape[1]])
    plt.yticks([magnitude_spectrum.shape[0], 0], [0, magnitude_spectrum.shape[0]])

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    # 将低频区去掉一部分，只剩下高频区， 换源出来的就只剩下轮廓（高频信号）
    region = int((ccol + crow) / 4)
    # fshift[crow - region:crow + region, ccol - region:ccol + region] = 0
    # 平移逆变换
    f_ishift = np.fft.ifftshift(fshift)
    # 傅里叶反变换
    img_back = np.fft.ifft2(f_ishift)
    # 取绝对值
    img_back = np.abs(img_back)
    plt.subplot(323)
    plt.imshow(img, cmap='gray')
    plt.title('Input Image')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(324)
    plt.imshow(img_back, cmap='gray')
    plt.title('Image after HPF')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(325)
    plt.imshow(img_back)
    plt.title('Result in JET')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # fft in cv2
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()


def main2():
    import numpy as np
    import matplotlib.pyplot as plt
    fs = 10
    ts = 1 / fs
    t = np.arange(-5, 5, ts)  # 生成时间序列，采样间隔0.1s
    k = np.arange(t.size)  # DFT的自变量
    N = t.size  # DFT的点数量
    x = np.zeros_like(t)  # 生成一个与t相同结构，内容为0的np.array
    x[40:60] = 1  # 设置信号的方波，范围是40-60
    f = np.fft.fft(x)
    y = np.fft.fftshift(f)  # 先np.fft.fft进行fft计算，这时的y是【0,fs】频率相对应的
    # 调用np.fft.fftshift将y变为与频率范围【-fs/2,fs/2】对应，就是将大于fs/2的部分放到
    # -fs/2到0之间,然后绘图的时候将用频率是f=(k*fs/N-fs/2),将频率变为【-fs/2,fs/2】之间
    yf = np.abs(y)  # 计算频率域的振幅
    # plt.rcParams["font.sans - serif"]=["SimHei"]  # 中文乱码处理
    plt.rcParams["axes.unicode_minus"]=False
    plt.subplot(311)
    plt.plot(t, x)
    plt.title("原始方波信号")
    plt.legend(("采样频率fs=10"))
    plt.subplot(312)
    plt.title("方波信号经过fft变换后的频谱")
    f = fs * k / N - fs / 2  # 计算频率
    plt.plot(f, yf)
    iy = np.abs(np.fft.ifft(y))  # 注意这里进行ifft的y要是fft计算出的y经过np.fft.fftshift的
    # 否则会显示错误的结果,y是没有经过np.abs的，y是个复数，计算结果iy是个实数
    plt.subplot(313)
    plt.title("方波fft的ifft")
    plt.plot(t, iy)
    plt.tight_layout()  # 显示完整图片
    plt.show()
    plt.close()


if __name__ == '__main__':
   main2()