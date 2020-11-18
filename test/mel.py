# -*- coding: utf-8 -*-
# @File mel.py
# @Time 2020/11/18 10:04
# @Author wcy
# @Software: PyCharm
# @Site
import os
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import librosa


def get_random_wave(frequency, sr=8000):
    """
    返回对应频率的二维波形
    :param sr:
    :param frequency: 频率
    :return:
    """
    sampling_rate = sr  # 一个周期采样数（采样率）
    show_T = 2  # 显示多少秒对应频率的波形
    sample = sampling_rate * show_T  # 总采样数
    angular_frequency = 2 * np.pi * frequency  # 角频率
    initial_phase = 0  # 初相
    amplitude = 1  # 振幅
    t = np.linspace(0, show_T, sample)  # 时间数组
    t = t[:-1]  # t[-1] 是另一个周期的起点需要去掉
    y = amplitude * np.sin(angular_frequency * t + initial_phase)
    # plt.plot(t, y)
    # plt.show()
    return y


def mel1(y, sr=8000):
    y = y[:sr*2]
    y11 = get_random_wave(1024, sr=sr)
    y12 = get_random_wave(3000, sr=sr)
    y1 = y11+y12
    y21 = get_random_wave(512, sr=sr)
    y22 = get_random_wave(1500, sr=sr)
    y2 = y21 + y22
    y = np.concatenate((y1, y2))
    # 提取 mel spectrogram feature
    melspec = librosa.stft(y, n_fft=1024, hop_length=512)

    # melspec = librosa.feature.melspectrogram(y1, sr, n_fft=1024, hop_length=512, n_mels=2048)
    logmelspec = librosa.power_to_db(melspec)  # 转换为对数刻度
    # 绘制 mel 频谱图
    plt.figure()
    librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='hz')
    # librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')  # 右边的色度条
    plt.title('Beat wavform')
    plt.show()


def mel2(y):
    X = librosa.stft(audio_datas)
    X = np.abs(X)
    melW = librosa.filters.mel(sr=sr, n_fft=X.shape[1] * 2, n_mels=40, fmin=0., fmax=22100)
    melW /= np.max(melW, axis=-1)[:, None]
    melX = np.dot(X, melW.T)
    print()


if __name__ == '__main__':
    root = r"E:\DATA\坐席辅助项目\坐席辅助公积金的录音下载文件\录音下载"
    audio_file = os.path.join(root, "8679594520200915122118_18285386619.wav")
    audio_datas, sr = librosa.load(audio_file, sr=8000)
    mel1(audio_datas, sr)
