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
from tqdm import tqdm
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]


def get_random_wave(frequency, sr=8000, amplitude=1, initial_phase=0, show_T=1):
    """
    返回对应频率的二维波形
    :param sr: 采样率
    :param frequency: 频率
    :param initial_phase: 初相
    :param amplitude: 振幅
    :param show_T: 显示多少秒对应频率的波形
    :return:
    """
    sampling_rate = sr  # 一个周期采样数（采样率）
    sample = sampling_rate * show_T  # 总采样数
    if frequency == 0:
        return np.array([amplitude] * (sample - 1), np.float64)
    angular_frequency = 2 * np.pi * frequency  # 角频率
    t = np.linspace(0, show_T, sample)  # 时间数组
    t = t[:-1]  # t[-1] 是另一个周期的起点需要去掉
    y = amplitude * np.cos(angular_frequency * t + initial_phase)
    # plt.plot(t, y)
    # plt.show()
    return y


def get_y(sr=8000):
    y11 = get_random_wave(1024, sr=sr)
    y12 = get_random_wave(3000, sr=sr)
    y1 = y11 + y12
    y21 = get_random_wave(1023, sr=sr)
    y22 = get_random_wave(3001, sr=sr)
    y2 = y21 + y22
    y = np.concatenate((y1, y2))
    return y


def get_y1(sr=8000):
    y = np.zeros_like(get_random_wave(0, sr=sr))
    for i in tqdm(np.arange(1, 5000, 2)):
        y += get_random_wave(i, sr=sr) / i
    return y


def get_y2(*frequencys, sr=8000, amplitude=1, initial_phase=0, show_T=1):
    """
    获取多个频率组合成的一维信号
    :param frequencys: 需要的频率数组
    :param sr: 采样率
    :return: 多个频率组合成的一维信号
    """
    y = np.zeros_like(get_random_wave(0, sr=sr))
    for frequency in frequencys:
        y += get_random_wave(frequency, sr=sr, amplitude=amplitude, initial_phase=initial_phase, show_T=show_T)
    return y


def get_y3(frequencys: list, sr=8000, amplitude=None, initial_phase=None, show_T=1):
    """
    获取多个频率组合成的一维信号
    :param frequencys: 需要的频率数组
    :param sr: 采样率
    :param amplitude:
    :param initial_phase:
    :param show_T:
    :return: 多个频率组合成的一维信号
    """
    if amplitude is None:
        amplitude = [1] * len(frequencys)
    if initial_phase is None:
        initial_phase = [1] * len(frequencys)
    y = np.zeros_like(get_random_wave(0, sr=sr))
    for i, frequency in enumerate(frequencys):
        y += get_random_wave(frequency, sr=sr, amplitude=amplitude[i], initial_phase=initial_phase[i], show_T=show_T)
    return y


def mel1(y, sr=8000):
    #  采样频率应至少为最高频率的2倍，为避免边缘混叠，最好大于2.56倍 https://www.zhihu.com/question/50501790
    # 当关心频率成分时，可以按2.56倍的关系设置采样频率；但如果关心信号的幅值（时域），那样，采样频率应设置成关心的最高频率的10倍以上，才不会使信号幅值有明显的失真。
    sr = 50000
    # y = y[:sr * 50]
    y = get_y2(500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
               9000, 10000, 20000, 30000, 40000, 50000, sr=sr)
    # plt.plot(np.linspace(0, 2, len(y))[:1000], y[:1000])
    # plt.show()
    # 提取 mel spectrogram feature

    n_fft = 1024
    hop_length = 512
    win_length = None
    power = 1.0
    n_mels = 40
    fmin = 0.
    fmax = sr / 2
    if fmax is None:
        fmax = float(sr) / 2
    # 直接调用函数计算梅尔频谱
    melspec1 = librosa.feature.melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
                                              win_length=win_length, power=power, fmin=fmin, fmax=fmax)
    # 自行计算梅尔频谱
    spec = librosa.stft(y, n_fft=n_fft, hop_length=512, win_length=win_length, center=True)
    amplitude_spec = np.abs(spec) ** power
    #   构建梅尔滤波器
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    melspec2 = np.dot(mel_basis, amplitude_spec)

    logmelspec0 = librosa.power_to_db(amplitude_spec, top_db=80)  # 转换为对数刻度
    logmelspec1 = librosa.power_to_db(melspec1, top_db=80)  # 转换为对数刻度
    logmelspec2 = librosa.power_to_db(melspec2, top_db=80)  # 转换为对数刻度

    plt.figure()
    librosa.display.specshow(logmelspec0, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
    # librosa.display.specshow(logmelspec0, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')  # 右边的色度条
    plt.title('spec')
    plt.show()

    # 绘制 mel 频谱图
    plt.figure()
    # librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='hz')
    librosa.display.specshow(melspec1, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax)
    plt.colorbar(format='%+2.0f dB')  # 右边的色度条
    plt.title('melspec1')
    plt.show()

    plt.figure()
    # librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='hz')
    librosa.display.specshow(logmelspec2, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmin=fmin,
                             fmax=fmax)
    plt.colorbar(format='%+2.0f dB')  # 右边的色度条
    plt.title('melspec2')
    plt.show()


def mel2(y):
    # X = librosa.stft(audio_datas)
    # X = np.abs(X)
    # melW = librosa.filters.mel(sr=sr, n_fft=X.shape[1] * 2, n_mels=40, fmin=0., fmax=22100)
    # melW /= np.max(melW, axis=-1)[:, None]
    # melX = np.dot(X, melW.T)
    print()


def fft():
    sr = 768
    # y = get_y2(100, 200, 300, sr=sr)
    y = get_y3([0, 50, 75], sr=sr, amplitude=[2, 3, 1.5],
               initial_phase=[0 * np.pi / 180, -30 * np.pi / 180, 90 * np.pi / 180])
    t = np.linspace(0, 1, len(y))
    f = np.fft.fft(y)
    freal = f.real
    fimag = f.imag
    norm = np.abs(f)  # 模
    phase = np.angle(f)  # 相位
    phase1 = np.arctan2(fimag, freal)
    Fn = np.arange(0, len(y)) * sr / len(y)  # 某点n所表示的频率
    y_amplitude = np.concatenate(((norm / len(y))[:1], (norm / (len(y) / 2))[1:]))
    angle = phase * 180 / np.pi
    angle1 = phase1 * 180 / np.pi

    fig, axes = plt.subplots(6, 1, figsize=(8, 18))
    axes[0].plot(t, y, label="原始波形")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("signal")
    axes[0].set_xlim(0, 0.3)
    Y = np.zeros_like(y)
    label = ""
    for i, frequency in enumerate([0, 1, 50, 75]):
        label+=str(i)
         # 频率为i的波形
        y_ = y_amplitude[frequency] * np.cos(2 * np.pi * Fn[frequency] * t + phase[frequency])
        Y += y_
        axes[i+1].plot(t, Y, color="green", lw=2, label=f"{label}的波形")
        axes[i+1].set_xlabel("time (s)")
        axes[i+1].set_ylabel("signal")
        axes[i+1].legend()
        axes[i+1].set_xlim(0, 0.3)
    norm_shift = np.fft.fftshift(f)
    y1 = np.fft.ifft(f)
    y1_abs = np.abs(y1)
    y1_real = y1.real
    y1_imag = y1.imag
    axes[5].plot(t, y1_real+y1_imag, color="blue", lw=2, label=f"还原的波形")
    axes[5].set_xlim(0, 0.3)
    plt.show()
    print()


if __name__ == '__main__':
    root = r"E:\DATA\坐席辅助项目\坐席辅助公积金的录音下载文件\录音下载"
    root = r"/home/wcirq/Music"
    audio_file = os.path.join(root, "苏晗 - 没人再回来.flac")
    # audio_datas, sr = librosa.load(audio_file, sr=8000)
    # mel1(audio_datas, sr)
    fft()
