# -*- coding: utf-8 -*- 
# @File scipy_demo.py
# @Time 2020/11/20 9:20
# @Author wcy
# @Software: PyCharm
# @Site
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]

def signal_samples(t, min_f=2, max_f=20):
    return np.sin(2 * np.pi * min_f * t) + np.sin(2 * np.pi * max_f * t)


if __name__ == '__main__':
    min_f = 2
    max_f = 20
    f_s = int(2.56*4*max_f) # 4个2.56倍, 采样了
    T = 10
    N = f_s * T
    t = np.linspace(0, T, N)
    f_t = signal_samples(t, min_f=min_f, max_f=max_f)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    axes[0].plot(t, f_t)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("signal")
    axes[1].plot(t, f_t)
    axes[1].set_xlim(0, T/5)
    axes[1].set_xlabel("time (s)")
    plt.show()

    F = fftpack.fft(f_t)
    f = fftpack.fftfreq(N, 1.0 / f_s)
    F_filtered = F * (abs(f) <5)
    a = np.append(np.expand_dims(f, axis=1), np.expand_dims((abs(f) < 10), axis=1), axis=1)
    f_t_filtered = fftpack.ifft(F_filtered)
    mask = np.where(f >= 0)

    fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    axes[0].plot(f[mask], np.log(abs(F[mask])), label="real")
    axes[0].plot(min_f, 0, 'b*', markersize=10)  # 最小频率位置
    axes[0].plot(max_f, 0, 'r*', markersize=10)  # 最大频率位置
    axes[0].set_ylabel("$\log(|F|)$", fontsize=14)

    axes[1].plot(f[mask], abs(F[mask]) / N, label="real")
    axes[1].plot(min_f, 0, 'b*', markersize=10)  # 最小频率位置
    axes[1].plot(max_f, 0, 'r*', markersize=10)  # 最大频率位置
    axes[1].set_ylabel("$|F|$", fontsize=14)

    axes[2].plot(t, f_t, color="brown", lw=7, label='原始')
    axes[2].plot(t, f_t_filtered.real, color="green", lw=5, label='实数部分')
    axes[2].plot(t, f_t_filtered.imag, color="blue", lw=3, label='虚数部分')
    axes[2].plot(t, f_t_filtered.real+f_t_filtered.imag, color="black", lw=1, label='合并部分')
    axes[2].legend()
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("time (s)", fontsize=14)
    axes[2].set_ylabel("$|F|$", fontsize=14)
    plt.show()
    print()
