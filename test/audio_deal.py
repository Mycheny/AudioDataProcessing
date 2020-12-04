# -*- coding: utf-8 -*- 
# @File audio_deal.py
# @Time 2020/12/4 11:28
# @Author wcy
# @Software: PyCharm
# @Site
import math
import wave
import numpy as np


class AudioDeal():
    def __init__(self, signal_maximum=32768, frame_time=32, frame_step=None):
        """
        :param signal_maximum: 读取的语音信号最大值, 因np.int16/np.short类型范围为（-32768至32767），所以默认为32768
        :param frame_time: 语音信号分帧时单帧代表的时间（ms），帧长一般取为 20 ~ 50 毫秒
        :param frame_step: 语音信号分帧时帧的步长(帧移)（ms）
        """
        self.signal_maximum = signal_maximum
        self.frame_time = frame_time  # 多少ms一帧(ms)
        if frame_step is None:
            self.frame_step = self.frame_time / 2  # 帧的步长
        else:
            self.frame_step = frame_step

    def read_wav(self, wav_path, limit_signal_length=None, normalization=True):
        """
        读取wav文件信号
        :param wav_path: 文件路径
        :param limit_signal_length: 限制信号长度，None 为不限制
        :param normalization: 是否归一化
        :return: 采样率, 语音信号
        """
        with wave.open(wav_path, "rb") as read_file:
            params = read_file.getparams()
            nchannels, sampwidth, sampling_rate, nframes = params[:4]
            total_time = int(nframes / sampling_rate * 1000)  # wav文件时长(ms)
            # logger.info("{0} 总时长 {1} ms".format(wav_path, total_time))
            data = read_file.readframes(nframes)
            wave_data = np.fromstring(data, dtype=np.short)
        if nchannels == 2:
            # 若是双通道则将其转为单通道
            wave_data = wave_data[range(0, int(nframes * nchannels), 2)]
            nchannels = 1
        if limit_signal_length is not None:
            # 如果限制信号的长度，则进行裁剪
            nframes = limit_signal_length
            wave_data = wave_data[:limit_signal_length]
        if normalization:
            wave_data = wave_data / self.signal_maximum  # 归一化
        return sampling_rate, wave_data

    def hanming(self, x):
        """汉明窗"""
        winfunc = 0.54 - (0.46 * np.cos((2 * np.pi * (np.arange(0, x))) / (x - 1)))
        return winfunc

    def princen_bradley(self, x):
        """princen_bradley加窗函数"""
        winfunc = np.sin((np.pi/2)*np.power(np.sin(np.pi*np.arange(0, x)/x),2))
        return winfunc

    def piecewise(self, speech_signal, sampling_rate=8800, winfunc=lambda x: np.ones((x,))):
        """
        对语音信号进行分帧 参考资料 https://www.zhihu.com/question/52093104
        :param speech_signal: 语音信号
        :param sampling_rate: 采样率
        :param winfunc: 加窗函数
        :return: shape [frames_num, frame_length]
        """
        signal_length = len(speech_signal)  # 信号总长度
        self.frame_length = int(round(sampling_rate / 1000 * self.frame_time))  # 以帧帧时间长度
        frame_step = int(round(sampling_rate / 1000 * self.frame_step))  # 相邻帧之间的步长
        if signal_length <= self.frame_length:  # 若信号长度小于一个帧的长度，则帧数定义为1
            frames_num = 1
        else:  # 否则，计算帧的总长度
            frames_num = 1 + int(math.ceil((1.0 * signal_length - self.frame_length) / frame_step))
        pad_length = int((frames_num - 1) * frame_step + self.frame_length)  # 所有帧加起来总的铺平后的长度
        zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
        pad_signal = np.concatenate((speech_signal, zeros))  # 填补后的信号记为pad_signal
        x = np.arange(0, self.frame_length)
        y = np.arange(0, frames_num * frame_step, frame_step)
        a = np.tile(x, (frames_num, 1))
        b = np.tile(y, (self.frame_length, 1))
        bt = b.T
        indices = a + bt  # 相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
        indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
        frames = pad_signal[indices]  # 得到帧信号
        win = winfunc(self.frame_length) # window窗函数，这里默认取1
        frames *= win  # 信号加窗处理
        return frames


if __name__ == '__main__':
    audio_file = r"E:\FFOutput\20200907095114_18076088691.wav"
    audio_deal = AudioDeal()
    sampling_rate, speech_signal = audio_deal.read_wav(audio_file)
    frames = audio_deal.piecewise(speech_signal, sampling_rate, winfunc=audio_deal.hanming)
    print()