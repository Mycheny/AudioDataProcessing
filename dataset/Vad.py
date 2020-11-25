import math
import numpy as np
import copy
from dataset.ReadWav import ReadWav
import matplotlib.pyplot as plt
from util.config import logger


class Vad(ReadWav):
    def __init__(self):
        super().__init__()
        self.frame_time = 1000  # 多少ms一帧(ms)
        self.frame_step = self.frame_time/2  # 帧的步长

        # 初始短时能量高门限
        self.amp1 = 10
        # 初始短时能量低门限
        self.amp2 = 0.1
        # 初始短时过零率高门限
        self.zcr1 = 8
        # 初始短时过零率低门限
        self.zcr2 = 3
        # 允许最大静音长度
        self.maxsilence = 5
        # 语音的最短长度
        self.minlen = 5
        # 偏移值
        self.offsets = 1  # 有人声时的前部分偏差数 × frame_time ÷2 = 偏差时间 eg: 2 × 100(ms) ÷ 2=100(ms)
        self.offsete = 1  # 有人声时的后部分偏差数 × frame_time ÷2 = 偏差时间 eg: 2 × 100(ms) ÷ 2=100(ms)
        # 初始状态为静音
        self.status = 0  # 当前语音状态 0= 静音， 1= 可能开始, 2= 确定进入语音段, 3- 语音结束
        self.count = 0
        self.silence = 0
        self.last_status = 0  # 上次语音状态 0= 静音， 1= 可能开始, 2= 确定进入语音段, 3- 语音结束
        self.frames = []
        # 数据开始偏移
        self.frames_start = []
        self.frames_start_num = 0
        # 数据结束偏移
        self.frames_end = []
        self.frames_end_num = 0
        self.end_flag = False

    def piecewise(self, data, winfunc=lambda x: np.ones((x,))):
        """
        处理音频数据，将其分成part_num部分
        :param winfunc:
        :param data:
        :return:
        """
        nchannels, sampwidth, framerate, nframes, wave_data = data
        signal_length = nframes  # 信号总长度
        frame_length = int(round(framerate/1000*self.frame_time))  # 以帧帧时间长度
        self.frame_length = frame_length
        frame_step = int(round(framerate/1000*self.frame_step))  # 相邻帧之间的步长
        if signal_length <= frame_length:  # 若信号长度小于一个帧的长度，则帧数定义为1
            frames_num = 1
        else:  # 否则，计算帧的总长度
            frames_num = 1 + int(math.ceil((1.0 * signal_length - frame_length) / frame_step))
        pad_length = int((frames_num - 1) * frame_step + frame_length)  # 所有帧加起来总的铺平后的长度
        zeros = np.zeros((pad_length - signal_length,))  # 不够的长度使用0填补，类似于FFT中的扩充数组操作
        pad_signal = np.concatenate((wave_data, zeros))  # 填补后的信号记为pad_signal
        x = np.arange(0, frame_length)
        y = np.arange(0, frames_num * frame_step, frame_step)
        a = np.tile(x, (frames_num, 1))
        b = np.tile(y, (frame_length, 1))
        bt = b.T
        indices = a + bt  # 相当于对所有帧的时间点进行抽取，得到frames_num*frame_length长度的矩阵
        indices = np.array(indices, dtype=np.int32)  # 将indices转化为矩阵
        frames = pad_signal[indices]  # 得到帧信号
        t = winfunc(frame_length)
        win = np.tile(t, (frames_num, 1))  # window窗函数，这里默认取1
        return frames * win  # 返回帧信号矩阵

    def get_vad(self, cache_frames):
        """
        开始执行音频端点检测
        :param cache_frames:
        :return:
        """
        cache_frames = cache_frames.tolist()
        cache_frames = copy.deepcopy(cache_frames)
        fine_vad = np.zeros((int(len(cache_frames))))
        num = 0
        while len(cache_frames) > 0:
            # 开始端点
            if len(cache_frames) != 1:
                data = np.reshape(np.array(cache_frames[:2]), (-1))
            else:
                data = np.reshape(np.array(cache_frames[:1]), (-1))
            # 获得音频过零率
            zcr = self.ZCR(data)
            # 获得音频的短时能量(取绝对值求和), 平方放大
            amp = self.STE(data)
            # 移除第一帧并返回移除的帧
            speech_data = cache_frames.pop(0)
            # 返回当前音频数据状态
            # print(amp, zcr)
            status = self.speech_status(amp, zcr)
            if len(cache_frames) == 1 and status == 2:
                fine_vad[-1] = 1
            self.frames_start.append(1)
            self.frames_start_num += 1
            if self.frames_start_num == self.offsets:
                # 开始音频开始的缓存部分
                self.frames_start.pop(0)
                self.frames_start_num -= 1
            if self.end_flag:
                # 当音频结束后进行后部缓存
                self.frames_end_num += 1
                if status == 2 or self.frames_end_num == self.offsete:
                    amount = len(self.frames + self.frames_end)
                    # 处理
                    start = num - amount
                    end = num
                    fine_vad[start:end] = 1
                    self.end_flag = False
                    self.frames = []
                    self.frames_end_num = 0
                    self.frames_end = []
                self.frames_end.append(1)
            if status == 0:
                pass
            if status == 1:
                pass
            if status == 2:
                if self.last_status in [0, 1]:
                    # 添加开始偏移数据到数据缓存
                    self.frames = self.frames + self.frames_start
                # 添加当前的语音数据
                self.frames.append(1)
            if status == 3:
                self.frames.append(1)
                # 开启音频结束标志
                self.end_flag = True
            num = num + 1
            self.last_status = status
        coarse_vad = np.array([np.mean(fine_vad[i*2:i*2+2]) for i in range(int(len(fine_vad)/2))])
        return fine_vad, coarse_vad

    def speech_status(self, amp, zcr):
        status = 0
        # 0= 静音， 1= 可能开始, 2= 确定进入语音段, 3- 语音结束
        if self.last_status in [0, 1]:
            # 确定进入语音段
            if amp > self.amp1:
                status = 2
                self.silence = 0
                self.count += 1
            # 可能处于语音段
            elif amp > self.amp2 or zcr > self.zcr2:
                status = 1
                self.count += 1
            # 静音状态
            else:
                status = 0
                self.count = 0
                self.count = 0
        # 2 = 语音段
        elif self.last_status == 2:
            # 保持在语音段
            if amp > self.amp2 or zcr > self.zcr2:
                self.count += 1
                status = 2
            # 语音将结束
            else:
                # 静音还不够长，尚未结束
                self.silence += 1
                if self.silence < self.maxsilence:
                    self.count += 1
                    status = 2
                # 语音长度太短认为是噪声
                elif self.count < self.minlen:
                    status = 0
                    self.silence = 0
                    self.count = 0
                # 语音结束
                else:
                    status = 3
                    self.silence = 0
                    self.count = 0
        return status

    @staticmethod
    def ZCR(curFrame):
        # 过零率
        tmp1 = curFrame[:-1]
        tmp2 = curFrame[1:]
        sings = (tmp1 * tmp2 <= 0)
        diffs = (tmp1 - tmp2) > 0.02
        zcr = np.sum(sings * diffs)
        return zcr

    @staticmethod
    def STE(curFrame):
        # 短时能量
        amp = np.sum(np.abs(curFrame)) ** 2
        return amp

