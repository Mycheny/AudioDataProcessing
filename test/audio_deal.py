# -*- coding: utf-8 -*- 
# @File audio_deal.py
# @Time 2020/12/4 11:28
# @Author wcy
# @Software: PyCharm
# @Site
import math
import wave
import cv2
import librosa
import numpy as np
import pyaudio
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
        initial_phase = [0] * len(frequencys)
    y = np.zeros_like(get_random_wave(0, sr=sr, show_T=show_T))
    for i, frequency in enumerate(frequencys):
        y += get_random_wave(frequency, sr=sr, amplitude=amplitude[i], initial_phase=initial_phase[i], show_T=show_T)
    return y


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
        self.sampwidth = sampwidth
        self.nchannels = nchannels
        self.sampling_rate = sampling_rate
        return sampling_rate, wave_data

    def hanming(self, x):
        """汉明窗"""
        winfunc = 0.54 - (0.46 * np.cos((2 * np.pi * (np.arange(0, x))) / (x - 1)))
        return winfunc

    def princen_bradley(self, x):
        """princen_bradley加窗函数"""
        winfunc = np.sin((np.pi / 2) * np.power(np.sin(np.pi * np.arange(0, x) / x), 2))
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
        win = winfunc(self.frame_length)  # window窗函数，这里默认取1
        frames *= win  # 信号加窗处理
        return frames

    def imshow(self, images, win_name="", delay=0):
        win_h, win_w = 1366, 746
        h, w = images.shape
        if h > win_h or w > win_w:
            if h > w:
                h, w = win_h, int(w * win_h / h)
            else:
                h, w = int(win_w * w / h), win_w
        images = cv2.resize(images, (w, h))
        # images = (images-images.min())/(images.max()-images.min())
        images = np.log10(np.maximum(1e-10, images))
        images = (images - images.min()) / (images.max() - images.min())
        cv2.imshow(win_name, images.T)
        cv2.waitKey(delay=delay)

    def frames_to_spectrogram(self, frames, n=None):
        if n is None:
            n = frames.shape[-1]
        complex_spectrum = np.fft.fft(frames, n=n)  # n默认为frames.shape[-1]，当信号长度刚好为1秒时， 结果的索引即对应频率，若n为默认值乘以2，则索引乘以2等于频率
        freq = np.fft.fftfreq(n, 1 / self.sampling_rate)
        amp_spectrum = np.absolute(complex_spectrum)
        amp_spectrum = np.concatenate(((amp_spectrum / frames.shape[1])[:, :1],
                                       (amp_spectrum / (frames.shape[1] / 2))[:, 1:]), axis=1)
        phase = np.angle(complex_spectrum)
        # 欧拉公式 e^ix = cos(x)+i*sin(x), 因(e^a)*(e^b)=e^(a+b) --> (e^0)*(e^x)=e^(0+x)
        restore_complex_spectrum = amp_spectrum * np.exp(1j * phase)
        restore_complex_spectrum2 = amp_spectrum * (np.cos(phase) + 1j * np.sin(phase))

        self.imshow(amp_spectrum)

        spec_log = np.log10(np.maximum(1e-10, amp_spectrum))
        return amp_spectrum, spec_log, phase
    
    def microphone(self, NUM_SAMPLES=None):
        if NUM_SAMPLES is None:
            NUM_SAMPLES = self.frame_length
        TIME= 100
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1,
                         rate=self.sampling_rate, input=True,
                         frames_per_buffer=NUM_SAMPLES)
        count = 0
        while count < TIME * 8:  # 控制录音时间
            # count+=1
            string_audio_data = stream.read(NUM_SAMPLES)
            wave_data = np.fromstring(string_audio_data, dtype=np.short)
            wave_data = wave_data/self.signal_maximum
            yield wave_data

    def play(self, frames, winfunc=lambda x: np.ones((x,))):
        def save_wave_file(data, filename="result.wav"):
            '''save the data to the wavfile'''
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.nchannels)
            wf.setsampwidth(self.sampwidth)
            wf.setframerate(self.sampling_rate)
            wf.writeframes(b"".join(data))
            wf.close()

        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(self.sampwidth),
                        channels=self.nchannels,
                        rate=self.sampling_rate,
                        output=True)
        effective = int(self.frame_step / self.frame_time * self.frame_length)
        effective = effective*2  # 实时采集音频时
        # 若使用窗函数处理过信号，则需进行还原
        iwin = 1 / winfunc(frames.shape[1])
        frames = frames * iwin
        show_len = 1200
        spectrogram = np.zeros((effective, show_len))

        wave_datas = []
        # for frame in frames:
        for frame in self.microphone():
            # frame = get_y3([180], sr=self.sampling_rate)
            # frame = (frame - frame.min()) / (frame.max() - frame.min())

            frame = frame[:effective]
            # frame = frame[int((frame.shape[0]-effective)/2):int((frame.shape[0]+effective)/2)]
            frame_restore = frame * self.signal_maximum
            wave_data_short = np.asarray(frame_restore, np.short)
            wave_data_short = np.maximum(np.minimum(wave_data_short, 32767), -32768)
            wave_data = wave_data_short.tobytes()
            # stream.write(wave_data)  # 播放原始语音

            sepc = np.fft.fft(frame)
            freq = np.fft.fftfreq(np.size(frame, 0), 1 / self.sampling_rate)
            # sepc = librosa.stft(frame, n_fft=frame.shape[0]*2, hop_length=512, center=True)
            # sepc = sepc[:-1, 0]
            amp = np.abs(sepc)
            phase = np.angle(sepc)
            # amp[amp.shape[0]//4:amp.shape[0]*3//4] = 0
            # phase[phase.shape[0]//4:phase.shape[0]*3//4] = 0


            restore_sepc = amp*np.exp(1j*phase)
            restore_signal_complex = np.fft.ifft(restore_sepc)
            restore_signal_norm = np.abs(restore_signal_complex)  # 复数的模，作为语音信号，播放失真
            restore_signal_real = restore_signal_complex.real  # 复数的实数部分，作为语音信号正常
            restore_signal_imag = restore_signal_complex.imag  # 复数的实数部分，不能作为语音信号，播放为噪音

            restore_wave_data = np.asarray(restore_signal_real * self.signal_maximum, np.short)
            restore_wave_data = np.maximum(np.minimum(restore_wave_data, 32767), -32768)
            restore_wave_data = restore_wave_data.tobytes()
            stream.write(restore_wave_data)   # 播放还原的语音
            wave_datas.append(restore_wave_data)

            amp_real = np.concatenate(((amp / len(frame))[:1], (amp / (len(frame) / 2))[1:]), axis=0)
            amp_real_log = (amp_real - amp_real.min()) / (amp_real.max() - amp_real.min())
            # amp_real_log = np.log10(np.maximum(1e-10, amp_real))
            # amp_real_log = (amp_real_log - amp_real_log.min()) / (amp_real_log.max() - amp_real_log.min())
            spectrogram[:, :-1] = spectrogram[:, 1:]
            spectrogram[:, -1] = amp_real_log
            images = np.copy(spectrogram)
            stop = sepc.shape[0]
            for i in range(stop):
                if i % 100 == 0:
                    text = f"{round(freq[i], 1)}HZ"
                    cv2.putText(images, text, (12, i + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.line(images, (0, i), (10, i), (255, 255, 255), 1)
            cv2.line(spectrogram, (show_len-1, effective//4+int(frame_restore.min()/(self.signal_maximum/1)* effective//4)), (show_len-1, effective//4+int(frame_restore.max()/(self.signal_maximum/1) * effective//4)), (127, 127, 127), 1)
            win_h, win_w = 746, 1366
            h, w = images.shape
            if h > win_h or w > win_w:
                if h > w:
                    h, w = win_h, int(w * win_h / h)
                else:
                    h, w = int(win_w * w / h), win_w
            images = cv2.resize(images, (w, h))
            cv2.imshow("", images)
            cv2.waitKey(1)
        save_wave_file(wave_datas)


if __name__ == '__main__':
    # audio_file = r"E:\FFOutput\20200907095114_18076088691.wav"
    audio_file = r"E:\PycharmProjects\AudioDataProcessing\test\data\15KHz-44.1K-sine_0dB.wav"
    audio_file = r"E:\PycharmProjects\AudioDataProcessing\test\rensheng.wav"
    audio_deal = AudioDeal(frame_time=25)
    sampling_rate, speech_signal = audio_deal.read_wav(audio_file, )
    frames = audio_deal.piecewise(speech_signal, sampling_rate, winfunc=audio_deal.hanming)
    # frames = audio_deal.piecewise(speech_signal, sampling_rate)
    audio_deal.play(frames, winfunc=audio_deal.hanming)
    # audio_deal.frames_to_spectrogram(frames)
    print()
