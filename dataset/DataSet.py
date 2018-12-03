from datetime import datetime
import os
import random
import re
import wave

import cv2
import h5py
import librosa
import numpy as np
import pyaudio
import matplotlib.pyplot as plt

from util.config import logger
from dataset.Audio import Audio
from dataset.Vad import Vad


class DataSet(Vad, Audio):
    def __init__(self, denoise_all=None, noise=None, merge_noise=None):
        super().__init__()
        if not denoise_all is None:
            self.denoise_all = self.readFilePath(denoise_all)
            self.noise = self.readFilePath(noise, use_file_name=True)
            self.merge_noise = self.readFilePath(merge_noise, use_file_name=True)

    def hanming(self, x):
        winfunc = 0.54 - (0.46 * np.cos((2 * np.pi * (np.arange(0, x))) / (x - 1)))
        return winfunc

    def princen_bradley(self, x):
        winfunc = np.sin((np.pi/2)*np.power(np.sin(np.pi*np.arange(0, x)/x),2))
        return winfunc

    def readFilePath(self, path, use_file_name=False):
        wav_file = {}
        files = os.listdir(path)
        for file in files:
            wav_file_path = os.path.join(path, file)
            if not use_file_name:
                if os.path.isfile(wav_file_path) and os.path.splitext(file)[-1][1:] == "wav":
                    name = os.path.splitext(os.path.basename(wav_file_path))[-2]
                    wav_file[name] = wav_file_path
            else:
                if os.path.isdir(wav_file_path):
                    name = re.split("[//|/|\\\]", wav_file_path)[-1]
                    data = []
                    noise_paths = os.listdir(wav_file_path)
                    for noise_path in noise_paths:
                        noise_file_path = os.path.join(wav_file_path, noise_path)
                        data.append(noise_file_path)
                    wav_file[name] = data
        assert (wav_file)
        return wav_file

    def start(self):
        for filename in self.wav_files:
            nchannels, sampwidth, framerate, nframes, wave_data = self.readWavFile(filename)
            frames = self.piecewise((nchannels, sampwidth, framerate, nframes, wave_data))
            fine_vad, coarse_vad = self.get_vad(frames)
            spectrogram, phase = self.audioToSpectrogram(frames)
            frame_len = len(frames[0])

            # audio = self.spectrogramToAudio(spectrogram, phase=phase, frame_len=frame_len, frames=frames)
            # self.drawSpectrogramAndPlay(audio, framerate, spectrogram, fine_vad, sampwidth, frame_len, filename)

            bark_spectrogram = self.spectrogramToBark(spectrogram, samplerate=framerate, filters_num=22, n=2000)
            ibark_spectrogram = self.spectrogramToBark(bark_spectrogram, samplerate=framerate, filters_num=22, n=2000)
            cv2.imwrite("./spectrogram.jpg", spectrogram * 255)
            cv2.imwrite("./bark_spectrogram.jpg", bark_spectrogram * 255)
            cv2.imwrite("./ibark_spectrogram.jpg", ibark_spectrogram * 255)
            audio = self.spectrogramToAudio(ibark_spectrogram, phase=phase, frame_len=frame_len)
            self.drawSpectrogramAndPlay(audio, framerate, ibark_spectrogram, fine_vad, sampwidth, frame_len, filename)

            # h5f = h5py.File("denoise_data9.h5", 'w')
            # h5f.create_dataset('data', data=data)
            # h5f.close()
            # print("ok")

    def drawSpectrogramAndPlay(self, wave_data, framerate, spectrogram, fine_vad, sampwidth, frame_len, filename=""):
        image1 = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        image1 = cv2.resize(image1, (image1.shape[1], 500))

        image2 = np.tile(fine_vad, 25)
        image2 = np.reshape(image2, (25, -1))
        image = np.vstack((image2, image1))

        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(sampwidth),
                        channels=1,
                        rate=framerate,
                        output=True)
        wave_data = np.asarray(wave_data * 32768, np.short)
        wave_data = np.maximum(np.minimum(wave_data, 32767), -32768)
        # plt.plot(wave_data[2000:2100])
        # plt.show()
        data = wave_data[:1600].tobytes()
        i = 1
        while data != b'':
            cv2.imshow("image", image[:, int((i - 1) * (1600 / frame_len)):int((i - 1) * (1600 / frame_len) + 1000)])
            stream.write(data)
            cv2.waitKey(1)
            data = wave_data[int(i * 1600):int(i * 1600 + 1600)].tobytes()
            i += 1
        # cv2.imwrite("./" + filename + ".jpg", image * 255)

    def build(self):
        datas = None
        for noise_name, noise_paths in self.merge_noise.items():
            save_path = os.path.join("../test/data", noise_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i, merge_noise in enumerate(noise_paths):
                key = os.path.splitext(os.path.basename(merge_noise))[-2]
                denoise = self.denoise_all[key]
                # 读取wav文件
                nchannels1, sampwidth1, framerate1, nframes1, x_train_wave = self.readWavFile(merge_noise)
                nchannels2, sampwidth2, framerate2, nframes2, y_train_wave = self.readWavFile(denoise)
                # 将声波转为矩阵
                x_train_matrix = self.piecewise((nchannels1, sampwidth1, framerate1, nframes1, x_train_wave),
                                                winfunc=self.princen_bradley)
                y_train_matrix = self.piecewise((nchannels2, sampwidth2, framerate2, nframes2, y_train_wave),
                                                winfunc=self.princen_bradley)
                # 从无噪音的数据中进行语音激活检测
                fine_vad, coarse_vad = self.get_vad(y_train_matrix)
                # 将转为矩阵的声波转为语谱图
                x_train_amp_spec, x_train_spec, phase_x = self.audioToSpectrogram(x_train_matrix, n=2000)
                y_train_amp_spec, y_train_spec, phase_y = self.audioToSpectrogram(y_train_matrix, n=2000)
                # 将语谱振幅图转为倒谱图
                x_ceps, x_one_derived, x_two_derived = self.spectrogramToCepstrum(x_train_amp_spec, n_derived=2)
                y_ceps, y_one_derived, y_two_derived = self.spectrogramToCepstrum(y_train_amp_spec, n_derived=0)

                # fine_vad = np.tile(fine_vad, 10)
                # fine_vad = np.transpose(np.reshape(fine_vad, (10, -1)), (1,0))
                fine_vad = np.reshape(fine_vad, (-1, 1))

                x_ceps2 = (x_ceps - np.mean(x_ceps)) / np.std(x_ceps)
                x_ceps3 = x_ceps2 * np.std(x_ceps2) + np.mean(x_ceps2)

                data = np.hstack((fine_vad, x_one_derived, x_two_derived, x_ceps, y_ceps))
                # cv2.imshow("y", np.transpose(data, (1,0)))
                # cv2.waitKey(0)

                if datas is None:
                    datas = data
                else:
                    datas = np.vstack((datas, data))
                # logger.info("{0}/{1} % ".format(i, len(noise_paths)))
                if (i % 50 == 0 and i != 0) or i == len(noise_paths) - 1:
                    print("第{0}个,第{1}个文件:".format(i, i // 50))
                    filename = os.path.join(save_path, datetime.now().strftime("%Y%m%d%H%M%S") + "_%s.h5")
                    flag = 0
                    while 1:
                        tmp = filename % flag
                        if os.path.isfile(tmp):
                            flag += 1
                        else:
                            filename = tmp
                            break
                    h5f = h5py.File(filename, 'w')
                    h5f.create_dataset('data', data=datas)
                    h5f.close()
                    datas = None

    def getSpectrogram(self, wavPath):
        nchannels1, sampwidth1, framerate1, nframes1, wav_data = self.readWavFile(wavPath)
        # 将声波转为矩阵
        # matrix = self.piecewise((nchannels1, sampwidth1, framerate1, nframes1, wav_data))
        matrix = self.piecewise((nchannels1, sampwidth1, framerate1, nframes1, wav_data), winfunc=self.hanming)
        frame_len = len(matrix[0])
        # 将转为矩阵的声波转为语谱图
        spec, phase = self.audioToSpectrogram(matrix)
        # 将语谱图转换为Bark域
        bark = self.spectrogramToBark(spec, samplerate=framerate1, filters_num=22, n=2000)
        return bark, phase, frame_len, spec, (nchannels1, sampwidth1, framerate1, nframes1, wav_data)

    def toAudio(self, bark, phase, samplerate=16000, frame_len=80):
        spec = self.barkToSpectrogram(bark, samplerate=samplerate)
        audio = self.spectrogramToAudio(spec, phase=phase, frame_len=frame_len)
        return audio, spec


if __name__ == "__main__":
    # wavPath = "../resources/data/"
    # denoise_train = "D:/BaiduNetdiskDownload/声音数据/data_thchs30/train"
    # denoise_test = "D:/BaiduNetdiskDownload/声音数据/data_thchs30/train"
    # denoise_dev = "D:/BaiduNetdiskDownload/声音数据/data_thchs30/dev"


    denoise_all = "D:/BaiduNetdiskDownload/声音数据/data_thchs30/data"

    noise = "D:/BaiduNetdiskDownload/声音数据/test_noise/noise"

    merge_noise = "D:/BaiduNetdiskDownload/声音数据/test_noise/0db"

    dataset = DataSet(denoise_all, noise, merge_noise)
    dataset.princen_bradley(100)
    dataset.build()
    print()
