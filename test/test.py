from pydub import AudioSegment
import re


def trans_mp3_to_wav(filepath):
    song = AudioSegment.from_mp3(filepath)
    song.export("./man_1.wav", format="wav")


# trans_mp3_to_wav("../test/man_1.mp3")
amp = str(0.1)
amp = amp.split(".")[1]
pattern = re.compile(r'^[0]+')
match = pattern.search(amp)
print(match)
