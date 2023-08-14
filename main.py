from EdgeGPT.EdgeUtils import Query,Cookie # Importing the Query class from EdgeUtils
import whisper # speech to text
import time
import pydub # for playing the audio
from pydub import playback # for playing the audio
import re # for removing unwanted characters
import os
from gtts import gTTS # for text to speech
import sounddevice as sd # for recording audio
import soundfile as sf # for saving audio


# from TTS.api import TTS
# model_name = "tts_models/en/ljspeech/fast_pitch"
# model_name = TTS.list_models()[0]
# tts = TTS(model_name, progress_bar=False, gpu=False)

fs =44100 # Sample rate
sec = 7 # Duration of recording

print("start recording")
file_path = "recorded_audio.wav"
absolute_path = os.path.abspath(file_path)
open(absolute_path, 'w').close() # clearing the file , w is for write mode
myrecording  = sd.rec(int(sec*fs),samplerate = fs,channels = 2) # recording audio
sd.wait() # Wait until recording is finished
sf.write(file_path, myrecording, fs) # Save the recorded audio to a file
print("stop recording")
time.sleep(1)

model = whisper.load_model("base")

# load audio 
audio = whisper.load_audio(absolute_path)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel) # probs is a dictionary of probabilities w.r.t each language
print(f"Detected language: {max(probs, key=probs.get)}") # printing the language with max probability

# decode the audio
options = whisper.DecodingOptions(fp16=False) # fp16 is for 16 bit floating point precision
result = whisper.decode(model, mel, options) # result is a class object, it has a text attribute which contains the recognized text

# print the recognized text
print(result.text)

# querying Bing AI
q = Query(result.text)

# removing extra unwanted characters
q = str(q)
q = re.sub(r'\[\^\d+\^\]', '', q)
q = q.replace("**","")
print(q)

# can also use Coqui-TTS for text to speech

# tts.tts_to_file(text="hello world",speed="1.2", file_path="output.wav")
# time.sleep(1)
# sound = pydub.AudioSegment.from_file("output.wav", format="wav")
# playback.play(sound)

file_path = "output.mp3"
absolute_path = os.path.abspath(file_path)
# converting text to speech using gTTS (google text to speech)
myobj = gTTS(text=q, lang='en', slow=False) 
myobj.save("output.mp3")
# Playing the converted file
sound = pydub.AudioSegment.from_file(absolute_path, format="mp3")
fastersound = sound.speedup(playback_speed=1.2)
# making new temp file to store the speedup audio
temp_file = "./tempfile.mp3"
fastersound.export(temp_file, format="mp3")
file_path = "tempfile.mp3"
absolute_path = os.path.abspath(file_path)
# running tempfile
sound = pydub.AudioSegment.from_file(absolute_path, format="mp3")
playback.play(sound)