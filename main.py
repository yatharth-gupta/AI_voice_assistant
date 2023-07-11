import asyncio, json
from EdgeGPT.EdgeGPT import Chatbot, ConversationStyle
from EdgeGPT.EdgeUtils import Query,Cookie
import whisper
import time
from TTS.api import TTS
# from playsound import playsound
import pydub
from pydub import playback
import re
import os
import sounddevice as sd
import soundfile as sf
model_name = "tts_models/en/ljspeech/fast_pitch"
# model_name = TTS.list_models()[0]
tts = TTS(model_name, progress_bar=False, gpu=False)
fs =44100
sec = 7
# async def main():
    # model = whisper.load_model("base")

    # # load audio and pad/trim it to fit 30 seconds
    # audio = whisper.load_audio("recorded_audio.wav")
    # # audio = whisper.load_audio("input.wav")
    # audio = whisper.pad_or_trim(audio)

    # # make log-Mel spectrogram and move to the same device as the model
    # mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # # detect the spoken language
    # _, probs = model.detect_language(mel)
    # print(f"Detected language: {max(probs, key=probs.get)}")

    # # decode the audio
    # options = whisper.DecodingOptions(fp16=False)
    # result = whisper.decode(model, mel, options)

    # # print the recognized text
    # print(result.text)
    # time.sleep(1)

#     cookies = json.loads(open("./bing_cookies_*.json", encoding="utf-8").read())  # might omit cookies option
#     bot = await Chatbot.create(cookies=cookies)
#     response = await bot.ask(prompt=result.text, conversation_style=ConversationStyle.precise)
#     print(response["item"]["messages"])
#     # for message in response["item"]["messages"]:
#     #     if message["author"] == "bot":
#     #         bot_response = message["text"]
#     # # Remove [^#^] citations in response
#     # bot_response = re.sub('\[\^\d+\^\]', '', bot_response)

    # tts.tts_to_file(text="hello world",speed="1.2", file_path="output.wav")
    # time.sleep(1)
    # sound = pydub.AudioSegment.from_file("output.wav", format="wav")
    # playback.play(sound)
#     await bot.close()

# if __name__ == "__main__":
#     asyncio.run(main())

print("start recording")
file_path = "recorded_audio.wav"
absolute_path = os.path.abspath(file_path)
open(absolute_path, 'w').close()
# subprocess.run(['Sound Recorder', '/FILE', file_path, '/DURATION', str(sec)])
myrecording  = sd.rec(int(sec*fs),samplerate = fs,channels = 2)
sd.wait()
# Save the recorded audio to a file
sf.write(file_path, myrecording, fs)
print("stop recording")
time.sleep(1)

model = whisper.load_model("base")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(absolute_path)
# audio = whisper.load_audio("input.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions(fp16=False)
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
time.sleep(1)
# Cookie.dir_path =os.path.join(os.getcwd(),"bing_cookies_*.json")

file_path = "output.wav"
absolute_path = os.path.abspath(file_path)
open(absolute_path, 'w').close()
q = Query(result.text)
q = str(q)
q = re.sub(r'\[\^\d+\^\]', '', q)
q = q.replace("**","")
print(q)
# wav = tts.tts("This is a test! This is also a test!!")
tts.tts_to_file(text=q,speed="1.2", file_path=absolute_path)
time.sleep(1)
sound = pydub.AudioSegment.from_file(absolute_path, format="wav")
playback.play(sound)