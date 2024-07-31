# from EdgeGPT.EdgeUtils import Query,Cookie # Importing the Query class from EdgeUtils
# import whisper # speech to text
# import time
# import pydub # for playing the audio
# from pydub import playback # for playing the audio
# import re # for removing unwanted characters
# import os
# from gtts import gTTS # for text to speech
# import sounddevice as sd # for recording audio
# import soundfile as sf # for saving audio


# # from TTS.api import TTS
# # model_name = "tts_models/en/ljspeech/fast_pitch"
# # model_name = TTS.list_models()[0]
# # tts = TTS(model_name, progress_bar=False, gpu=False)

# fs =44100 # Sample rate
# sec = 7 # Duration of recording

# print("start recording")
# file_path = "recorded_audio.wav"
# absolute_path = os.path.abspath(file_path)
# open(absolute_path, 'w').close() # clearing the file , w is for write mode
# myrecording  = sd.rec(int(sec*fs),samplerate = fs,channels = 2) # recording audio
# sd.wait() # Wait until recording is finished
# sf.write(file_path, myrecording, fs) # Save the recorded audio to a file
# print("stop recording")
# time.sleep(1)

# model = whisper.load_model("base")

# # load audio 
# audio = whisper.load_audio(absolute_path)

# # make log-Mel spectrogram and move to the same device as the model
# mel = whisper.log_mel_spectrogram(audio).to(model.device)

# # detect the spoken language
# _, probs = model.detect_language(mel) # probs is a dictionary of probabilities w.r.t each language
# print(f"Detected language: {max(probs, key=probs.get)}") # printing the language with max probability

# # decode the audio
# options = whisper.DecodingOptions(fp16=False) # fp16 is for 16 bit floating point precision
# result = whisper.decode(model, mel, options) # result is a class object, it has a text attribute which contains the recognized text

# # print the recognized text
# print(result.text)

# # querying Bing AI
# q = Query(result.text)

# # removing extra unwanted characters
# q = str(q)
# q = re.sub(r'\[\^\d+\^\]', '', q)
# q = q.replace("**","")
# print(q)

# # can also use Coqui-TTS for text to speech

# # tts.tts_to_file(text="hello world",speed="1.2", file_path="output.wav")
# # time.sleep(1)
# # sound = pydub.AudioSegment.from_file("output.wav", format="wav")
# # playback.play(sound)

# file_path = "output.mp3"
# absolute_path = os.path.abspath(file_path)
# # converting text to speech using gTTS (google text to speech)
# myobj = gTTS(text=q, lang='en', slow=False) 
# myobj.save("output.mp3")
# # Playing the converted file
# sound = pydub.AudioSegment.from_file(absolute_path, format="mp3")
# fastersound = sound.speedup(playback_speed=1.2)
# # making new temp file to store the speedup audio
# temp_file = "./tempfile.mp3"
# fastersound.export(temp_file, format="mp3")
# file_path = "tempfile.mp3"
# absolute_path = os.path.abspath(file_path)
# # running tempfile
# sound = pydub.AudioSegment.from_file(absolute_path, format="mp3")
# playback.play(sound)


import sounddevice as sd
import playsound as ps
import numpy as np
import speech_recognition as sr
import os
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from os import system
import streamlit as st

# Load the API key from the environment
load_dotenv()

def record_audio(duration=5, fs=44100):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished")
    return recording

def recognize_speech_from_audio(audio, fs=44100):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioData(audio.tobytes(), fs, 2)
    try:
        text = recognizer.recognize_google(audio_data)
        print("Recognized Text:", text)
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return ""

def chat_with_assistant():
    groq_api_key = os.environ['GROQ_KEY']
    openai_api_key = os.environ['OPENAI_KEY']
    model = 'llama3-8b-8192'

    client = OpenAI(api_key=openai_api_key)

    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )

    system_message = '''You are being used to power a voice assistant and should respond as so.
            As a voice assistant, use short sentences. '''
    system_message = system_message.replace('\n', '')

    conversational_memory_length = 5  # Number of previous messages the chatbot will remember during the conversation

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    while True:
        # Record user question
        audio = record_audio(duration=5, fs=44100)
        user_question = recognize_speech_from_audio(audio, fs=44100)
        
        if user_question:

            # Construct a chat prompt template using various components
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=system_message),  # This is the persistent system prompt that is always included at the start of the chat.
                    MessagesPlaceholder(variable_name="chat_history"),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.
                    HumanMessagePromptTemplate.from_template("{human_input}"),  # This template is where the user's current input will be injected into the prompt.
                ]
            )

            # Create a conversation chain using the LangChain LLM (Language Learning Model)
            conversation = LLMChain(
                llm=groq_chat,  # The Groq LangChain chat object initialized earlier.
                prompt=prompt,  # The constructed prompt template.
                verbose=False,  # TRUE Enables verbose output, which can be useful for debugging.
                memory=memory,  # The conversational memory object that stores and manages the conversation history.
            )

            # The chatbot's answer is generated by sending the full prompt to the Groq API.
            response = conversation.predict(human_input=user_question)
            print("Chatbot:", response)

            with client.audio.speech.with_streaming_response.create(
                model='tts-1',
                voice='alloy',
                input=response
            ) as speech_response:
                speech_response.stream_to_file("speech.mp3")

            ps.playsound("speech.mp3")
        
def main():
    wake_word = "jarvis"
    duration = 2  # seconds
    fs = 44100  # Sample rate

    while True:
        audio = record_audio(duration, fs)
        recognized_text = recognize_speech_from_audio(audio, fs)
        if wake_word.lower() in recognized_text.lower():
            print("Wake word detected!")
            chat_with_assistant()

if __name__ == "__main__":
    main()
