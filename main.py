import datetime
import io
import os
import threading
import time
from dotenv import load_dotenv

import numpy as np
from openai import OpenAI
import pyaudio
import questionary
import whisper
from scipy.io import wavfile

# Load environment variables
load_dotenv(".env")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("No OPENAI_API_KEY. Please set OPENAI_API_KEY env param.")
client = OpenAI(api_key=OPENAI_API_KEY)

# Helper functions
def create_filename(dirname):
    now = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return os.path.join(dirname, f"{now}.txt")

def get_language_choice():
    return questionary.select("Which language do you want to see?", choices=["ja", "en"]).ask()

def load_whisper_model(lang):
    print("Loading model...")
    model = whisper.load_model("base")
    options = whisper.DecodingOptions(fp16=False, language=lang)
    print("Done")
    return model, options

def convert_local(audio, model, options):
    audio = audio.flatten().astype(np.float32) / 32768.0
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    result = whisper.decode(model, mel, options)
    print(f"{max(probs, key=probs.get)}: {result.text}")

def convert_api(audio, filename=None):
    if not client.api_key:
        return
    buffer = io.BytesIO()
    audio = audio.flatten()
    wavfile.write(buffer, 16000, audio.astype(np.int16))
    buffer.seek(0)
    buffer.name = "temp.wav"
    transcript = client.audio.transcriptions.create(model="whisper-1", file=buffer)
    text = "\n".join(transcript.text.split(" "))
    print(f"\033[92m{text}\033[0m")
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "a") as f:
            f.write(text + "\n")

# Worker thread for processing audio data
class WorkerThread(threading.Thread):
    def __init__(self, block_length, margin_length, model, options, filename=None):
        super(WorkerThread, self).__init__()
        self.is_stop = False
        self.lock = threading.Lock()
        self.buffer = []
        self.model = model
        self.options = options
        self.filename = filename

    def stop(self):
        self.is_stop = True
        self.join()

    def process_buffer(self, data, all_data):
        with self.lock:
            buf = self.buffer.pop(0)

        if buf is None:
            data = self.handle_none_buffer(data)
        else:
            data, all_data = self.handle_data_buffer(data, all_data, buf)
        return data, all_data

    def handle_none_buffer(self, data):
        if len(data) > 8:
            convert_local(np.concatenate(data), self.model, self.options)
            data = []
        return data

    def handle_data_buffer(self, data, all_data, buf):
        if len(data) > 100:
            convert_local(np.concatenate(data), self.model, self.options)
            data = []
        elif len(all_data) > 1000:
            convert_api(np.concatenate(all_data), self.filename)
            all_data = []
        data.append(buf)
        all_data.append(buf)
        return data, all_data

    def run(self):
        data, all_data = [], []
        while not self.is_stop:
            if self.buffer:
                data, all_data = self.process_buffer(data, all_data)
            else:
                time.sleep(0.01)

    def push_chunk(self, chunk):
        with self.lock:
            self.buffer.append(chunk)

# Audio filter class for managing input streams
class AudioFilter:
    def __init__(self, worker, block_length, margin_length, mic_index, bypass_index, output_index, debug=False):
        self.debug = debug
        self.p = pyaudio.PyAudio()
        self.mic_index = mic_index
        self.bypass_index = bypass_index
        self.streams = [
            self.create_stream(mic_index, self.callback_mic),
            self.create_stream(bypass_index, self.callback_bypass)
        ]
        self.worker = worker
        self.block_length = block_length
        self.margin_length = margin_length
        self.age = 0
        self.chunk_data_mic = None
        self.chunk_data_bypass = None
        self.output_stream = self.create_output_stream(output_index)

    def create_stream(self, index, callback):
        return self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            frames_per_buffer=1024,
            input_device_index=index,
            output=False,
            input=True,
            stream_callback=callback,
        )

    def create_output_stream(self, output_device_index):
        return self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            frames_per_buffer=1024,
            output_device_index=output_device_index,
            output=True,
        )

    def callback_mic(self, in_data, frame_count, time_info, status):
        decoded_data = np.frombuffer(in_data, np.int16).copy()
        if self.debug:
            print(f"Mic input data: {decoded_data}")
        self.chunk_data_mic = decoded_data
        self.mix_data()

        return in_data, pyaudio.paContinue

    def callback_bypass(self, in_data, frame_count, time_info, status):
        decoded_data = np.frombuffer(in_data, np.int16).copy()
        if self.debug:
            print(f"Bypass input data: {decoded_data}")
        self.chunk_data_bypass = decoded_data
        self.mix_data()

        self.output_stream.write(in_data)
        return in_data, pyaudio.paContinue

    def mix_data(self):
        if self.chunk_data_mic is not None and self.chunk_data_bypass is not None:
            min_len = min(len(self.chunk_data_mic), len(self.chunk_data_bypass))
            truncated_data_mic = self.chunk_data_mic[:min_len]
            truncated_data_bypass = self.chunk_data_bypass[:min_len]
            mixed_audio = np.mean([truncated_data_mic, truncated_data_bypass], axis=0).astype(np.int16)
            if self.debug:
                print(f"Mixed audio input data: {mixed_audio}")
            self.worker.push_chunk(mixed_audio)
            self.chunk_data_mic = None
            self.chunk_data_bypass = None

    def close(self):
        self.p.terminate()
        self.output_stream.close()

# Main execution
def main():
    lang = get_language_choice()
    model, options = load_whisper_model(lang)
    block_length = 4
    margin_length = 1
    filename = create_filename("data")

    worker = WorkerThread(block_length, margin_length, model, options, filename)
    worker.daemon = True
    worker.start()

    mic_index = get_device_index("Select your microphone:")
    bypass_index = get_device_index("Select the device for bypass:")
    output_index = get_device_index("Select the output device for monitoring:")

    debug = questionary.confirm("Enable debug mode to print audio data?").ask()

    af = AudioFilter(worker, block_length, margin_length, mic_index, bypass_index, output_index, debug=debug)
    for stream in af.streams:
        stream.start_stream()

    try:
        while any(stream.is_active() for stream in af.streams):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    worker.stop()
    for stream in af.streams:
        stream.stop_stream()
        stream.close()
    af.close()

def get_device_index(prompt):
    p = pyaudio.PyAudio()
    devices = [
        questionary.Choice(title=p.get_device_info_by_index(idx)["name"], value=idx)
        for idx in range(p.get_device_count())
    ]
    index = questionary.select(prompt, choices=devices).ask()
    p.terminate()
    return index

if __name__ == "__main__":
    main()

