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


load_dotenv(".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

if len(OPENAI_API_KEY):
    print("No OPENAI_API_KEY. Please set OPENAI_API_KEY env param.")
client = OpenAI(api_key=OPENAI_API_KEY)


def create_filename(dirname):
    now = datetime.datetime.utcnow()
    now = now.strftime("%Y%m%d_%H%M%S")
    return os.path.join(dirname, f"{now}.txt")


def get_language_choice():
    return questionary.select(
        "Which language do you want to see?",
        choices=["ja", "en"],
    ).ask()


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
    if len(client.api_key) == 0:
        return

    buffer = io.BytesIO()
    audio = audio.flatten()
    wavfile.write(buffer, 16000, audio.astype(np.int16))
    buffer.seek(0)
    buffer.name = "temp.wav"
    transcript = client.audio.transcriptions.create(model="whisper-1", file=buffer)
    text = transcript.text
    text = "\n".join(text.split(" "))
    print(f"\033[92m{text}\033[0m")

    if filename:
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        with open(filename, "a") as f:
            f.write(text + "\n")


class WorkerThread(threading.Thread):
    def __init__(self, block_length, margin_length, model, options, filename=None):
        super(WorkerThread, self).__init__()
        self.is_stop = False
        self.lock = threading.Lock()
        self.buffer = []
        self.result = []
        self.prev_samples = []
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
        data = []
        all_data = []

        while not self.is_stop:
            if self.buffer:
                data, all_data = self.process_buffer(data, all_data)
            else:
                time.sleep(0.01)

    def push_chunk(self, chunk):
        with self.lock:
            self.buffer.append(chunk)


class AudioFilter:
    def __init__(self, worker, block_length, margin_length):
        self.p = pyaudio.PyAudio()
        input_index = self.get_input_device_index()

        self.channels = 1
        self.rate = 16000
        self.format = pyaudio.paInt16
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            frames_per_buffer=1024,
            input_device_index=input_index,
            output=False,
            input=True,
            stream_callback=self.callback,
        )

        self.age = 0
        self.worker = worker
        self.block_length = block_length
        self.margin_length = margin_length

    def get_input_device_index(self):
        selects = [
            questionary.Choice(
                title=self.p.get_device_info_by_index(idx)["name"], value=idx
            )
            for idx in range(self.p.get_device_count())
        ]
        return questionary.select(
            "Which device do you want to use?",
            choices=selects,
        ).ask()

    def callback(self, in_data, frame_count, time_info, status):
        decoded_data = np.frombuffer(in_data, np.int16).copy()
        if decoded_data.max() > 400:
            self.age = self.block_length
        else:
            self.age = max(0, self.age - 1)
        if self.age == 0:
            self.worker.push_chunk(None)
        else:
            self.worker.push_chunk(decoded_data)

        return in_data, pyaudio.paContinue

    def close(self):
        self.p.terminate()


if __name__ == "__main__":
    lang = get_language_choice()
    #use_api = get_api_usage_choice()

    #if use_api:
    #    set_openai_api_key()

    model, options = load_whisper_model(lang)

    block_length = 4
    margin_length = 1

    filename = create_filename("data")

    worker_th = WorkerThread(block_length, margin_length, model, options, filename)
    worker_th.daemon = True
    worker_th.start()

    af = AudioFilter(worker_th, block_length, margin_length)
    af.stream.start_stream()

    try:
        while af.stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    worker_th.stop()
    af.stream.stop_stream()
    af.stream.close()
    af.close()
