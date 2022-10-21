import threading
import time

import numpy as np
import pyaudio
import questionary
import whisper
from questionary import Choice

lang = questionary.select(
    "Which langage you want to see?",
    choices=["ja", "en"],
).ask()

print("Loading model...")
model = whisper.load_model("base")
options = whisper.DecodingOptions(fp16=False, language=lang)
print("Done")


def convert(audio):
    audio = audio.flatten().astype(np.float32) / 32768.0
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)

    # decode the audio
    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(f"{max(probs, key=probs.get)}: {result.text}")


class WorkerThread(threading.Thread):
    def __init__(self, block_length, margin_length):
        super(WorkerThread, self).__init__()
        self.is_stop = False
        self.lock = threading.Lock()
        self.buffer = []
        self.result = []

        self.prev_samples = []

    def stop(self):
        self.is_stop = True
        self.join()

    def run(self):
        data = []
        while not self.is_stop:
            if len(self.buffer) > 0:
                with self.lock:
                    buf = self.buffer[0]
                    self.buffer = self.buffer[1:]

                if buf is None:
                    if len(data) > 8:
                        # 短すぎる文章はカット
                        chunk = np.concatenate(data)
                        convert(chunk)
                        data = []
                    continue
                elif len(data) > 100:
                    # ある程度長い文章は途中で送信
                    chunk = np.concatenate(data)
                    convert(chunk)
                    data = []
                    continue

                data.append(buf)
            else:
                time.sleep(0.01)

    def push_chunk(self, chunk):
        with self.lock:
            self.buffer.append(chunk)


class AudioFilter:
    def __init__(self, worker, block_length, margin_length):
        self.p = pyaudio.PyAudio()
        input_index = self.get_channels(self.p)

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

        # Status:0が待ち
        self.age = 0
        self.index = 0
        self.chunk = []
        self.buffer = []
        self.worker = worker

        self.block_length = block_length
        self.margin_length = margin_length

    def get_channels(self, p):
        # input_index = self.p.get_default_input_device_info()['index']

        selects = []
        for idx in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(idx)
            selects.append(Choice(title=info["name"], value=info["index"]))

        input_index = questionary.select(
            "Which device you want to use?",
            choices=selects,
        ).ask()

        return input_index

    def callback(self, in_data, frame_count, time_info, status):
        decoded_data = np.frombuffer(in_data, np.int16).copy()

        # 音声がONの場合、入力モードにする
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
    block_length = 4
    margin_length = 1

    worker_th = WorkerThread(block_length, margin_length)
    worker_th.setDaemon(True)
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
