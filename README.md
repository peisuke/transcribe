# transcribe

## 概要

- PCの音声と自分の声をミックスして、文字起こし・翻訳を行う
- BlackHoleをインストールしておく
  - https://github.com/ExistentialAudio/BlackHole
- BlackHoleとは
  -　MacOS向けの仮想オーディオループバックドライバー、アプリケーションがオーディオを他のアプリケーションに遅延なしで渡すことができる。

## Setting

- アプリの出力をBlackHole 2chとしておく

## How to run

- Select your microphoneの質問に、自分のマイクのデバイスを設定
- Select the device for bypassの質問に、blackhole 16chを設定
- Select the output device for monitoringの質問に、自分のヘッドフォンのデバイスを設定
