from transformers import pipeline

pipe = pipeline(
    "automatic-speech-recognition",
    model="/home/tiedun/models/whisper-tiny",
    device="cuda:0"  # 没有GPU就改成 "cpu"
)

# 支持 wav/mp3/flac 等（依赖系统 ffmpeg/解码能力；wav 最省事）
result = pipe(
    "/home/tiedun/ffmpeg_workspace/clear_res.wav",
    return_timestamps=True,
    generate_kwargs={"language": "zh", "task": "transcribe"})
print(result["text"])
