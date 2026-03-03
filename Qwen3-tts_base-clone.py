import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import numpy as np

# Load the model
model = Qwen3TTSModel.from_pretrained(
    # "/home/tiedun/models/qwen3-tts-base",
    "/root/autodl-tmp/models/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Reference audio for cloning
ref_audio = "clear.wav"
ref_text  = """
            哈喽，大家好！今天再来给大家带来几个内推啊。
            首先第一个呢，是一个嵌入式——嗯，汇川的一个内推。
            可以暴露名字了啊。它虽然说是一个外包，但是给的这个薪资待遇还可以啊。
            24届的话是能给到10~12K，23届呢能给到12~14K。
            """

            # """
            # 哈喽 大家好 今天再来给大家带来几个这个内推啊。首先第一个呢是一个嵌入式 嗯 汇川的一个内推，可以暴露名字了啊 它虽然说是一个外包 但是给的这个薪资待遇还可以啊 24届的话是能给到10~12K，23届呢是能给到12~14K。
            # """

# 额……对应的如果是25届呢 说的是暂不考虑，但是HR也说了就是 可以优化一下学历 那这个呢也是有操作空间的 而且是不进行任何背调的 啊 所以只要你足够优秀那这就可以啊 它这个工作地点呢目前是苏州，如果要感兴趣同学呢 可以直接发简历 啊 给我就可以了啊 嗯 谢谢各位

text_long = """
            AI大模型会不会取代程序员？

            眼下，AI大模型在国内外如火如荼。
            有人说：程序员搞出来大模型，终将自我反噬，最先被取代。对吗？
            显然是笑话。

            我更愿意说——程序员会是AGI时代最后被淘汰的一批人。
            因为程序开发是一个智力密集型产业，可以说，是目前人类社会里对智力要求最高的工作之一。

            这波AI大潮，有点像蒸汽机的出现、互联网技术的出现。
            它会让一批人失业，但也会创造更多新的就业机会。

            一方面，AI的出现，会大大提高程序员的门槛，淘汰掉很多低端程序员。
            另一方面，程序员的核心能力，会变成：怎么用AI解决问题。

            掌握先进工具的人，不仅不会被AI取代，而且——升职加薪不是梦。
            所以，与其天天担心，不如早点拥抱AI。

            否则，潮水退去，你才发现自己在裸泳——那就尴尬了。
            """

            # """
            # AI大模型会不会取代程序员？
            # 眼下，AI大模型在国内外如火如荼，有人说，程序员搞出来大模型，终将自我反噬，最先被取代。对吗？显然是笑话。
            # 程序员将会是AGI时代最后被淘汰的一批人。程序开发是一个智力密集型产业，可以说是目前人类社会中对智力要求最高的工作之一了。
            # 这波AI大潮，好比是蒸汽机的出现、互联网技术的出现，逐步让一批人失业，但是也创造了更多的就业机会。
            # 一方面，AI的出现，会大大提高程序员的门槛，淘汰掉很多低端程序员。
            # 另一方面，程序员的核心变成了怎么用AI 。
            # 掌握先进工具的人不仅不会被AI取代，而且“升职加薪不是梦”。
            # 所以，与其天天担心，不如早日拥抱AI。否则，潮水过后，发现自己在裸泳，那就尴尬了。
            # """

text_short = """
            各位小伙伴大家好，我是数字人 大海哥，欢迎大家收藏加关注
            """

# Generate speech
# wavs, sr = model.generate_voice_clone(
#     text=text_long,
#     language="Chinese",
#     ref_audio=ref_audio,
#     ref_text=ref_text,
# )

wavs, sr = model.generate_voice_clone(
    text=text_long,
    language="Chinese",
    ref_audio=ref_audio,
    ref_text=ref_text,
    # 关键：限制长度 + 抑制重复
    # max_new_tokens=4096,        # 先别放太大（按你文本长度再调）
    repetition_penalty=1.21,    # 1.10~1.25 常用
    # do_sample=True,
    # temperature=0.8,            # 0.7~0.95
    # top_p=0.95,                 # 0.9~1.0
    # top_k=40,                   # 20~50
)

# Save the resulting audio
# sf.write("output_voice_clone.wav", wavs[0], sr)

audio = wavs[0]
if hasattr(audio, "detach"):
    audio = audio.detach().cpu().float().numpy()
audio = np.asarray(audio, dtype=np.float32)

# 可选：去掉 DC
audio = audio - np.mean(audio)

# 可选：归一化（避免整体太小）
peak = np.max(np.abs(audio)) + 1e-9
audio = audio / peak * 0.95

sf.write("res_1.7B.wav", audio, sr, subtype="PCM_16")
