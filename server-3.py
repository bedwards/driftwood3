# server.py
# pip install websockets ollama TTS numpy torch
import os, asyncio, numpy as np, websockets, torch, re
from ollama import Client
from TTS.api import TTS

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
HOST = OLLAMA_HOST if OLLAMA_HOST.startswith("http") else f"http://{OLLAMA_HOST}"

# MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
# MODEL="gemma2:9b"
# MODEL="qwen2.5:7b"
# MODEL="deepseek-r1:7b"
MODEL="mistral:7b"
# MODEL="llama3.2:3b"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
assert DEVICE == "mps"

# crash
# tts = TTS("tts_models/en/ljspeech/vits").to(DEVICE)
# tts = TTS("tts_models/en/vctk/vits").to(DEVICE)

# no male/female?
# tts = TTS("tts_models/en/ljspeech/glow-tts").to(DEVICE)

# tts = TTS("tts_models/en/ljspeech/tacotron2-DDC").to(DEVICE)
# tts = TTS("tts_models/en/ljspeech/fastspeech2").to(DEVICE)
# tts = TTS("tts_models/en/ljspeech/glow-tts").to(DEVICE)

# tts = TTS("tts_models/multilingual/multi-dataset/your_tts").to(DEVICE)
tts = TTS("tts_models/en/vctk/fast_pitch").to(DEVICE)

SR = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)

# Male: p225, p228, p236
# Female: p262, p270, p315
FEMALE, MALE = "p315", "p236"

turn = {"n": 0}
SENT = re.compile(r"([^.?!\n]+[.?!\n]+)")


async def stream_audio(ws, text, voice):
    """Helper function to generate and stream audio for given text"""
    audio = np.asarray(tts.tts(text, speaker=voice), dtype=np.float32)
    for i in range(0, len(audio), SR // 2):
        await ws.send(audio[i:i + SR // 2].tobytes())


async def stream_chat(ws, client, messages, prompt):
    messages.append({"role":"user","content":prompt})
    await ws.send(f"META:SR={SR}")
    buf, full = "", ""
    voice = FEMALE if (turn["n"] % 2 == 0) else MALE

    for part in client.chat(model=MODEL, messages=messages, stream=True):
        tok = part["message"]["content"]
        full += tok; buf += tok
        await ws.send(tok)
        for s in SENT.findall(buf):
            await stream_audio(ws, s, voice)
        buf = SENT.sub("", buf)

    if buf.strip():
        await stream_audio(ws, buf, voice)

    turn["n"] += 1
    messages.append({"role":"assistant","content":full})
    await ws.send("END")

async def handler(ws):
    client = Client(host=HOST); messages = []
    async for prompt in ws:
        await stream_chat(ws, client, messages, prompt)

async def main():
    port = 8765
    print(f'Listening on {port}')
    async with websockets.serve(
            handler, "0.0.0.0", port, max_size=None, ping_interval=None):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
