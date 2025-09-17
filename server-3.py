# server.py
# pip install websockets ollama TTS numpy torch
import os, asyncio, numpy as np, websockets, torch, re
from itertools import cycle
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

# DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

TTS_MODEL_INFO = [
    ("tts_models/en/jenny/jenny", None, None),
    ("tts_models/en/ljspeech/tacotron2-DDC_ph", None, None),
]

tts_models = cycle((TTS(name).to("cpu"), speaker, lang) for name, speaker, lang in TTS_MODEL_INFO)
SENT = re.compile(r"([^.?!\n]+[.?!\n]+)")

async def stream_audio(ws, tts, speaker, lang, text, sample_rate):
    """Helper function to generate and stream audio for given text"""
    audio = np.asarray(tts.tts(text, speaker=speaker, language=lang), dtype=np.float32)
    for i in range(0, len(audio), sample_rate // 2):
        await ws.send(audio[i:i + sample_rate // 2].tobytes())


async def stream_chat(ws, client, messages, prompt):
    tts, speaker, lang = next(tts_models)
    print(f"{tts.model_name} responding to {prompt[:42]}")
    sample_rate = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)
    messages.append({"role":"user","content":prompt})
    await ws.send(f"META:SR={sample_rate}")
    buf, full = "", ""

    for part in client.chat(model=MODEL, messages=messages, stream=True):
        tok = part["message"]["content"]
        full += tok; buf += tok
        await ws.send(tok)
        for s in SENT.findall(buf):
            await stream_audio(ws, tts, speaker, lang, s, sample_rate)
        buf = SENT.sub("", buf)

    if buf.strip():
        await stream_audio(ws, tts, speaker, lang, buf, sample_rate)

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
