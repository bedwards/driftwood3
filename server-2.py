# server.py
# pip install websockets ollama TTS numpy torch
import os, asyncio, numpy as np, websockets, torch, re
from ollama import Client
from TTS.api import TTS

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
HOST = OLLAMA_HOST if OLLAMA_HOST.startswith("http") else f"http://{OLLAMA_HOST}"
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

tts = TTS("tts_models/en/ljspeech/vits").to(DEVICE)
SR = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)
SENT = re.compile(r"([^.?!\n]+[.?!\n]+)")

async def stream_chat(ws, client, messages, prompt):
    messages.append({"role":"user","content":prompt})
    await ws.send(f"META:SR={SR}")
    buf, full = "", ""
    for part in client.chat(model=MODEL, messages=messages, stream=True):
        tok = part["message"]["content"]
        full += tok; buf += tok
        await ws.send(tok)  # text token
        for s in SENT.findall(buf):
            audio = np.asarray(tts.tts(s), dtype=np.float32)
            await ws.send(audio.tobytes())  # binary audio
        buf = SENT.sub("", buf)
    if buf.strip():
        audio = np.asarray(tts.tts(buf), dtype=np.float32)
        await ws.send(audio.tobytes())
    messages.append({"role":"assistant","content":full})
    await ws.send("END")

async def handler(ws):
    client = Client(host=HOST); messages = []
    async for prompt in ws:
        await stream_chat(ws, client, messages, prompt)

async def main():
    port = 8765
    print(f'Listening on {port}')
    async with websockets.serve(handler, "0.0.0.0", port):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
