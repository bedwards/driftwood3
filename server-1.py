# server.py
# pip install websockets ollama TTS numpy torch
import os, re, asyncio, numpy as np, websockets, torch
from ollama import Client
from TTS.api import TTS

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
OLLAMA_HOST = OLLAMA_HOST if OLLAMA_HOST.startswith("http") else f"http://{OLLAMA_HOST}"
MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

tts = TTS("tts_models/en/ljspeech/vits").to(DEVICE)
SR = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)

async def stream_chat(ws, client, messages, user_prompt):
    messages.append({"role":"user","content":user_prompt})
    buf, full = "", ""
    await ws.send(f'META:SR={SR}')
    async for part in client.chat(model=MODEL, messages=messages, stream=True):
        tok = part["message"]["content"]
        full += tok; buf += tok
        await ws.send(tok)  # text token stream
        i = max(buf.rfind("."), buf.rfind("!"), buf.rfind("?"), buf.rfind("\n"))
        if i != -1:
            audio = np.asarray(tts.tts(buf[:i+1]), dtype=np.float32)
            await ws.send(audio.tobytes())  # binary frame
            buf = buf[i+1:]
    if buf.strip():
        audio = np.asarray(tts.tts(buf), dtype=np.float32)
        await ws.send(audio.tobytes())
    messages.append({"role":"assistant","content":full})

async def handler(ws):
    client = Client(host=OLLAMA_HOST)
    messages = []
    async for prompt in ws:                  # each received text message is a new prompt
        await stream_chat(ws, client, messages, prompt)

async def main():
    async with websockets.serve(handler, "0.0.0.0", 8765, ping_interval=None, max_size=None):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
