# server.py
# pip install websockets ollama TTS numpy torch
import os, asyncio, numpy as np, websockets, torch, re
from itertools import cycle
from ollama import Client
from TTS.api import TTS

OLLAMA_HOST = "127.0.0.1:11434"

# gemma2:9b
# qwen2.5:7b
# deepseek-r1:7b
# mistral:7b
# llama3.2:3b

MODEL_INFO = [
    ("gemma2:9b", "tts_models/en/jenny/jenny", None, None),
    ("qwen2.5:7b", "tts_models/en/ljspeech/tacotron2-DDC_ph", None, None),
]

model = cycle(
    (llm, TTS(name).to("cpu"), speaker, lang)
    for llm, name, speaker, lang in MODEL_INFO
)

SENT = re.compile(r"([^.?!\n]+[.?!\n]+)")

async def stream_audio(ws, tts, speaker, lang, text, sample_rate):
    """Helper function to generate and stream audio for given text"""
    # Skip empty or whitespace-only text
    if not text or not text.strip():
        return
        
    try:
        audio = np.asarray(tts.tts(text.strip(), speaker=speaker, language=lang), dtype=np.float32)
        # Check for empty audio array
        if len(audio) == 0:
            return
    except (ZeroDivisionError, RuntimeError) as e:
        print(f"{e!r} in {tts.model_name} - skipping text: '{text[:50]}'")
        return
    except Exception as e:
        print(f"Unexpected TTS error: {e!r} - skipping text: '{text[:50]}'")
        return
        
    for i in range(0, len(audio), sample_rate // 2):
        try:
            await ws.send(audio[i:i + sample_rate // 2].tobytes())
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed during audio streaming")
            break


async def stream_chat(ws, client, messages, prompt):
    try:
        llm, tts, speaker, lang = next(model)
        print(f"{llm}.. {tts.model_name} responding to {prompt[:42]}")
        sample_rate = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 22050)
        messages.append({"role":"user","content":prompt})
        await ws.send(f"META:SR={sample_rate}")
        buf, full = "", ""

        for part in client.chat(model=llm, messages=messages, stream=True):
            tok = part["message"]["content"]
            full += tok; buf += tok
            await ws.send(tok)
            for s in SENT.findall(buf):
                s_clean = s.strip()
                if len(s_clean) > 3:  # Only process substantial sentences
                    await stream_audio(ws, tts, speaker, lang, s_clean, sample_rate)
            buf = SENT.sub("", buf)

        if buf.strip() and len(buf.strip()) > 3:
            await stream_audio(ws, tts, speaker, lang, buf.strip(), sample_rate)

        messages.append({"role":"assistant","content":full})
        await ws.send("END")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected during streaming")
        return
    except Exception as e:
        print(f"Stream chat error: {e!r}")
        try:
            await ws.send("END")  # Try to signal end
        except:
            pass  # Connection already dead


async def handler(ws):
    client = Client(host=OLLAMA_HOST); messages = []
    async for prompt in ws:
        await stream_chat(ws, client, messages, prompt)

async def main():
    port = 8765
    print(f'Listening on {port}')

    async with websockets.serve(
                handler,
                "0.0.0.0",
                port,
                ping_interval=30,     # Send ping every n seconds
                ping_timeout=300,     # Wait n seconds for pong
                close_timeout=15,     # Time to wait for clean close
                max_size=1048576,     # n MB message limit
                max_queue=32,         # Limit queued messages
                compression=None      # Disable compression for lower latency
            ):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
