# client.py
# pip install websockets sounddevice numpy
import re, asyncio, numpy as np, sounddevice as sd, websockets

WS = "ws://192.168.1.121:8765"  # Mac Studio's IP

async def ask(ws, prompt):
    print(f"\n>> {prompt}\n")
    await ws.send(prompt)
    sr, stream, text = None, None, ""
    while True:
        msg = await ws.recv()
        if isinstance(msg, str):
            if msg.startswith("META:SR="):
                sr = int(msg.split("=",1)[1]); stream = sd.OutputStream(samplerate=sr, channels=1); stream.start()
            else:
                print(msg, end="", flush=True); text += msg
                if msg.endswith("\0"): break  # not used, but keeps door open for explicit end markers
        else:
            audio = np.frombuffer(msg, dtype=np.float32)
            if stream: stream.write(audio.reshape(-1,1))
            else: sd.play(audio, sr or 22050, blocking=False)
        # the server ends when the model stops; websockets closes the message burst naturally.
        if ws.closed: break
    if stream: stream.stop(); stream.close()
    print()
    return text

async def run():
    async with websockets.connect(WS, max_size=None) as ws:
        r1 = await ask(ws, "Is a hot dog a sandwich? Answer in one sentence.")
        r2 = await ask(ws, "How many licks does it take to get to the center of a Tootsie Pop? Answer in one sentence.")
        r3 = await ask(ws, "Justify your claim for the initial prompt in one or two sentences.")
        m = re.search(r"\d[\d,]*", r2)
        follow = f"Add one more sentence about the number {m.group(0)} you gave." if m else \
                 "Add one more sentence elaborating on your key point from your previous answer about the Tootsie Pop."
        r4 = await ask(ws, follow)

if __name__ == "__main__":
    asyncio.run(run())
