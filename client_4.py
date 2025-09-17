# client.py
# pip install websockets sounddevice numpy
from itertools import cycle
import asyncio, websockets, sounddevice as sd, numpy as np, re
from config import websockets_kwargs

WS = "ws://192.168.1.121:8765"
n_context = 22  # number of previous messages from the convo given as context in the prompt

topic = "Certainty and scarcity in jason Hickel's Less is More and Roberto Bolaño's 2666"

personas = [
    {
        "name": "Cindy",
        "thinker": "Miranda Fricker",
        "writer": "Leonora Carrington",
    },
    {
        "name": "Rebecca",
        "thinker": "Linda Zagzebski",
        "writer": "Zadie Smith",
    },
]


def create_prompt(persona, convo):
    prompt = f"""
    Write the next line of dialogue (one to three sentences.)
    The topic is {topic}.
    Flow from the coversation.
    """
    if len(convo) % 2 == 0:
        speakers = ["You", personas[1]["name"]]
    else:
        speakers = [personas[0]["name"], "You"]
    for speaker, resp in zip(cycle(speakers), convo):
        prompt += f"""
        {speaker}: "{resp}"
        """
    return prompt + f"""
    The ideas and position and argumentative style of your response should spring forth as if coming from {persona["thinker"]}
    But they should be expressed as if coming from a fictional character's dialogue written by {persona["writer"]}
    with the characters communication style described as {persona["description"]}
    Do not print commentary, or pose meta-level questions.
    Do not put content in quotes. Just print what would be enclosed in quotes in a novel or short story by {persona["writer"]}.
    Do not write any text that would appear outside the quotes (do not print- I said,)
    Write the next line of dialogue (one to three sentences.)
    """

async def ask(ws, prompt):
    print(f"\n>> {prompt}\n")
    await ws.send(prompt)
    text, sr, stream = "", None, None
    while True:
        msg = await ws.recv()
        if isinstance(msg, str):
            if msg.startswith("META:SR="):
                sr = int(msg.split("=",1)[1]); stream = sd.OutputStream(samplerate=sr, channels=1); stream.start()
            elif msg == "END":
                if stream: stream.stop(); stream.close()
                print()  # newline after final text
                return text
            else:
                print(msg, end="", flush=True); text += msg
        else:
            audio = np.frombuffer(msg, dtype=np.float32).reshape(-1,1)
            if stream: stream.write(audio)
            else: sd.play(audio, sr or 22050, blocking=False)

async def run():
    async with websockets.connect(WS, **websockets_kwargs) as ws:
        for persona in personas:
            prompt = f"""
            Act as if you are {persona["writer"]}.
            Create a female character that would fit in one of your books.
            Not a generic character, one that might fill the edges.
            Describe a detailed manner of speaking that this character exhibits.
            Give instructions that another writer could use to write dialogue coming from this character.
            Stick to the manner in which the character speaks, not their beliefs or values.
            Briefly. No meta-commentary. Only instructions to a would-be writer.
            Do not give her a name. Her name is {persona["name"]}
            """
            persona['description'] = await ask(ws, prompt)
        convo = []
        for persona in cycle(personas):
            prompt = create_prompt(persona, convo)
            convo.append(await ask(ws, prompt))
            convo = convo[-n_context:]

if __name__ == "__main__":
    asyncio.run(run())
