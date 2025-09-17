# client.py
# pip install websockets sounddevice numpy
import random
from itertools import cycle
import asyncio, websockets, sounddevice as sd, numpy as np, re
from config import websockets_kwargs
from topics import topics

WS = "ws://192.168.1.121:8765"
n_context = 5  # number of previous messages from the convo given as context in the prompt

topic = random.choice(topics)

thinkers = [
    "Jason Hickel",
    "David Graeber",
    "Slavoj Žižek",
    "Ivan Krastev",
    "Stephen Holmes",
    "Robert Kagan",
    "David Hume",
    "Michel Foucault",
    "Daniel Kahneman",
    "Amos Tversky",
    "Richard Rorty",
    "Jürgen Habermas",
    "Harry Frankfurt",
    "Jonathan Haidt",
    "Bruno Latour",
    "Miranda Fricker",
    "Jason Stanley",
    "Timothy Snyder",
    "W. V. O. Quine",
    "Dan Kahan",
    "Richard Bernstein",
    "Linda Zagzebski",
    "Bernard Williams",
    "Plato",
    "Aristotle",
    "Marcus Aurelius",
    "Epicurus",
    "Siddhartha Gautama (Buddha)",
    "Confucius",
    "Lao Tzu",
    "Baruch Spinoza",
    "René Descartes",
    "David Hume",
    "Immanuel Kant",
    "Søren Kierkegaard",
    "Friedrich Nietzsche",
    "Ludwig Wittgenstein",
    "Jean-Paul Sartre",
    "Simone de Beauvoir",
    "Hannah Arendt",
    "Rumi",
    "Martha Nussbaum",
]

writers = [
    "Roberto Bolaño",
    "Denis Johnson",
    "Cormac McCarthy",
    "George Orwell",
    "Fernando Pessoa",
    "Percival Everett",
    "Rachel Cusk",
    "Leonora Carrington",
    "Robert Kagan",
    "Ted Chiang",
    "J. M. Coetzee",
    "Joy Williams",
    "Karl Ove Knausgaard",
    "Anne Carson",
    "Zadie Smith",
    "Jane Austen",
    "Charles Dickens",
    "Oscar Wilde",
    "Ernest Hemingway",
    "Virginia Woolf",
    "James Joyce",
    "Franz Kafka",
    "J.R.R. Tolkien",
    "Kurt Vonnegut",
    "Jorge Luis Borges",
    "Gabriel García Márquez",
    "James Baldwin",
    "Italo Calvino",
    "Toni Morrison",
    "Ursula K. Le Guin",
    "Haruki Murakami",
    "Kazuo Ishiguro",
    "Margaret Atwood",
    "Octavia Butler",
    "Thomas Pynchon",
]


personas = [
    {
        "name": "Cindy",
        "thinker": random.choice(thinkers),
        "writer": random.choice(writers),
    },
    {
        "name": "Rebecca",
        "thinker": random.choice(thinkers),
        "writer": random.choice(writers),
    },
]

while personas[0]["thinker"] == personas[1]["thinker"]:
    print(".")
    personas[1]["thinker"] = random.choice(thinkers)

while personas[0]["writer"] == personas[1]["writer"]:
    print("o")
    personas[1]["writer"] = random.choice(writers)


def create_prompt(persona, convo):
    n_sentences = int(abs(random.gauss(1)) * 3) + 1
    prompt = f"""
    Write the next line of dialogue ({n_sentences} sentences.)
    Flow from the coversation.
    """
    if len(convo) == 0:
        prompt += f"""
        The topic is {topic}. But do not include the words found in the topic directly in your reponse.
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
    Extend the conversation in a realistic, believable way that this character would.
    Do not get stuck in repitition, segue into unexpected depths.
    Do not write parentheticals.
    Do not lead with '{persona["name"]}: '

    JUST WRITE THE SPOKEN DIALOGUE!!! NOTHING ELSE!!! NOTHING THE CHARACTER DOES NOT SPEAK!!!

    Write the next line of dialogue ({n_sentences} sentences.)
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
            Do not give examples.
            """
            persona['description'] = await ask(ws, prompt)
        convo = []
        for persona in cycle(personas):
            prompt = create_prompt(persona, convo)
            convo.append(await ask(ws, prompt))
            convo = convo[-n_context:]

if __name__ == "__main__":
    asyncio.run(run())
