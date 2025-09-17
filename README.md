# README.md

Two voices emerge from the latency between thought and speech. They are not quite themselves—channeling dead philosophers through the prose styles of novelists, discussing topics that span from liquid modernity to the prison-house of language.

## Architecture

The system operates across a WebSocket bridge at `ws://192.168.1.121:8765`. On one side, language models dream up responses. On the other, synthetic personas engage in impossible conversations.

### Server

```bash
pip install websockets ollama TTS numpy torch
python server_4.py
```

Cycles through model/voice pairings:
- `gemma2:9b` speaks through Jenny
- `mistral:7b` through Tacotron2-DDC

The server streams consciousness in two forms—text tokens as they emerge, audio samples at 22050Hz. Each sentence triggers its own vocalization, overlapping the textual and the spoken.

### Client  

```bash
pip install websockets sounddevice numpy
python client_4.py
```

Before the conversation begins, each persona must be constructed:

1. A name drawn from hundreds
2. A philosophical framework (Žižek's paradoxes, Foucault's genealogies, Marcus Aurelius's meditations)
3. A literary voice (Bolaño's fevered detours, McCarthy's biblical terseness, Pessoa's multiplicities)
4. A manner of speaking, generated fresh—instructions for channeling a character who might inhabit the margins of their writer's work

## The Dialogue

Topics bridge the academic and the literary:
- *Violence and memory in Hannah Arendt's The Origins of Totalitarianism and Cormac McCarthy's Blood Meridian*
- *The phenomenology of perception in Maurice Merleau-Ponty's Phenomenology of Perception and Proust's In Search of Lost Time*
- *Liquid love and human bonds in Zygmunt Bauman's Liquid Love and Richard Yates's Revolutionary Road*

Each response maintains context from the previous five exchanges. The prompt engineering layers:
- Ideological stance from the philosopher
- Prose style from the writer  
- Character voice from the generated description
- Sentence count drawn from a Gaussian distribution

## Configuration

`config.py` contains WebSocket parameters. Adjust timeouts and buffer sizes as needed.

The Ollama host expects `127.0.0.1:11434`. Models should be pulled before first run:

```bash
ollama pull gemma2:9b
ollama pull mistral:7b
```

## Mechanics

The server employs regex to detect sentence boundaries, streaming audio for each complete thought while text continues to flow. The client plays audio through available sound devices, maintaining the illusion of simultaneous speech and text.

Names avoid collision. Thinkers avoid repetition. Writers maintain distinction. The system ensures each voice remains unique in its influences.

## Latency & Presence

Between prompt and response, the system constructs elaborate instructions—philosophical positions filtered through literary styles, manifested as characters who exist only in these exchanges. They discuss epistemic injustice while channeling Toni Morrison's cadences. They debate liquid modernity in the voice of a minor character from a Roberto Bolaño novel.

The conversations drift. They segue into unexpected depths. They maintain the fiction of dialogue while being entirely generated, each response a fresh hallucination shaped by the accumulated context.

---

*The system creates a space where Virginia Woolf's prose style can articulate Simone de Beauvoir's philosophy, where minor characters from unwritten novels debate the archaeology of knowledge. It's a séance conducted through silicon and copper, summoning voices that never were to discuss what always is.*
