# Neurons

**Neurons** is a Python library that combines multiple AI capabilities into a single, easy-to-use interface. It includes support for **LLMs (Large Language Models)**, **TTS (Text-to-Speech)**, **STT (Speech-to-Text)**, **Wake Word Detection**, and more. Neurons is designed to simplify the integration of powerful AI tools into your projects with minimal effort.

---

## Features

- **LLMs (Large Language Models)**: Chat with AI models and maintain conversational context.
- **TTS (Text-to-Speech)**: Generate speech from text with voice cloning support.
- **STT (Speech-to-Text)**: Convert spoken words into text using Whisper.
- **Wake Word Detection**: Trigger interactions by detecting predefined wake words.
- **Flexible and Extensible**: Built with modularity to support additional AI tools in the future.

---

## Installation

To install Neurons locally using `pip`:

1. Clone or download the repository.
2. Run the following command:

   ```bash
   pip install .
   ```

   This will install Neurons along with its dependencies.

---

## Quick Start

### Example 1: Text-Based Chat with an LLM

```python
import neurons

llm = neurons.LLM("mistral")  # Initialize an LLM using the Mistral model

while True:
    question = input("USER >>> ")  # Get a prompt from the user
    response = llm.conversation(question)  # Ask the LLM
    print("LLM >>> " + response)  # Print out the response
```

### Example 2: Voice Assistant with TTS and STT

```python
import neurons

llm = neurons.LLM("mistral")  # Initialize an LLM
stt = neurons.STT("base")  # Initialize STT with Whisper's base model
tts = neurons.TTS("tts_models/multilingual/multi-dataset/xtts_v2")  # Initialize TTS
tts.set_speaker("voices/glados.wav")  # Set the voice XTTS should mimic

while True:
    command = stt.speech()  # Record and transcribe speech
    print("USER >>> ", command, "\n")
    response = llm.conversation(command)  # Generate a response from the LLM
    print("ROBOT >>> ", response, "\n")
    tts.speak(response)  # Speak the response aloud
```

---

## Documentation

### LLM (Large Language Model)

- **Description**: Interact with an LLM using simple prompts and maintain conversational context.
- **Methods**:
  - `LLM(model_name: str)`: Initialize the LLM with the specified Ollama model.
  - `chat(prompt: str) -> str`: Get a response from the LLM.
  - `conversation(prompt: str) -> str`: Maintain a conversation context with the LLM.

### TTS (Text-to-Speech)

- **Description**: Generate speech from text with voice cloning.
- **Methods**:
  - `TTS(model_name: str, gpu: bool = True)`: Initialize the TTS model.
  - `set_speaker(speaker_wav: str)`: Set a voice cloning sample.
  - `synthesize(text: str, output_file: str, language: str = "en")`: Save generated speech to a file.
  - `speak(text: str, language: str = "en")`: Generate speech and play it automatically.

### STT (Speech-to-Text)

- **Description**: Convert audio into text with support for recording.
- **Methods**:
  - `STT(model_name: str = "base")`: Initialize the STT model.
  - `speech() -> str`: Record and transcribe speech.
  - `record(duration: int) -> str`: Record for a fixed duration and transcribe.

---

## Roadmap

Neurons is actively being developed, with plans to expand its capabilities. Future features include:

- **Image Generation**: Generate images based on prompts.
- **Video Generation**: Create videos from text or other inputs.
- **TTS Voice Cloning Enhancements**: Improve speaker adaptation and voice cloning.
- **API Integration**: Simplify interactions with external APIs for more extensibility.
- **OCR (Optical Character Recognition)**: Extract text from images for real-world applications.

---

## Why Use Neurons?

Neurons combines multiple AI tools into a single library, simplifying the process of integrating advanced AI capabilities into your projects.

---

## Contributing

I welcome contributions to improve Neurons! Feel free to submit issues or pull requests.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.