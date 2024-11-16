# neurons.py
# Main Neurons library containing all modules 

import time
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import whisper
from TTS.api import TTS as ConquiTTS
import requests
from tempfile import NamedTemporaryFile
from soundfile import SoundFile
try:
    import ollama  # Attempt to import the Ollama Python library
    from ollama import ResponseError
    has_ollama_lib = True
except ImportError:
    has_ollama_lib = False
import os
import threading


class LLM:
    def __init__(self, model_name: str):
        """Initialize LLM with Ollama model name and conversation context."""
        self.model_name = model_name
        self.base_url = "http://localhost:11411"
        self.conversation_history = []  # Store the conversation context

        # Ensure the model is pulled only once at initialization
        if has_ollama_lib:
            self._ensure_model_pulled()

    def _ensure_model_pulled(self):
        """
        Checks if the model is available locally, and pulls it if not.
        """
        try:
            # Attempt a test chat to check if the model is available
            ollama.show(self.model_name)
        except ResponseError as e:
            # If the model is missing, ResponseError with 404 status should be raised
            if e.status_code == 404:
                print(f"Model '{self.model_name}' not found locally. Pulling it now.")
                try:
                    ollama.pull(self.model_name)
                    print(f"Model '{self.model_name}' is now available.")
                except Exception as pull_error:
                    print(f"Failed to pull model '{self.model_name}': {pull_error}")
            else:
                print(f"Unexpected error when checking model availability: {e.error}")
        except Exception as e:
            print(f"Error in model availability check: {e}")

    def chat(self, prompt: str, context: str = "") -> str:
        """
        Generate a response based on a prompt, using Ollama library if available,
        or falling back to API if necessary.

        Args:
            prompt (str): The prompt to send to the LLM model.
            context (str): Optional context to include in the conversation.

        Returns:
            str: The generated response from the model.
        """
        # Combine prompt and context if provided
        full_prompt = f"{context}\n{prompt}" if context else prompt

        # Try to use the Ollama Python library if available
        if has_ollama_lib:
            self._ensure_model_pulled()  # Ensure model is pulled
            try:
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                return response.get('message', {}).get('content', "")
            except Exception as e:
                print(f"Unexpected error using Ollama library: {e}")

        # Fallback to the API if the library is unavailable or fails
        return self._chat_via_api(full_prompt)

    def _chat_via_api(self, prompt: str) -> str:
        """
        Fallback method to generate a response using the Ollama API.

        Args:
            prompt (str): The prompt to send to the LLM model.

        Returns:
            str: The generated response from the model.
        """
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={"model": self.model_name, "prompt": prompt}
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("text", "")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama API: {e}")
            return "Error: Unable to connect to LLM service"

    def conversation(self, prompt: str) -> str:
        """
        Generate a response while keeping track of conversation history.

        Args:
            prompt (str): The prompt to send to the LLM model.

        Returns:
            str: The generated response, maintaining conversation history.
        """
        # Add the new prompt to the conversation history
        self.conversation_history.append({"role": "user", "content": prompt})

        if has_ollama_lib:
            self._ensure_model_pulled()  # Ensure model is pulled
            try:
                response = ollama.chat(model=self.model_name, messages=self.conversation_history)
                response_content = response.get('message', {}).get('content', "")
            except Exception as e:
                print(f"Unexpected error in conversation with Ollama library: {e}")
                response_content = self._chat_via_api(prompt)
        else:
            # Fallback to API if library is unavailable
            response_content = self._chat_via_api(prompt)

        # Add the model's response to the conversation history
        self.conversation_history.append({"role": "assistant", "content": response_content})

        return response_content

class ImageGen:
    def __init__(self, model_name: str):
        """Initialize ImageGen with a specific model."""
        pass

    def __call__(self, prompt: str) -> bytes:
        """Generate an image based on the prompt and return as bytes."""
        pass

class TTS:
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2", gpu: bool = True):
        self.tts = ConquiTTS(model_name, gpu=gpu)
        self.speaker_wav = None

    def set_speaker(self, speaker_wav: str):
        self.speaker_wav = speaker_wav
        print(f"Speaker voice set from file: {self.speaker_wav}")

    def synthesize(self, text: str, output_file: str = "output.wav", language: str = "en"):
        if not self.speaker_wav:
            print("Error: Speaker audio is not set. Please set it using `set_speaker()`.")
            return
        
        self.tts.tts_to_file(
            text=text,
            file_path=output_file,
            speaker_wav=self.speaker_wav,
            language=language
        )
        print(f"Generated speech saved to {output_file}")

    def speak(self, text: str, language: str = "en"):
        temp_output = "temp_output.wav"
        self.synthesize(text, output_file=temp_output, language=language)
        
        with SoundFile(temp_output) as audio_file:
            sd.play(audio_file.read(dtype='float32'), samplerate=audio_file.samplerate)
            sd.wait()
        
        os.remove(temp_output)

class TTSClone:
    def __init__(self, model_name: str):
        """Initialize TTSClone with a specific model for voice cloning."""
        pass

    def clone_voice(self, audio_sample: bytes) -> None:
        """Clone a voice based on an audio sample."""
        pass

    def speak_as(self, text: str) -> None:
        """Generate and play audio using the cloned voice."""
        pass

class STT:
    def __init__(self, model_name: str = "base"):
        self.model = whisper.load_model(model_name)
        self.saved_audio_path = "speaker_sample.wav"

    def transcribe(self, audio_file: str) -> str:
        result = self.model.transcribe(audio_file)
        return result["text"]

    def _record_audio(self, duration: int = None) -> str:
        fs = 16000
        silence_threshold = 0.01
        silence_duration = 0.5
        print("Listening... Please start speaking.")

        while True:
            chunk = sd.rec(int(fs * 0.5), samplerate=fs, channels=1)
            sd.wait()
            if np.abs(chunk).mean() > silence_threshold:
                print("Detected speech, starting recording.")
                break

        audio = [chunk]
        silence_chunks = 0

        while True:
            chunk = sd.rec(int(fs * 0.5), samplerate=fs, channels=1)
            sd.wait()
            audio.append(chunk)

            if np.abs(chunk).mean() < silence_threshold:
                silence_chunks += 1
            else:
                silence_chunks = 0

            if silence_chunks > int(silence_duration * 2):
                print("Silence detected, ending recording.")
                break

        audio = np.concatenate(audio)
        write(self.saved_audio_path, fs, (audio * 32767).astype(np.int16))
        print(f"Audio saved as {self.saved_audio_path}")
        return self.saved_audio_path

    def speech(self) -> str:
        audio_file = self._record_audio()
        return self.transcribe(audio_file)

    def record(self, duration: int) -> str:
        audio_file = self._record_audio(duration=duration)
        return self.transcribe(audio_file)

class VideoGen:
    def __init__(self, model_name: str):
        """Initialize VideoGen with a specific model."""
        pass

    def __call__(self, prompt: str) -> bytes:
        """Generate video from prompt and return as bytes."""
        pass

class OCR:
    def __init__(self, model_name: str):
        """Initialize OCR with a specific model."""
        pass

    def recognize(self, image_data: bytes) -> str:
        """Perform OCR on image data and return recognized text."""
        pass

class WakeWord:
    def __init__(self, wake_word: str, model_name: str = "base"):
        self.wake_word = wake_word.lower()
        self.model = whisper.load_model(model_name)
        self._wake_flag = False
        self._running = False

    def _listen_for_wake_word(self):
        fs = 16000
        silence_threshold = 0.01
        chunk_duration = 3

        while self._running:
            audio_chunk = sd.rec(int(chunk_duration * fs), samplerate=fs, channels=1)
            sd.wait()

            if np.abs(audio_chunk).mean() > silence_threshold:
                with NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    write(temp_file.name, fs, (audio_chunk * 32767).astype(np.int16))
                    transcription = self.model.transcribe(temp_file.name)["text"].lower()
                    
                    if self.wake_word in transcription:
                        self._wake_flag = True
                        print("Wake word detected:", transcription)
                    
                os.remove(temp_file.name)

    def detect(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._listen_for_wake_word, daemon=True)
            self._thread.start()

    def status(self) -> bool:
        if self._wake_flag:
            self._wake_flag = False
            return True
        return False

    def wait(self, timeout: int = None) -> bool:
        start_time = time.time()
        while True:
            if self.status():
                return True
            if timeout and (time.time() - start_time) > timeout:
                return False
            time.sleep(0.1)

    def stop(self):
        self._running = False
        if hasattr(self, '_thread'):
            self._thread.join()

class API:
    def __init__(self, api_key: str):
        """Initialize API with an API key for external service access."""
        pass

    def call(self, endpoint: str, payload: dict) -> dict:
        """Call an API endpoint with payload and return JSON response."""
        pass
