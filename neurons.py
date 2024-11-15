# neurons.py
# Main Neurons library containing all modules 
# neurons.py
try:
    import ollama  # Attempt to import the Ollama Python library
    from ollama import ResponseError
    has_ollama_lib = True
except ImportError:
    has_ollama_lib = False
import requests

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
    def __init__(self, model_name: str):
        """Initialize TTS with a specific model."""
        pass

    def audio(self, text: str) -> bytes:
        """Generate audio from text and return as bytes."""
        pass

    def speak(self, text: str) -> None:
        """Generate and play audio from text automatically."""
        pass


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
    def __init__(self, model_name: str = "whisper"):
        """Initialize STT with a specific model (e.g., 'whisper')."""
        pass

    def transcribe(self, audio_file: bytes) -> str:
        """Transcribe an audio file to text."""
        pass

    def record(self) -> str:
        """Record live audio and return its transcription."""
        pass


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
    def __init__(self, model_name: str):
        """Initialize WakeWord with a specific model."""
        pass

    def listen(self) -> bool:
        """Listen for a wake word and return True when detected."""
        pass


class API:
    def __init__(self, api_key: str):
        """Initialize API with an API key for external service access."""
        pass

    def call(self, endpoint: str, payload: dict) -> dict:
        """Call an API endpoint with payload and return JSON response."""
        pass

# import ollama
# response = ollama.chat(model='llama3.1', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])



# Initialize LLM and start a conversation
llm = LLM("mistral")
response1 = llm.conversation("Why is the sky blue?")
print("Assistant:", response1, "\n\n")

response2 = llm.conversation("What about sunsets?")
print("Assistant:", response2, "\n\n")

response2 = llm.chat("What about sunsets?")
print("Assistant:", response2, "\n\n")
