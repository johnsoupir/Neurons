import neurons

llm = neurons.LLM("mistral") # Make a new LLM with the mistral model
stt = neurons.STT("base") # Make a new speech to text with Whisper base model
tts = neurons.TTS("tts_models/multilingual/multi-dataset/xtts_v2") #Make a new text to speech with XTTS-V2
tts.set_speaker("voices/glados.wav") # Set the voice XTTS should mimmic

while True:
    command = stt.speech()  # Wait for sound, then trascribe speech until it's quiet again
    print("USER >>> ", command, "\n") # Print out what the user said
    response = llm.conversation(command) # Give the command to the LLM and get response
    print("ROBOT >>> ", response, "\n") # Print a response
    tts.speak(response) # Speak the response