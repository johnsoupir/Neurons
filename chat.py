import neurons

llm = neurons.LLM("mistral") # New LLM with the Mistral model

while True:
    question = input("USER >>> ") # Get a prompt from the user
    response = llm.conversation(question) # Ask the LLM
    print("LLM >>> " + response) # Print out the response