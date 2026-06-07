from transformers import pipeline

# Initialize the pipeline with the base gpt2 model
generator = pipeline("text-generation", model="openai-community/gpt2", device=0)

# Generate text
output = generator("Hello, I'm a language model")
print(output[0]["generated_text"])
