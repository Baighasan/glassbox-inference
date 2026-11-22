from transformers import AutoTokenizer, AutoModelForCausalLM
from glassbox.config import DEFAULT_MODEL, DEFAULT_DEVICE



# runtime should take input ids, and return output ids. we may also need to take the model name



tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL).to(DEFAULT_DEVICE)

input_ids = tokenizer("A list of colors: red, blue", return_tensors="pt").input_ids.to(DEFAULT_DEVICE)
generated_ids = model.generate(input_ids)
completion = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print(completion)
