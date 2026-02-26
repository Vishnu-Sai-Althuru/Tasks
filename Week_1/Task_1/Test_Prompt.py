import os
from langchain_ollama import OllamaLLM

# Connect to local LLM
llm = OllamaLLM(model="llama3")

# Get absolute path of current file
base_dir = os.path.dirname(os.path.abspath(__file__))
prompt_path = os.path.join(base_dir, "Prompt", "Prompt_Template.txt")

# Load prompt template safely
with open(prompt_path, "r", encoding="utf-8") as f:
 template = f.read()

# Example data
context = "AI stands for Artificial Intelligence."
question = "How does AI work?"

# Inject values into prompt
final_prompt = template.format(context=context, question=question)

print("\n--- PROMPT SENT TO LLM ---\n")
print(final_prompt)

print("\n--- LLM RESPONSE ---\n")
response = llm.invoke(final_prompt)
print(response)