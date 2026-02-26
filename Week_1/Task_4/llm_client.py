# from langchain_community.llms.ollama import Ollama

from langchain_ollama import OllamaLLM

class LLMClient:
    def __init__(self):
        # self.llm = Ollama(model="llama3")

        self.llm = OllamaLLM(model="llama3")
        self.cache = {}

    # def generate(self, prompt):
    #     return self.llm.invoke(prompt)

    def generate(self, prompt):
        if prompt in self.cache:
            print("Returned from cache")
            return self.cache[prompt] , True  # True indicates means comes from cache
        
        ## Response is not in cache, call the LLM
        response = self.llm.invoke(prompt)

        ## Store in cache
        self.cache[prompt] = response
        # return response, False  # False indicates means comes from LLM
        return str(response), False  # False indicates means comes from LLM