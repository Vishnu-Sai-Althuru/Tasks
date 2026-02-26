# from functools import lru_cache
# @lru_cache(maxsize=100)
# def cached_llm_call(prompt):
#     from llm_client import LLMClient
#     client = LLMClient()

#     if prompt in client.cache:
#         print("Returned from cache")
#         return 
#     print("LLM Executed")
#     return client.generate(prompt)


cache ={}
def cached_llm_call(prompt):
    from llm_client import LLMClient
    client = LLMClient()

    if prompt in cache:
        print("Returned from cache")
        return cache[prompt], True  # True indicates means comes from cache
    
    ## Response is not in cache, call the LLM
    print("LLM Executed")
    response = client.generate(prompt)


    ## otherwise call the LLM
    print("LLM Executed")
    response = client.generate(prompt)
    ## Store in cache
    cache[prompt] = response

    # return response, False  # False indicates means comes from LLM
    return str(response), False  # False indicates means comes from LLM
