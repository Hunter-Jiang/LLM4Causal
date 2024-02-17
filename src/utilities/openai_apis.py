import numpy as np
import pandas as pd
import openai
import os
import pickle
from retry import retry
from tqdm import tqdm
from io import StringIO

with open("openai_key.txt", "rb") as f:
    key_str = f.readline().decode('UTF-8')
print("using key", str(key_str))
client = openai.OpenAI(api_key = str(key_str))

#@retry(tries=100, delay=10)
def get_completion(prompt, model="gpt-4-1106-preview", temp = 0): #gpt-3.5-turbo-16k
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message.content#.choices[0].message["content"]

if __name__ == '__main__':
    response = get_completion("hello gpt, how are you?")
    print(response)