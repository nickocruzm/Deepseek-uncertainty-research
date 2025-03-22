from openai import OpenAI
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
from datetime import datetime


def write_to_file(fileName,data):
    # Write to file in a readable format
    with open(fileName, "w") as file:
        for item in data:
            file.write(json.dumps(item) + "\n") 
    

API_KEY = "sk-310f243bc1be47e4852d643d3465076b"
MODEL_NAME = 'deepseek-chat'
base_url = "https://api.deepseek.com"

current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
current_datetime = current_datetime.replace(" ", "@")


client = OpenAI(api_key=API_KEY, base_url=base_url)

system_prompt01 = """
    Respond to the question that is prefixed with Q: <question>. Some questions have multiple correct answers, only respond with one word answers. 
    Take into account the entire conversation history. Previous responses will be given. If possible provide a new answer, else if it a question that only
    has one correct answer then continue responding with that answer.
"""

usr_prompt = "Q: What is the capital of France?"
system_msg = {'role': "system", 'content':system_prompt01}
usr_msg = {'role': "user", 'content':usr_prompt}

messages = [system_msg, usr_msg]
false_response = "Madrid"

iterations = 5
for i in range(iterations):
    response = client.chat.completions.create(
        model=MODEL_NAME, 
        messages=messages,
        temperature=1.0,
        max_tokens=8000,
        stream=False
    )
    
    response_content = response.choices[0].message.content
    
    response_msg = {'role':"assistant", 'content':response_content}
    messages.append(response_msg)
    
    
    if(i == 0):
        usr_prompt += f"\n A: {response_content}"
        msg = {'role':"user", 'content':usr_prompt}
    
    if(i != 0):
        usr_prompt += f"\n Another response: {false_response}"
        msg = {'role':"user", 'content':usr_prompt}
    
    
    print('i: ', i)
    messages.append(msg)

file_name = f'../chats/convo-{current_datetime}.txt'
write_to_file(file_name, messages)