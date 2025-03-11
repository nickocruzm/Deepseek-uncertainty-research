from openai import OpenAI


client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant"
        },
        
        {
            "role": "user", 
            "content": "what is 2 + 2?"
         },
    ],
    
    stream=False
)


for msg in response.choices:
    print(msg.message.content)