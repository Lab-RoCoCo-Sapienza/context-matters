from openai import OpenAI
import base64
import cv2
import os

import torch
import json
import requests


base_url = "http://127.0.0.1:11434"
chat_history = []

def local_llm_call(prompt, question):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "llama3.2:1b",
        "prompt": question,
        "stream": False,
        "system": prompt,
        "keep_alive": "10m"
    }

    response = requests.post(f"{base_url}/api/generate", headers=headers, data=json.dumps(data), stream=False)
    return response.json()['response']



class Agent:

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    # Call the Large Language Model
    def llm_call(self,prompt, question):
        completion = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {
                "role": "user",
                "content": question
            }
        ]
        )

        return (completion.choices[0].message.content)

    # Encode the image to base64 and call the Visual Language Model
    def vlm_call(self, prompt, image):
        self.image_to_buffer(image)
        agent = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                    "text":f"{prompt}"}, #TODO
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{self.encoded_image}",
                },
                },
            ],
            }
        ],
        temperature=0.1,
        )
        response = (agent.choices[0].message.content)
        return response
    
    # Encode the image to base64
    def image_to_buffer(self,image):
        if os.path.isfile(image):
            with open(image, "rb") as f:
                self.encoded_image = base64.b64encode(f.read()).decode("utf-8")
        else:
            _, buffer = cv2.imencode('.png', image)
            self.encoded_image = base64.b64encode(buffer).decode("utf-8")

