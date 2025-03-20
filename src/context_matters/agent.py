from openai import OpenAI
import base64, cv2, os, json, requests

base_url = "http://127.0.0.1:11434"
chat_history = []
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

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

    
def llm_call(prompt, question, temperature=None, top_p=None, model="gpt-4o"):
    completion_args = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": question}
        ]
    }
    if temperature is not None:
        completion_args["temperature"] = temperature
    if top_p is not None:
        completion_args["top_p"] = top_p

    completion = client.chat.completions.create(**completion_args)

    return (completion.choices[0].message.content)

def vlm_call(prompt, image):
    encoded_image = image_to_buffer(image)
    agent = client.chat.completions.create(
    model="gpt-4o",
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
                "url": f"data:image/jpeg;base64,{encoded_image}",
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
def image_to_buffer(image):
    if os.path.isfile(image):
        with open(image, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode("utf-8")
    else:
        _, buffer = cv2.imencode('.png', image)
        encoded_image = base64.b64encode(buffer).decode("utf-8")
        return encoded_image

