import gradio as gr
#from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
#from transformers.image_utils import load_image
#from threading import Thread
#import time
from openai import OpenAI
import base64
import os

def model_inference(image_file):

    images = []

    with open(image_file, "rb") as f:
        image = base64.b64encode(f.read()).decode('utf-8')
        images.append(image)

    # Validate input
    if  not images:
        gr.Error("Please input a query and optionally image(s).")
        return

    # Apply chat template and process inputs
    ocrclient = OpenAI(
        api_key=os.getenv("MODELSCOPE_API_KEY"),  # MODELSCOPE_SDK_TOKEN, 请替换成您的ModelScope SDK Token
        base_url="https://api-inference.modelscope.cn/v1"
    )

    response = ocrclient.chat.completions.create(
        model="Qwen/Qwen2.5-VL-7B-Instruct",  # ModleScope Model-Id
        messages=[
        {
            "role": "user",
            "content": [
                *[{'type': 'image_url',
                    'image_url': {
                           'url': f"data:image/jpeg;base64,{image}",
                    },
                   } for image in images],
                {"type": "text", "text": "提取图片中的文本，只输出文本内容"},
            ],
        }
        ],
        stream=True
    )
    picresult = ""

    for chunk in response:
        picresult += chunk.choices[0].delta.content

    client = OpenAI(
        base_url='https://api-inference.modelscope.cn/v1/',
        api_key=os.getenv("MODELSCOPE_API_KEY"),  # ModelScope Token
    )

    response = client.chat.completions.create(
        model='deepseek-ai/DeepSeek-R1',  # ModelScope Model-Id
        messages=[
            {
                'role': 'user',
                'content': f'请解答一下日语题，给出答案和解题思路：{picresult}'
            }
        ],
        stream=True
    )
    done_reasoning = False
    buffer = ""
    yield "Thinking..."
    for chunk in response:
        reasoning_chunk = chunk.choices[0].delta.reasoning_content
        answer_chunk = chunk.choices[0].delta.content
        if reasoning_chunk != '':
            buffer +=reasoning_chunk
        elif answer_chunk != '':
            if not done_reasoning:
                buffer += '\n\n === Final Answer ===\n'
                done_reasoning = True
            buffer+= answer_chunk
        yield  buffer
    #return picresult  # 返回生成的解答部分

# Example inputs
examples = ["images/demo1_query.png", "images/demo2_query.png"]


demo = gr.Interface(model_inference,
                  inputs=[gr.Image(label="上传图片",type='filepath')],
                  examples=examples,
                  outputs=[gr.Textbox(label="AI解题", lines=2, show_copy_button=True),
                           # gr.Image(label="Output Image")
                           ],
                  title="Deepseek-R1 & Qwen2.5-VL-7B-Instruct Janpanese Quiz assistant"
                  )

demo.launch(debug=True)