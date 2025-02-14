import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from transformers.image_utils import load_image
from threading import Thread
import time
from openai import OpenAI
import base64

def model_inference(input_dict, history):
    text = input_dict["text"]
    files = input_dict["files"]

    images = []
    for f in files:
        with open(f, "rb") as image_file:
            image = base64.b64encode(image_file.read()).decode('utf-8')
            images.append(image)

    # Validate input
    if text == "" and not images:
        gr.Error("Please input a query and optionally image(s).")
        return
    if text == "" and images:
        gr.Error("Please input a text query along with the image(s).")
        return


    # Apply chat template and process inputs
    ocrclient = OpenAI(
        api_key="********************",  # MODELSCOPE_SDK_TOKEN, 请替换成您的ModelScope SDK Token
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
                   'url':f"data:image/jpeg;base64,{image}",
                   },
        }         for image in images],
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
        api_key='37e69a2d-cfed-4c54-88de-7074c5c2c39c',  # ModelScope Token
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
examples = [
    [{"text": "请解答一下日语题，给出答案和解题思路：", "files": ["images/demo1_query.png"]}],
    [{"text": "请解答一下日语题，给出答案和解题思路：", "files": ["images/demo2_query.png"]}]
]

demo = gr.ChatInterface(
    fn=model_inference,
    description="# **Deepseek-R1 & Qwen2.5-VL-7B-Instruct Janpanese Quiz assistant**",
    examples=examples,
    textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image"],value="请解答一下日语题，给出答案和解题思路：", file_count="multiple"),
    stop_btn="Stop Generation",
    multimodal=True,
    cache_examples=False,
)

demo.launch(debug=True)