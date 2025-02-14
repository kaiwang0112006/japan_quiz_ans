import gradio as gr
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from transformers.image_utils import load_image
from threading import Thread
import time
import torch
import spaces

# Fine-tuned for OCR-based tasks from Qwen's [ Qwen/Qwen2-VL-2B-Instruct ]
QWEN_MODEL_ID = r"/mnt/d/bigmodel/Qwen2-VL-OCR-2B-Instruct"
DS_MODEL_ID = r"/mnt/d/bigmodel/DeepSeek-R1-Distill-Qwen-7B"
Qwenprocessor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
Qwenmodel = Qwen2VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16
).to("cuda").eval()

dstokenizer = AutoTokenizer.from_pretrained(DS_MODEL_ID)
dsmodel = AutoModelForCausalLM.from_pretrained(DS_MODEL_ID).to("cuda").eval()

@spaces.GPU
def model_inference(image_file):
    image = [load_image(image_file)]


    # Validate input
    if not image:
        gr.Error("Please input a query and optionally image(s).")
        return

    # Prepare messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "提取图片中的文本，只输出文本内容"},
            ],
        }
    ]

    # Apply chat template and process inputs
    prompt = Qwenprocessor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = Qwenprocessor(
        text=[prompt],
        images=image if image else None,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    inputs = inputs.to("cuda")

    # OCR
    generated_ids = Qwenmodel.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = Qwenprocessor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    japanese_text = "\n".join(output_text)
    japanese_text = japanese_text.replace('<|im_end|>', '')

    prompt = f'请解答一下日语题，给出答案和解题思路：{japanese_text}'
    streamer = TextIteratorStreamer(dstokenizer)
    messages = [
        {"role": "user", "content": prompt},
    ]
    input_ids = dstokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=500,
        temperature=0.0,
        do_sample=False,
        streamer=streamer,
    )

    thread = Thread(target=dsmodel.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    yield "Thinking..."

    for new_text in streamer:
        buffer += new_text
        yield buffer


# Example inputs
examples = ["images/demo1_query.png", "images/demo2_query.png"]


demo = gr.Interface(model_inference,
                  inputs=[gr.Image(label="上传图片",type='filepath')],
                  examples=examples,
                  outputs=[gr.Textbox(label="AI解题", lines=2, show_copy_button=True),
                           # gr.Image(label="Output Image")
                           ],
                  title="Deepseek-R1 & Qwen2.5-VL-7B-Instruct Janpanese Quiz ssistent"
                  )

demo.launch(debug=True)