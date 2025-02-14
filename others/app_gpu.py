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
def model_inference(input_dict, history):
    text = input_dict["text"]
    files = input_dict["files"]

    # Load images if provided
    if len(files) > 1:
        images = [load_image(image) for image in files]
    elif len(files) == 1:
        images = [load_image(files[0])]
    else:
        images = []

    # Validate input
    if text == "" and not images:
        gr.Error("Please input a query and optionally image(s).")
        return
    if text == "" and images:
        gr.Error("Please input a text query along with the image(s).")
        return

    # Prepare messages for the model
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": image} for image in images],
                {"type": "text", "text": "extract text from the image"},
            ],
        }
    ]

    # Apply chat template and process inputs
    prompt = Qwenprocessor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = Qwenprocessor(
        text=[prompt],
        images=images if images else None,
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

    prompt = f"{text}：{japanese_text.replace('<|im_end|>','')}"

    inputs = dstokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = dsmodel.generate(
        inputs.input_ids,
        max_length=1024,
        temperature=0.7,
        do_sample=True,
        pad_token_id=dstokenizer.eos_token_id
    )

    answer = dstokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer[len(prompt):]  # 返回生成的解答部分

# Example inputs
examples = [

    [{"text": "Extract JSON from the image", "files": ["example_images/document.jpg"]}],
    [{"text": "summarize the letter", "files": ["examples/1.png"]}],
    [{"text": "Describe the photo", "files": ["examples/3.png"]}],
    [{"text": "Extract as JSON table from the table", "files": ["examples/4.jpg"]}],
    [{"text": "Summarize the full image in detail", "files": ["examples/2.jpg"]}],
    [{"text": "Describe this image.", "files": ["example_images/campeones.jpg"]}],
    [{"text": "What is this UI about?", "files": ["example_images/s2w_example.png"]}],
    [{"text": "Can you describe this image?", "files": ["example_images/newyork.jpg"]}],
    [{"text": "Can you describe this image?", "files": ["example_images/dogs.jpg"]}],
    [{"text": "Where do the severe droughts happen according to this diagram?", "files": ["example_images/examples_weather_events.png"]}],

]

demo = gr.ChatInterface(
    fn=model_inference,
    description="# **Multimodal OCR**",
    examples=examples,
    textbox=gr.MultimodalTextbox(label="Query Input", file_types=["image"],value="以下是日语习题，请给出答案以及解答步骤：", file_count="multiple"),
    stop_btn="Stop Generation",
    multimodal=True,
    cache_examples=False,
)

demo.launch(debug=True)