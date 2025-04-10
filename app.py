import json

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "thundax/Qwen2.5-0.5B-Sign"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map=device)

with open("text2sign.json", 'r', encoding='utf-8') as f:
    text2sign_dict = json.load(f)


def do_predict(text):
    input_text = f'Translate sentence into labels\n{text}\n'

    model_inputs = tokenizer([input_text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    signs = response_text.split(' ')
    actions = {x: text2sign_dict.get(x, '') for x in signs}

    return json.dumps({'text': response_text, 'actions': actions}, ensure_ascii=False, indent=4)


def run():
    with gr.Blocks(title="Qwen2.5-Sign") as app:
        gr.HTML("<h1><center>Qwen2.5-Sign</center></h1>")
        input_text = gr.TextArea(label="Input", lines=2, value="站一个制高点看上海，上海的弄堂是壮观的景象。它是这城市背景一样的东西。")
        submit_btn = gr.Button('Submit')
        output_text = gr.TextArea(label="Output", lines=20)

        submit_btn.click(do_predict, inputs=[input_text], outputs=[output_text])

        app.launch()


if __name__ == "__main__":
    run()