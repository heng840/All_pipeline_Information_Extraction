import torch
import transformers
import gradio as gr

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
# model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

tokenizer = AutoTokenizer.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")
model = AutoModelForTokenClassification.from_pretrained("Davlan/bert-base-multilingual-cased-ner-hrl")

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"
#
# ner_results = nlp(example)
# print(ner_results)

# 创建predict函数来执行信息抽取
def predict(sentence):
    ner_results = nlp(sentence)
    # print(ner_results)
    return ner_results

# predict(example)
# 创建输入组件和输出组件
input_text = gr.inputs.Textbox(label="输入句子")
output_df = gr.outputs.Dataframe(type='array', label="输出结果")

# 将输入组件和输出组件传递给Gradio接口
iface = gr.Interface(fn=predict, inputs=input_text, outputs=output_df)

# 启动Gradio界面
iface.launch(share=True)
