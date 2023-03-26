import os

# Disable Tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline
)

import spacy

nlp_trigger = spacy.load('en_core_web_sm')
# Define a set of auxiliary verbs
aux_verbs = {"be", "am", "is", "are", "was", "were", "been", "being", "have", "has", "had", "do", "does", "did", "can",
             "could", "will", "would", "shall", "should", "may", "might", "must"}

# load models and tokenizers
# tokenizer_ie = AutoTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-BERT-120M-IE-Chinese")
# model_ie = AutoModel.from_pretrained("IDEA-CCNL/Erlangshen-BERT-120M-IE-Chinese")
# nlp_ie = pipeline("ner", model=model_ie, tokenizer=tokenizer_ie)

tokenizer_ner = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
model_ner = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")

tokenizer_trigger = AutoTokenizer.from_pretrained("lfcc/bert-portuguese-event-trigger")
model_trigger = AutoModelForTokenClassification.from_pretrained("lfcc/bert-portuguese-event-trigger")

tokenizer_causal = AutoTokenizer.from_pretrained("noahjadallah/cause-effect-detection")
model_causal = AutoModelForTokenClassification.from_pretrained("noahjadallah/cause-effect-detection")

tokenizer_factual = AutoTokenizer.from_pretrained("amandakonet/climatebert-fact-checking")
model_factual = AutoModelForSequenceClassification.from_pretrained("amandakonet/climatebert-fact-checking")

tokenizer_event = AutoTokenizer.from_pretrained("facebook/bart-large")
model_event = AutoModel.from_pretrained("facebook/bart-large")


# define predict function to perform information extraction
def predict(sentence):
    # extract triggers
    doc = nlp_trigger(sentence)
    triggers = {}
    for token in doc:
        if token.dep_ == "ROOT" or (token.pos_ == "VERB" and token.lemma_ not in aux_verbs):
            triggers = token.text
            break
    # extract triggers
    # trigger_nlp = pipeline("ner", model=model_trigger, tokenizer=tokenizer_trigger)
    # triggers = trigger_nlp(sentence)

    argument_nlp = pipeline('ner', model=model_ner, tokenizer=tokenizer_ner)
    arguments = argument_nlp(sentence)
    arguments_new = []
    for arg in arguments:
        arguments_new.append(
            {'entity': arg['entity'], 'word': arg['word']}
        )
    # extract causal relations
    causal_nlp = pipeline("ner", model=model_causal, tokenizer=tokenizer_causal)
    causal_relations = causal_nlp(sentence)

    # extract factual information
    factual_nlp = pipeline("text-classification", model=model_factual, tokenizer=tokenizer_factual)
    factual_info = factual_nlp(sentence)

    # # extract events
    # event_nlp = pipeline("text2text-generation", model=model_event, tokenizer=tokenizer_event)
    # event_input = ""
    # for trigger in triggers:
    #     event_input += trigger['word'] + ": "
    #     event_input += sentence[trigger['start']:trigger['end']] + " "
    #     for entity in named_entities:
    #         if entity['start'] >= trigger['start'] and entity['end'] <= trigger['end']:
    #             event_input += entity['word'] + ": " + entity['entity_group'] + " "
    #     event_input += "\n"
    # events = event_nlp(event_input, max_length=1024, do_sample=False)
    # events = [e['generated_text'].strip() for e in events]

    # combine results
    causal_relations_new = []
    for cau in causal_relations:
        if cau['entity'] != 'OTHER':
            causal_relations_new.append({'entity': cau['entity'], 'word': cau['word']})
    fact_label_mapping = ['entailment', 'contradiction', 'neutral']
    factual_new = []
    for fact in factual_info:
        if fact['label'] == 'LABEL_0':
            factual_new.append(fact_label_mapping[0])
        elif fact['label'] == 'LABEL_1':
            factual_new.append(fact_label_mapping[1])
        else:
            factual_new.append(fact_label_mapping[2])

    results = {
        # 'entities': named_entities,
        'triggers': triggers,
        'arguments': arguments_new,
        'causal': causal_relations_new,
        'factual': factual_new,
        # 'events': events
    }
    results = str('triggers:' + str(triggers) + '\n\n' + 'arguments:' + str(arguments_new) + '\n\n' + 'causal:' + str(
        causal_relations_new) + '\n\n' + 'factual:' + str(factual_new))
    # print(results)
    # results_pd = pandas.json_normalize(results)
    # return results_pd
    return results


sentence = 'Bob, I think that the reason everybody in the south -- you know, first of all, we were -- when Franklin ' \
           'Roosevelt was elected president, ' \
           'we had been living what we thought was still a conquered nation after the Civil War'
x = predict(sentence)
print(x)
with gr.Blocks() as demo:
    name = gr.Textbox(label="Enter a sentence or a document")
    output = gr.Textbox(label="Output Box")
    greet_btn = gr.Button("Extract")
    greet_btn.click(fn=predict, inputs=name, outputs=output)

demo.launch(share=True)

# fixme done 总是出现numpy.float的原因在score上。
