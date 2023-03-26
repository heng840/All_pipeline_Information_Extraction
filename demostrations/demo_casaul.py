import torch
import transformers
import pandas
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline
)

# load models and tokenizers
tokenizer_ie = AutoTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-BERT-120M-IE-Chinese")
model_ie = AutoModel.from_pretrained("IDEA-CCNL/Erlangshen-BERT-120M-IE-Chinese")
nlp_ie = pipeline("ner", model=model_ie, tokenizer=tokenizer_ie)

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
    trigger_nlp = pipeline("ner", model=model_trigger, tokenizer=tokenizer_trigger)
    triggers = trigger_nlp(sentence)

    argument_nlp = pipeline('ner', model=model_ner, tokenizer=tokenizer_ner)
    arguments = argument_nlp(sentence)
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
    results = {
        # 'entities': named_entities,
        'triggers': triggers,
        'arguments': arguments,
        'causal': causal_relations,
        'factual': factual_info,
        # 'events': events
    }
    print(results)
    results_pd = pandas.DataFrame.from_dict(results)
    return results_pd


sentence = 'Bob, I think that the reason everybody in the south -- you know, first of all, we were -- when Franklin Roosevelt was elected president, we had been living what we thought was still a conquered nation after the Civil War'
x = predict(sentence)
print(x)