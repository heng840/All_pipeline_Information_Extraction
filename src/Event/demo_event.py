import nltk
import numpy as np
import torch
import gradio as gr
from consts import CLS, SEP, NONE, max_length
# from consts import test_event1, test_event2,
from data_load import tokenizer, trigger2idx, pad, idx2trigger
from params import get_hparams
from torch.utils import data

from utils import find_triggers


class Data_input(data.Dataset):
    def __init__(self, document):
        self.sent_li = []
        self.entities_li = []
        self.postags_li = []
        self.triggers_li = []
        self.arguments_li = []
        self.entities_li = []

        words = nltk.word_tokenize(document)
        split_num = len(words) // max_length
        for i in range(split_num + 1):
            if i < split_num:
                w = words[i * max_length: (i + 1) * max_length]
            else:
                w = words[i * max_length:]
            triggers = [NONE] * len(w)
            arguments = {
                'candidates': [],
                'events': {},
            }
            self.sent_li.append([CLS] + w + [SEP])
            self.triggers_li.append(triggers)
            self.arguments_li.append(arguments)

    def __len__(self):
        return len(self.sent_li)

    def __getitem__(self, idx):
        words = self.sent_li[idx]
        triggers = self.triggers_li[idx]
        arguments = self.arguments_li[idx]

        # We give credits only to the first piece.
        tokens_x = []
        is_heads = []
        for w in words:
            tokens = tokenizer.tokenize(w) if w not in [CLS, SEP] else [w]
            tokens_xx = tokenizer.convert_tokens_to_ids(tokens)

            if w in [CLS, SEP]:
                is_head = [0]
            else:
                is_head = [1] + [0] * (len(tokens) - 1)
            tokens_x.extend(tokens_xx)
            is_heads.extend(is_head)

        triggers_y = [trigger2idx[t] for t in triggers]
        head_indexes = []
        for i in range(len(is_heads)):
            if is_heads[i]:
                head_indexes.append(i)

        seqlen = len(tokens_x)

        return tokens_x, triggers_y, arguments, seqlen, head_indexes, words, triggers

    def get_samples_weight(self):
        samples_weight = []
        for triggers in self.triggers_li:
            not_none = False
            for trigger in triggers:
                if trigger != NONE:
                    not_none = True
                    break
            if not_none:
                samples_weight.append(5.0)
            else:
                samples_weight.append(1.0)
        return np.array(samples_weight)


def extract_events(user_input):
    hp = get_hparams()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(f'{hp.model_save_path}/best_model_11.pt')

    if device == 'cuda':
        model = model.cuda()
    if hasattr(model, 'module'):
        model = model.module
    model.eval()

    data_input = Data_input(user_input)
    data_iter = data.DataLoader(dataset=data_input,
                                batch_size=len(data_input.triggers_li),
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)
    words_all = []
    triggers_all = []
    triggers_hat_all = []
    arguments_all = []
    arguments_hat_all = []
    # if direct:
    #     return test_event2
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            tokens_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = batch

            trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys \
                = model.predict_triggers(tokens_x_2d=tokens_x_2d, head_indexes_2d=head_indexes_2d,
                                         triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d)

        words_all.extend(words_2d)
        triggers_all.extend(triggers_2d)
        triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
        arguments_all.extend(arguments_2d)

        if len(argument_keys) > 0:
            argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = \
                model.predict_arguments(argument_hidden, argument_keys, arguments_2d)
            arguments_hat_all.extend(argument_hat_2d)
        else:
            batch_size = len(arguments_2d)
            argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
            arguments_hat_all.extend(argument_hat_2d)
    triggers_pred = []
    arguments_pred = []
    events = []
    for i, (words, triggers, triggers_hat, arguments, arguments_hat) \
            in enumerate(zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
        triggers_hat = triggers_hat[:len(words)]
        triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

        # [(ith sentence, t_start, t_end, t_type_str)]
        triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

        for trigger in arguments_hat['events']:
            t_start, t_end, t_type_str = trigger
            for argument in arguments_hat['events'][trigger]:
                a_start, a_end, a_type_idx = argument
                arguments_pred.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))
        event = {
            'trigger': triggers_pred,
            'argument': arguments_pred,
        }
        events.append(event)

    return events


if __name__ == "__main__":
    document1 = 'As of early Tuesday there was no claim of responsibility. Prayuth Chan-ocha, the head of Thailand\u2019s military government, said that the authorities were searching for a person seen on closed-circuit footage but that it was not clear who the person was, news agencies reported.A spokesman for the police, Lt. Gen. Prawut Thavornsiri, told a Thai television interviewer that \u201cwe haven\u2019t concluded anything.\u201d The authorities said they were reviewing footage from 15 security cameras in the area but that the rush-hour crowds made deciphering the video difficult.\u201cThe shrine was very crowded,\u201d General Prawut said. \u201cIt\u2019s not clear even looking at the CCTV footage.\u201dThe bomb, General Prawut said, was placed under a bench on the outer rim of the shrine\u2019s grounds. Initially, the police said they had discovered at least two additional devices that they suspected were unexploded bombs inside the shrine and said other bombs may have been placed in the area, yelling at bystanders: \u201cGet out! Get out!\u201d'
    document2 = "A settlement has been reached in a $1-million lawsuit filed by a taxi driver accusing police of negligence after he got caught up in the August 2016 take-down of ISIS-sympathizer Aaron Driver.READ MORE: FBI agent whose tip thwarted 2016 ISIS attack in Ontario says he was glad to helpTerry Duffield was injured when Driver detonated a homemade explosive in the back of his cab in August 2016.\u201cI have to be very careful because there is an agreement to not disclose any of the terms of the settlement,\u201d Duffield\u2019s lawyer Kevin Egan told 980 CFPL.\u201cThe statement of claim, I guess, speaks for itself in regard to what we alleged.\u201dWATCH: Ontario taxi driver files $1M lawsuit against police2:18 Ontario taxi driver files $1M lawsuit against police Ontario taxi driver files $1M lawsuit against policeThat statement of claim, which Global News obtained a copy of in late March 2018, said police had more than enough time to intervene before Driver got into Duffield\u2019s taxi. The Attorney General of Canada, the Ontario government, Strathroy-Caradoc Police Service and London Police Service were named as defendants.Story continues below advertisementOn the morning of Aug. 10, 2016, U.S. authorities notified the RCMP they had detected a so-called martyrdom video in which a Canadian man said he was about to conduct an attack.The RCMP identified the man in the video as Driver and a tactical team surrounded his house in Strathroy.At 3:45 p.m., Driver called for a cab to take him to Citi Plaza in London. The claim alleged that despite the police presence, Duffield was not stopped from pulling into Driver\u2019s driveway. Driver then came out of the house and got into the back seat of the cab.\u201cWhen the SWAT team approached the vehicle, [Duffield] turned to Mr. Driver and said, \u2018I think they\u2019re here to talk to you\u2019 and he leaned over to get his cigarettes, it\u2019s a bench seat in the front of the taxicab, as he put his head down below the bench seat, the bomb went off.\u201dStory continues below advertisementThe inside of the cab where Aaron Driver detonated an explosive device on Aug. 10, 2016. Handout / RCMPEgan said Duffield had a preexisting back injury and the bomb blast triggered recurring pain. He also noted that his client was psychologically impacted by the event and is no longer able to work as a taxi driver.\u201cHe did try it. Got in a vehicle, turned the key on and started to shake and sweat and got out of the vehicle and vomited,\u201d said Egan. Tweet This\u201cHe was so traumatized by the event. He realized that any time any potential passenger was approaching the vehicle with a package he would be hyper-vigilant about that and just couldn\u2019t handle it emotionally.\u201dDetails of the settlement will not be made public, but Egan noted that no amount of money can properly compensate someone for physical or psychological injuries, but \u201cis the best we can do in the circumstance.\u201d He also noted that, while he was unwilling to disclose too much of Duffield\u2019s personal health, he has received some counselling and is \u201ccoping better now than he was then.\u201dStory continues below advertisement\u2013 with files from Stewart Bell and Andrew Russell."
    input_text = gr.inputs.Textbox(lines=10, label="Input Text")
    output_text = gr.outputs.Textbox(label="Output")

    gr.Interface(fn=extract_events, inputs=input_text, outputs=output_text, title="Event Extraction",
                 description="Enter some text and the model will extract events.").launch(share=True)
    # # todo for debug
    # events = extract_events(document2)
    # print("Triggers and Arguments:")
    # for event in events:
    #     print(f"Trigger: {event['trigger']}")
    #     print("Arguments:")
    #     if event['argument']:
    #         for argument in event['argument']:
    #             print(f"\t{argument}")
    #     else:
    #         print('argument is None')
    # while True:
    #     user_input = input("Enter a sentence: ")
    #     if user_input == "quit":
    #         break
    #     elif user_input == 'None':
    #         events = extract_events(document2)
    #     else:
    #         events = extract_events(user_input)
    #     print("Triggers and Arguments:")
    #     for event in events:
    #         print(f"Trigger: {event['trigger']}")
    #         print("Arguments:")
    #         for argument in event['arguments']:
    #             print(f"\t{argument}")
