import nltk
import numpy as np
import torch
import gradio as gr
# from consts import test_event1, test_event2,
from torch.utils import data

NONE = 'O'
PAD = "[PAD]"
UNK = "[UNK]"

# for BERT
CLS = '[CLS]'
SEP = '[SEP]'
max_length = 400

all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)
hparams = get_hparams()
tokenizer = BertTokenizer.from_pretrained(hparams.model_name, do_lower_case=False, never_split=(PAD, CLS, SEP, UNK))
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
    input_text = gr.inputs.Textbox(lines=10, label="Input Text")
    output_text = gr.outputs.Textbox(label="Output")

    gr.Interface(fn=extract_events, inputs=input_text, outputs=output_text, title="Event Extraction",
                 description="Enter some text and the model will extract events.").launch(share=True)
