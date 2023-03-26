import numpy as np
from torch.utils import data
import json

from transformers import BertTokenizer

# init vocab
from consts import TRIGGERS, ARGUMENTS, NONE, SEP, CLS, PAD, UNK
from params import get_hparams
from utils import build_vocab

all_triggers, trigger2idx, idx2trigger = build_vocab(TRIGGERS)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS, BIO_tagging=False)
hparams = get_hparams()
tokenizer = BertTokenizer.from_pretrained(hparams.model_name, do_lower_case=False, never_split=(PAD, CLS, SEP, UNK))


class Dataset(data.Dataset):
    def __init__(self, fpath):
        self.sent_li = []
        self.entities_li = []
        self.postags_li = []
        self.triggers_li = []
        self.arguments_li = []
        self.entities_li = []

        with open(fpath, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = json.loads(line)
                words = item['words']
                triggers = [NONE] * len(words)
                arguments = {
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }
                for entity_mention in item['golden_entity_mentions']:
                    arguments['candidates'].append(
                        (entity_mention['start'], entity_mention['end']))
                for event_mention in item['golden_event_mentions']:
                    for i in range(event_mention['trigger']['start'], event_mention['trigger']['end']):
                        trigger_type = event_mention['event_type']
                        if i == event_mention['trigger']['start']:
                            triggers[i] = 'B-{}'.format(trigger_type)
                        else:
                            triggers[i] = 'I-{}'.format(trigger_type)

                    event_key = (event_mention['trigger']['start'], event_mention['trigger']['end'], event_mention['event_type'])
                    arguments['events'][event_key] = []
                    for argument in event_mention['arguments']:
                        role = argument['role']
                        arguments['events'][event_key].append((argument['start'], argument['end'], argument2idx[role]))

                self.sent_li.append([CLS] + words + [SEP])
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


def pad(batch):
    tokens_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = list(map(list, zip(*batch)))
    maxlen = np.array(seqlens_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        head_indexes_2d[i] = head_indexes_2d[i] + [0] * (maxlen - len(head_indexes_2d[i]))
        triggers_y_2d[i] = triggers_y_2d[i] + [trigger2idx[PAD]] * (maxlen - len(triggers_y_2d[i]))

    return tokens_x_2d, \
           triggers_y_2d, arguments_2d, \
           seqlens_1d, head_indexes_2d, \
           words_2d, triggers_2d
