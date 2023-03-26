import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from consts import NONE
from data_load import idx2trigger, argument2idx
from utils import find_triggers


class Net(nn.Module):
    def __init__(self, trigger_size=None, argument_size=None, device=torch.device("cpu"), hparams=None):
        super().__init__()
        self.bert_model = BertModel.from_pretrained(hparams.model_name).to(device)
        self.bert_tokenizer = BertConfig.from_pretrained(hparams.model_name)
        hidden_size = self.bert_tokenizer.hidden_size
        self.fc1 = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
        ).to(device)
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size),
        ).to(device)
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size * 2, argument_size),
        ).to(device)
        self.device = device

    def predict_triggers(self, tokens_x_2d, head_indexes_2d, triggers_y_2d, arguments_2d):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        triggers_y_2d = torch.LongTensor(triggers_y_2d).to(self.device)
        head_indexes_2d = torch.LongTensor(head_indexes_2d).to(self.device)

        if self.training:
            self.bert_model.train()
            output = self.bert_model(tokens_x_2d)
            encoded_layers = output['last_hidden_state']

        else:
            self.bert_model.eval()
            with torch.no_grad():
                output = self.bert_model(tokens_x_2d)
                encoded_layers = output['last_hidden_state']

        batch_size = tokens_x_2d.shape[0]
        for i in range(batch_size):
            encoded_layers[i] = torch.index_select(encoded_layers[i], 0, head_indexes_2d[i])

        trigger_logits = self.fc_trigger(encoded_layers)
        trigger_hat_2d = trigger_logits.argmax(-1)

        argument_hidden, argument_keys = [], []
        for i in range(batch_size):
            candidates = arguments_2d[i]['candidates']
            golden_entity_tensors = {}

            for j in range(len(candidates)):
                e_start, e_end = candidates[j]
                golden_entity_tensors[candidates[j]] = encoded_layers[i, e_start:e_end, ].mean(dim=0)

            predicted_triggers = find_triggers([idx2trigger[trigger] for trigger in trigger_hat_2d[i].tolist()])
            for predicted_trigger in predicted_triggers:
                t_start, t_end, t_type_str = predicted_trigger
                event_tensor = encoded_layers[i, t_start:t_end, ].mean(dim=0)
                for j in range(len(candidates)):
                    e_start, e_end = candidates[j]
                    entity_tensor = golden_entity_tensors[candidates[j]]

                    argument_hidden.append(torch.cat([event_tensor, entity_tensor]))
                    argument_keys.append((i, t_start, t_end, t_type_str, e_start, e_end))

        return trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys

    def predict_arguments(self,
                          argument_hidden,
                          argument_keys,
                          arguments_2d):
        argument_hidden = torch.stack(argument_hidden)
        argument_logits = self.fc_argument(argument_hidden)
        argument_hat_1d = argument_logits.argmax(-1)

        arguments_y_1d = []
        for i, t_start, t_end, t_type_str, \
            e_start, e_end in argument_keys:
            arg_label = argument2idx[NONE]
            if (t_start, t_end, t_type_str) in arguments_2d[i]['events']:
                for (a_start, a_end, a_type_idx) in arguments_2d[i]['events'][(t_start, t_end, t_type_str)]:
                    if e_start == a_start and e_end == a_end:
                        # raise Exception
                        arg_label = a_type_idx
                        break
            arguments_y_1d.append(arg_label)

        arguments_y_1d = torch.LongTensor(arguments_y_1d).to(self.device)

        batch_size = len(arguments_2d)
        argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
        for (i, st, ed, event_type_str, e_st, e_ed), a_label in zip(argument_keys, argument_hat_1d.cpu().numpy()):
            if a_label == argument2idx[NONE]:
                continue
            if (st, ed, event_type_str) not in argument_hat_2d[i]['events']:
                argument_hat_2d[i]['events'][(st, ed, event_type_str)] = []
            argument_hat_2d[i]['events'][(st, ed, event_type_str)].append((e_st, e_ed, a_label))

        return argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d

