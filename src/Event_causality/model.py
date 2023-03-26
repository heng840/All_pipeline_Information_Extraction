import torch
import torch.nn as nn
from transformers import BertModel


class BertCausalModel(nn.Module):
    def __init__(self, y_num):
        super(BertCausalModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(768*3, y_num)

    def forward(self, sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask):
        """
        :param sentences_s: source
        :param mask_s:
        :param sentences_t: target
        :param mask_t:
        :param event1:
        :param event1_mask:
        :param event2:
        :param event2_mask:
        :return:
        """
        enc_s = self.bert(sentences_s, attention_mask = mask_s)
        enc_t = self.bert(sentences_t, attention_mask=mask_t)

        hidden_enc_s = enc_s[0]
        hidden_enc_t = enc_t[0]

        event1 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(hidden_enc_s, event1)], dim=0)
        event2 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(hidden_enc_t, event2)], dim=0)

        m1 = event1_mask.unsqueeze(-1).expand_as(event1).float()
        m2 = event2_mask.unsqueeze(-1).expand_as(event2).float()

        event1 = event1 * m1
        event2 = event2 * m2

        opt1 = torch.sum(event1, dim=1)
        opt2 = torch.sum(event2, dim=1)

        opt = torch.cat((enc_s[1], opt1, opt2), 1)
        opt = self.fc(opt)
        return opt

