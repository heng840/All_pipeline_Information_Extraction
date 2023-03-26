import pickle
from os.path import exists

import torch
from transformers import BertTokenizer

from dataset import Dataset
from preprocess import make_data_pickle
from utils import split_train_test, compute_f1, get_hparams


def evaluate(processed_files, batch_size_test=20, saved_models=None):
    data_pickle = f'{processed_files}/data.pickle'
    if not exists(data_pickle):
        raw_pickle = f'{processed_files}/document_raw.pickle'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        make_data_pickle(raw_pickle=raw_pickle, data_pickle=data_pickle, tokenizer=tokenizer)
    with open(data_pickle, 'rb') as f:
        data = pickle.load(f)

    _, test_set = split_train_test(data)
    test_dataset = Dataset(batch_size=batch_size_test, dataset=test_set)
    device = torch.device("cuda")
    model = torch.load(f'{saved_models}/best_model_12.pt')
    model.eval()
    with torch.no_grad():
        predicted_all = []
        gold_all = []
        for batch in test_dataset.reader(device, False):
            sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y = batch
            opt = model(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
            predicted = torch.argmax(opt, -1)
            predicted = list(predicted.cpu().numpy())
            predicted_all += predicted

            gold = list(data_y.cpu().numpy())
            gold_all += gold
        p, r, f = compute_f1(gold_all, predicted_all)
        print(p, r, f)


if __name__ == '__main__':
    hparams = get_hparams()
    evaluate(processed_files=hparams.processed_files, batch_size_test=hparams.batch_size_test,
             saved_models=hparams.saved_models)
