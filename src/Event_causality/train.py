import os
import pickle
import random
from os.path import exists

import torch
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from dataset import Dataset
from model import BertCausalModel
from preprocess import make_data_pickle
from utils import split_train_test, compute_f1, get_hparams
import logging


def train(processed_files, use_scheduler=True, batch_size_train=25, batch_size_test=20, epoch_nums=100,
          saved_models=None, learning_rate=2e-5, output_dir=None):
    logging.basicConfig(level=logging.INFO, filename=output_dir, filemode='w')
    data_pickle = f'{processed_files}/data.pickle'
    debug = False
    if not exists(data_pickle) or debug:
        raw_pickle = f'{processed_files}/document_raw.pickle'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        make_data_pickle(raw_pickle=raw_pickle, data_pickle=data_pickle, tokenizer=tokenizer, debug=True)
    with open(data_pickle, 'rb') as f:
        data = pickle.load(f)

    train_set, test_set = split_train_test(data)
    filter = []
    for d in train_set:
        if d[-1] == 'NULL':
            if random.random() < 0.7:
                continue
        filter.append(d)
    train_set = filter

    train_dataset = Dataset(batch_size=batch_size_train, dataset=train_set)
    test_dataset = Dataset(batch_size=batch_size_test, dataset=test_set)

    device = torch.device("cuda")

    model = BertCausalModel(y_num=2).to(device)  # binary
    if not exists(saved_models):
        os.makedirs(saved_models)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = None
    if use_scheduler:
        logging.info('with_scheduler')
        t_total = int(len(train_set) / 25 / 1 * 60)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=t_total)
    else:
        logging.info('wo_scheduler')
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    max_f1 = 0
    saved_model_path = None
    for epoch in range(epoch_nums):
        model.train()
        for batch in train_dataset.get_tqdm(device, True):
            sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, data_y = batch
            opt = model(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask)
            loss = loss_fn(opt, data_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()

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
            if f > max_f1:
                max_f1 = f
                if saved_model_path:
                    os.remove(saved_model_path)
                saved_model_path = f'{saved_models}/best_model_{epoch}.pt'
                torch.save(model, saved_model_path)
                logging.info(f'epoch={epoch}: p={p}, r={r}, f1={f}')


if __name__ == '__main__':
    hparams = get_hparams()
    train(processed_files=hparams.processed_files, use_scheduler=hparams.use_scheduler,
          batch_size_train=hparams.batch_size_train, batch_size_test=hparams.batch_size_test,
          epoch_nums=hparams.epoch_nums, saved_models=hparams.saved_models, learning_rate=hparams.learning_rate,
          output_dir=hparams.output_dir)


