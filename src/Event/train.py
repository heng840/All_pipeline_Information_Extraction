import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from data_load import Dataset, pad, all_triggers, all_arguments
from model import Net
from params import get_hparams
from train_eval import train, evaluate

import logging
logging.basicConfig(level=logging.INFO, filename='output.txt', filemode='w')


if __name__ == "__main__":
    hp = get_hparams()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net(
        device=device,
        trigger_size=len(all_triggers),
        argument_size=len(all_arguments),
        hparams=hp
    )
    if device == 'cuda':
        model = model.cuda()

    model = nn.DataParallel(model)
    data_dir = None
    if hp.data_class == 'wiki_src':
        data_dir = 'Datasets/wiki_processed_data/source/'
    elif hp.data_class == 'wiki_info':
        data_dir = 'Datasets/wiki_processed_data/info/'
    elif hp.data_class == 'ace':
        data_dir = 'Datasets/ace2005/'
    train_set = data_dir + 'train.json'
    test_set = data_dir + 'test.json'
    dev_set = data_dir + 'dev.json'
    train_dataset = Dataset(train_set)
    dev_dataset = Dataset(dev_set)
    test_dataset = Dataset(test_set)

    samples_weight = train_dataset.get_samples_weight()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 sampler=sampler,
                                 num_workers=4,
                                 collate_fn=pad)
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    # optimizer = optim.Adadelta(model.parameters(), lr=1.0, weight_decay=1e-2)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)
    f1_max_dev = 0
    f1_max_test = 0
    for epoch in range(1, hp.n_epochs + 1):
        train(model, train_iter, optimizer, criterion)

        f_name = os.path.join(hp.logdir, str(epoch))
        print(f"=========eval dev at epoch={epoch}=========")
        logging.info(f"=========eval dev at epoch={epoch}=========")
        f1_dev = evaluate(model, dev_iter, f_name + '_dev', f1_max_dev)
        if f1_dev > f1_max_dev:
            f1_max_dev = f1_dev
            if not os.path.exists(hp.model_save_path):
                os.makedirs(hp.model_save_path)
            torch.save(model, f"{hp.model_save_path}/best_model_{epoch}.pt")
            print('best_model has been saved')
            print(f"weights were saved to {f_name}.pt")
            logging.info('best_model has been saved')
            logging.info(f"weights were saved to {f_name}.pt")
        print(f"=========eval test at epoch={epoch}=========")
        logging.info(f"=========eval test at epoch={epoch}=========")
        f1_test = evaluate(model, test_iter, f_name + '_test', f1_max_test)
        if f1_test > f1_max_test:
            f1_max_test = f1_test

