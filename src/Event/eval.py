import os

import torch
from torch.utils import data

from data_load import Dataset, pad
from params import get_hparams
from train_eval import evaluate

if __name__ == "__main__":
    hp = get_hparams()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(f'{hp.model_save_path}/best_model_12.pt')

    if device == 'cuda':
        model = model.cuda()

    test_dataset = Dataset(hp.test_set)

    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=pad)
    dev_dataset = Dataset(hp.dev_set)

    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=4,
                               collate_fn=pad)
    if not os.path.exists(hp.logdir):
        os.makedirs(hp.logdir)

    print(f"=========eval test=========")
    evaluate(model, test_iter, 'eval_test')

    print(f"=========eval dev=========")
    evaluate(model, dev_iter, 'eval_dev')
