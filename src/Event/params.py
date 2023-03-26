import argparse
# from transformers import set_seed


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--n_epochs", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="logdir")
    # fixme_done path problem
    parser.add_argument("--data_class", type=str, choices=['ace', 'wiki_src', 'wiki_info'], default='wiki_src')
    parser.add_argument("--model_name", type=str, default="bert-large-cased")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_save_path", type=str, default='model_saved')
    hparams = parser.parse_args()

    # set_seed(hparams.seed)
    return hparams
