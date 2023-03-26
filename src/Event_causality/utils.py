import argparse


def get_hparams():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_files', type=str, default='processed_files')
    parser.add_argument('--use_scheduler', default=True, action='store_false')
    parser.add_argument('--batch_size_train', type=int, default=25)
    parser.add_argument('--batch_size_test', type=int, default=20)
    parser.add_argument('--epoch_nums', type=int, default=100)
    parser.add_argument('--saved_models', type=str, default='saved_models/wo_scheduler')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--output_dir', type=str, default='output_with_scheduler.txt')
    hparams = parser.parse_args()
    return hparams


def split_train_test(dataset):
    train_set = []
    test_set = []

    test_topic = ['1', '3', '4', '5']
    for data in dataset:
        t = data[0]
        if t.split('/')[-2] in test_topic:
            test_set.append(data)
        else:
            train_set.append(data)
    return train_set, test_set


def compute_f1(gold, predicted):
    c_predict = 0
    c_correct = 0
    c_gold = 0

    for g, p in zip(gold, predicted):
        if g != 0:
            c_gold += 1
        if p != 0:
            c_predict += 1
        if g != 0 and p != 0:
            c_correct += 1

    p = c_correct / (c_predict + 1e-100)
    r = c_correct / c_gold
    f = 2 * p * r / (p + r + 1e-100)

    # print('correct', c_correct)
    # print('predicted', c_predict)
    # print('golden', c_gold)

    return p, r, f
