import json
import logging
import os
import random
import sys
import time

import torch
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from entity.models import EntityModel
from entity.utils import convert_dataset_to_samples, batchify, NpEncoder
from shared.const import task_ner_labels, get_labelmap
from shared.data_structures import Dataset
from shared.get_hparams import get_hparams_entity

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger('root')


def save_model(model, args):
    """
    Save the model to the output directory
    """
    logger.info('Saving model to %s...' % (args.output_dir))
    model_to_save = model.bert_model.module if hasattr(model.bert_model, 'module') else model.bert_model
    model_to_save.save_pretrained(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)


def output_ner_predictions(model, batches, dataset, output_file):
    """
    Save the prediction as a json file
    """
    ner_result = {}
    span_hidden_table = {}
    tot_pred_ett = 0
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            off = sample['sent_start_in_doc'] - sample['sent_start']
            k = sample['doc_key'] + '-' + str(sample['sentence_ix'])
            ner_result[k] = []
            for span, pred in zip(sample['spans'], preds):
                span_id = '%s::%d::(%d,%d)' % (sample['doc_key'], sample['sentence_ix'], span[0] + off, span[1] + off)
                if pred == 0:
                    continue
                ner_result[k].append([span[0] + off, span[1] + off, ner_id2label[pred]])
            tot_pred_ett += len(ner_result[k])

    logger.info('Total pred entities: %d' % tot_pred_ett)

    js = dataset.js
    for i, doc in enumerate(js):
        doc["predicted_ner"] = []
        doc["predicted_relations"] = []
        for j in range(len(doc["sentences"])):
            k = doc['doc_key'] + '-' + str(j)
            if k in ner_result:
                doc["predicted_ner"].append(ner_result[k])
            else:
                logger.info('%s not in NER results!' % k)
                doc["predicted_ner"].append([])

            doc["predicted_relations"].append([])

        js[i] = doc

    logger.info('Output predictions to %s..' % (output_file))
    with open(output_file, 'w') as f:
        f.write('\n'.join(json.dumps(doc, cls=NpEncoder) for doc in js))


def evaluate(model, batches, tot_gold):
    """
    Evaluate the entity model
    """
    logger.info('Evaluating...')
    c_time = time.time()
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0

    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, \
            preds in \
                zip(batches[i], pred_ner):
            for gold, \
                pred in zip(sample['spans_label'], preds):
                l_tot += 1
                if pred == gold:
                    l_cor += 1
                if pred != 0 and gold != 0 and pred == gold:
                    cor += 1
                if pred != 0:
                    tot_pred += 1

    acc = l_cor / l_tot
    logger.info('Accuracy: %5f' % acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d' % (cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f' % (p, r, f1))
    logger.info('Used time: %f' % (time.time() - c_time))
    return f1


def setseed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    hparams = get_hparams_entity()
    hparams.train_data_path = os.path.join(hparams.data_dir, 'train.json')
    hparams.dev_data_path = os.path.join(hparams.data_dir, 'dev.json')
    hparams.test_data_path = os.path.join(hparams.data_dir, 'test.json')

    if 'albert' in hparams.model:
        logger.info('Use Albert: %s' % hparams.model)
        hparams.use_albert = True

    setseed(hparams.seed)

    if not os.path.exists(hparams.output_dir):
        os.makedirs(hparams.output_dir)

    if hparams.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(hparams.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(hparams.output_dir, "eval.log"), 'w'))

    logger.info(sys.argv)
    logger.info(hparams)

    ner_label2id, ner_id2label = get_labelmap(task_ner_labels[hparams.task])

    num_ner_labels = len(task_ner_labels[hparams.task]) + 1
    model = EntityModel(hparams, num_ner_labels=num_ner_labels)

    dev_data = Dataset(hparams.dev_data_path)
    dev_samples, dev_ner = convert_dataset_to_samples(dev_data, hparams.max_span_length, ner_label2id=ner_label2id,
                                                      context_window=hparams.context_window)
    dev_batches = batchify(dev_samples, hparams.eval_batch_size)

    if hparams.do_train:
        train_data = Dataset(hparams.train_data_path)
        train_samples, train_ner = convert_dataset_to_samples(train_data, hparams.max_span_length,
                                                              ner_label2id=ner_label2id,
                                                              context_window=hparams.context_window)
        train_batches = batchify(train_samples, hparams.train_batch_size)
        best_result = 0.0

        param_optimizer = list(model.bert_model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if 'bert' in n]},
            {'params': [p for n, p in param_optimizer
                        if 'bert' not in n], 'lr': hparams.task_learning_rate}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=hparams.learning_rate, correct_bias=not hparams.bertadam)
        t_total = len(train_batches) * hparams.num_epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, int(t_total * hparams.warmup_proportion), t_total)

        tr_loss = 0
        tr_examples = 0
        global_step = 0
        eval_step = len(train_batches) // hparams.eval_per_epoch
        for epoch in tqdm(range(hparams.num_epoch)):
            if hparams.train_shuffle:
                random.shuffle(train_batches)
            for i in tqdm(range(len(train_batches))):
                output_dict = model.run_batch(train_batches[i], training=True)
                loss = output_dict['ner_loss']
                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % hparams.print_loss_step == 0:
                    logger.info('Epoch=%d, iter=%d, loss=%.5f' % (epoch, i, tr_loss / tr_examples))
                    tr_loss = 0
                    tr_examples = 0

                if global_step % eval_step == 0:
                    f1 = evaluate(model, dev_batches, dev_ner)
                    if f1 > best_result:
                        best_result = f1
                        logger.info('!!! Best valid (epoch=%d): %.2f' % (epoch, f1 * 100))
                        save_model(model, hparams)

    if hparams.do_eval:
        hparams.bert_model_dir = hparams.output_dir
        model = EntityModel(hparams, num_ner_labels=num_ner_labels)
        eval_dev_data = Dataset(hparams.dev_data_path)
        prediction_file = os.path.join(hparams.output_dir, hparams.dev_pred_filename)
        eval_dev_samples, eval_dev_ner = convert_dataset_to_samples(eval_dev_data, hparams.max_span_length,
                                                                    ner_label2id=ner_label2id,
                                                                    context_window=hparams.context_window)
        eval_dev_batches = batchify(eval_dev_samples, hparams.eval_batch_size)
        evaluate(model, eval_dev_batches, eval_dev_ner)
        output_ner_predictions(model, eval_dev_batches, eval_dev_data, output_file=prediction_file)
        if hparams.eval_test:
            eval_test_data = Dataset(hparams.test_data)
            prediction_file = os.path.join(hparams.output_dir, hparams.test_pred_filename)
            eval_test_samples, eval_test_ner = convert_dataset_to_samples(eval_test_data, hparams.max_span_length,
                                                                          ner_label2id=ner_label2id,
                                                                          context_window=hparams.context_window)
            eval_test_batches = batchify(eval_test_samples, hparams.eval_batch_size)
            evaluate(model, eval_test_batches, eval_test_ner)
            output_ner_predictions(model, eval_test_batches, eval_test_data, output_file=prediction_file)
