import torch.nn as nn
import os
import torch
from data_load import idx2trigger
from utils import calc_metric, find_triggers
import logging
logging.basicConfig(level=logging.INFO, filename='output.txt', filemode='w')


def train(model, iterator, optimizer, criterion):
    if hasattr(model, 'module'):
        model = model.module
    model.train()
    for i, batch in enumerate(iterator):
        tokens_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = batch
        optimizer.zero_grad()
        trigger_logits, \
        triggers_y_2d, \
        trigger_hat_2d,\
        argument_hidden, \
        argument_keys = \
            model.predict_triggers(tokens_x_2d=tokens_x_2d, head_indexes_2d=head_indexes_2d,
                                   triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d)

        trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
        trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))

        if len(argument_keys) > 0:
            argument_logits,\
            arguments_y_1d,\
            argument_hat_1d,\
            argument_hat_2d \
                = model.predict_arguments(argument_hidden, argument_keys, arguments_2d)
            argument_loss = criterion(argument_logits, arguments_y_1d)
            loss = trigger_loss + 2 * argument_loss
        else:
            loss = trigger_loss

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        loss.backward()
        optimizer.step()
        #
        # if i % 10 == 0:  # monitoring
        #     print("step: {}, loss: {}".format(i, loss.item()))


def evaluate(model, iterator, f_name, f1_max=0):
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    words_all = []
    triggers_all = []
    triggers_hat_all = []
    arguments_all = []
    arguments_hat_all = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, triggers_y_2d, arguments_2d, seqlens_1d, head_indexes_2d, words_2d, triggers_2d = batch

            trigger_logits, triggers_y_2d, trigger_hat_2d, argument_hidden, argument_keys \
                = model.predict_triggers(tokens_x_2d=tokens_x_2d, head_indexes_2d=head_indexes_2d,
                                         triggers_y_2d=triggers_y_2d, arguments_2d=arguments_2d)

            words_all.extend(words_2d)
            triggers_all.extend(triggers_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())
            arguments_all.extend(arguments_2d)

            if len(argument_keys) > 0:
                argument_logits, arguments_y_1d, argument_hat_1d, argument_hat_2d = \
                    model.predict_arguments(argument_hidden, argument_keys, arguments_2d)
                arguments_hat_all.extend(argument_hat_2d)
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)

    triggers_true = []
    triggers_pred = []
    arguments_true = []
    arguments_pred = []
    with open('temp', 'w') as f_out:
        for i, (words, triggers, triggers_hat, arguments, arguments_hat) \
                in enumerate(zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_true.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_pred.append((i, t_start, t_end, t_type_str, a_start, a_end, a_type_idx))

            for w, t, t_h in zip(words[1:-1], triggers, triggers_hat):
                f_out.write('{}\t{}\t{}\n'.format(w, t, t_h))
            f_out.write('#arguments#{}\n'.format(arguments['events']))
            f_out.write('#arguments_hat#{}\n'.format(arguments_hat['events']))
            f_out.write("\n")

    # print(classification_report([idx2trigger[idx] for idx in y_true], [idx2trigger[idx] for idx in y_pred]))

    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
    argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
    triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
    arguments_true = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in arguments_true]
    arguments_pred = [(item[0], item[1], item[2], item[3], item[4], item[5]) for item in arguments_pred]
    argument_p_, argument_r_, argument_f1_ = calc_metric(arguments_true, arguments_pred)
    trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred)
    f1 = argument_f1 + argument_f1_ + trigger_f1 + trigger_f1_
    if f1 > f1_max:
        print('[trigger classification]')
        print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))

        print('[argument classification]')
        print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))
        print('[trigger identification]')
        print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))

        print('[argument identification]')
        print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p_, argument_r_, argument_f1_))

        logging.info('[trigger classification]')
        logging.info('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))

        logging.info('[argument classification]')
        logging.info('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))
        logging.info('[trigger identification]')
        logging.info('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))

        logging.info('[argument identification]')
        logging.info('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p_, argument_r_, argument_f1_))

        metric = '[trigger classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p, trigger_r,
                                                                                    trigger_f1)
        metric += '[argument classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p, argument_r,
                                                                                      argument_f1)
        metric += '[trigger identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p_, trigger_r_,
                                                                                     trigger_f1_)
        metric += '[argument identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p_, argument_r_,
                                                                                      argument_f1_)
        final = f_name + ".P%.2f_R%.2f_F%.2f" % (trigger_p, trigger_r, trigger_f1)
        with open(final, 'w') as f_out:
            result = open("temp", "r").read()
            f_out.write("{}\n".format(result))
            f_out.write(metric)
        os.remove("temp")
    return f1
