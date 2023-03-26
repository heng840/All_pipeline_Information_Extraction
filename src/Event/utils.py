import json

max_length = 400


def build_vocab(labels, BIO_tagging=True):
    all_labels = ["[PAD]", 'O']
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label


def calc_metric(y_true, y_pred):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1


def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, labels[i][1]])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]



def get_pre_dict(sentence_e_mentions,  # 一个句子，包含了多个event
                 sentence_before_len,
                 context,
                 words,
                 data_new,
                 sent_entity_mentions,
                 entity_id_num_none=0, entity_id_num=0):
    """
    sent_entity_mentions:  todo entity是可以跨句的？
    """
    # label triggers and arguments
    golden_event_mentions = []
    for event in sentence_e_mentions:
        trigger = {"start": event['trigger']['start'] - sentence_before_len,
                   "end": event['trigger']['end'] - sentence_before_len,
                   "text": event['trigger']['text']}
        if trigger['start'] > max_length or trigger['end'] < 0:
            continue
        arguments = event['arguments']
        args_list = []
        for arg in arguments:
            entity_id = arg['entity_id']
            span = []
            for entity in sent_entity_mentions:
                if entity['id'] == entity_id:
                    span = [entity['start'] - sentence_before_len, entity['end'] - sentence_before_len]
                    break
            if span:
                args_dict = {
                    "role": arg['role'],
                    "text": arg['text'],
                    "start": span[0],
                    "end": span[1],
                    # 'span': span
                }
                args_list.append(args_dict)
        golden_event_mentions.append({
            'trigger': trigger,
            'arguments': args_list,
            'event_type': event['event_type']
        })

    entity_mentions = []
    for entity in sent_entity_mentions:
        enti = {'entity_type': entity['entity_type'],
                'start': entity['start'] - sentence_before_len,
                'end': entity['end'] - sentence_before_len}
        if enti['start'] > max_length or enti['end'] < 0:
            continue
        entity_mentions.append(enti)

    data_dict = {
        'sentence': context,
        'words': words,
        'golden_event_mentions': golden_event_mentions,
        'golden_entity_mentions': entity_mentions
    }
    data_new.append(data_dict)
    return entity_id_num_none, entity_id_num


def write_json(data, fn):
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)


def process(wiki_json, processed_json):
    """
    return:
        {'id': data_id, 'content': content, 'occur': type_occur, 'type': TYPE, 'triggers': triggers, 'index': index,
        'args': trigger_args[trigger_str]}
        todo 如何加上args，关键是在arg_span上，根据entity_id进行对应
    """
    data_new = []
    entity_id_num_none, entity_id_num = 0, 0
    # source: train:22/4542, dev:2/428 , test:3/566
    # info: train:1151/4413, dev:99/411 , test:152/556
    with open(wiki_json, 'r') as wiki:
        lines = wiki.readlines()
        for line in lines:
            record = json.loads(line)
            idx2sentence = {idx: tag for idx, tag in enumerate(record['sentences'])}
            event_mentions = record['event_mentions']
            entity_mentions = record['entity_mentions']
            # todo sentence_entity_mentions
            for sent_idx in range(len(idx2sentence)):
                sentence_before_len = 0
                for k in range(sent_idx):
                    sentence_before_len += len(idx2sentence[k][0])

                sentence_e_mentions = []
                sent_entity_mentions = []
                # 一个句子多个event_mention,加入到一个list里
                for e_mention in event_mentions:
                    if e_mention['trigger']['sent_idx'] == sent_idx:
                        sentence_e_mentions.append(e_mention)
                for entity_men in entity_mentions:
                    if entity_men['sent_idx'] == sent_idx:
                        sent_entity_mentions.append(entity_men)
                sentence_words_with_span = idx2sentence[sent_idx][0]
                words = [w[0] for w in sentence_words_with_span]
                context = idx2sentence[sent_idx][1]
                if len(idx2sentence[sent_idx][0]) >= max_length:
                    # continue
                    split_nums = len(idx2sentence[sent_idx][0]) // max_length + 1
                    for s_num in range(split_nums):
                        if s_num < split_nums - 1:
                            split_words = words[s_num * max_length: (s_num + 1) * max_length]
                            split_context = ' '.join(w for w in split_words)
                        else:
                            split_words = words[s_num * max_length:]
                            split_context = ' '.join(w for w in split_words)
                            # entity_id_num_none, entity_id_num = \
                        get_pre_dict(sentence_e_mentions=sentence_e_mentions,
                                     sentence_before_len=sentence_before_len,
                                     context=split_context, words=split_words, data_new=data_new,
                                     sent_entity_mentions=sent_entity_mentions,
                                     # entity_id_num_none=entity_id_num_none, entity_id_num=entity_id_num
                                     )
                        sentence_before_len += max_length
                else:  # 不长的句子
                    # label all occurrence types
                    # entity_id_num_none, entity_id_num = \
                    get_pre_dict(sentence_e_mentions=sentence_e_mentions,
                                 sentence_before_len=sentence_before_len, context=context,
                                 words=words,
                                 data_new=data_new, sent_entity_mentions=sent_entity_mentions,
                                 # entity_id_num_none=entity_id_num_none, entity_id_num=entity_id_num
                                 )
    x = 0

    with open(processed_json, 'w', encoding='utf-8') as f:
        for line in data_new:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')


def process_test(wiki_json, processed_json):
    """
    return:
        {'id': data_id,
        'content': content,
        events->dict_list:
        [{'type': event_type, 'triggers': {"span":[start, end], "word": text},
         {'type':,'trigger':,}...}, {}...]}
        'args': trigger_args[trigger_str]}  这个argument也可以加上试试效果。
    """
    data_new = []
    with open(wiki_json, 'r') as wiki:
        lines = wiki.readlines()
        for line in lines:
            suf_id = 0  # 加在doc_id的后缀
            record = json.loads(line)
            data_id = record['doc_id']
            idx2sentence = {idx: tag for idx, tag in enumerate(record['sentences'])}
            event_mentions = record['event_mentions']
            # 同一个sent_idx聚集到一起
            for sent_idx in range(len(idx2sentence)):
                sentence_before_len = 0
                for k in range(sent_idx):
                    sentence_before_len += len(idx2sentence[k][0])

                sentence_e_mentions = []
                # 一个句子多个event_mention,加入到一个list里
                for e_mention in event_mentions:
                    if e_mention['trigger']['sent_idx'] == sent_idx:
                        sentence_e_mentions.append(e_mention)
                context = idx2sentence[sent_idx][1]
                sentence_words_with_span = idx2sentence[sent_idx][0]
                words = [w[0] for w in sentence_words_with_span]

                # # label triggers and arguments
                event_dict_list = []
                for sent_event in sentence_e_mentions:
                    # get event_dict_list
                    tri_span = [sent_event['trigger']['start'] - sentence_before_len,
                                sent_event['trigger']['end'] - sentence_before_len]
                    event_dict = {'type': sent_event['event_type'],
                                  'triggers': {"span": tri_span}, "word": sent_event['trigger']['text']}
                    event_dict_list.append(event_dict)
                data_new_dict = {'id': data_id + '_' + str(suf_id), 'content': context,
                                 'words': words, 'events': event_dict_list}
                data_new.append(data_new_dict)
                suf_id += 1
    x = data_new
    with open(processed_json, 'w', encoding='utf-8') as f:
        for line in data_new:
            line = json.dumps(line, ensure_ascii=False)
            f.write(line + '\n')


def get_event_type(wiki_json, e_type):
    """
    read wikiEvents,
    return:
        event_type_list
        event_type2arg_role
        shared_args
    """
    with open(wiki_json, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            event_mentions = item['event_mentions']
            for e_mention in event_mentions:
                e_type.add(e_mention['event_type'])
    return e_type


def get_argument_role(wiki_json, arguments=None):
    if arguments is None:
        arguments = set()
    with open(wiki_json, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            event_mentions = item['event_mentions']
            for e_mention in event_mentions:
                for i in range(len(e_mention['arguments'])):
                    arguments.add(e_mention['arguments'][i]['role'])
    return arguments


def get_entity_type(wiki_json, entities=None, mention_type=None):
    if mention_type is None:
        mention_type = set()
    if entities is None:
        entities = set()
    with open(wiki_json, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = json.loads(line)
            entity_mentions = item['entity_mentions']
            for e_mention in entity_mentions:
                entities.add(e_mention['entity_type'])
                mention_type.add(e_mention['mention_type'])
    return entities, mention_type


def get_entity_type_ace(ace_json, entities=None):
    if entities is None:
        entities = set()
    with open(ace_json, 'r') as f:
        lines = json.load(f)
        for line in lines:
            entity_mentions = line['golden-entity-mentions']
            for e_mention in entity_mentions:
                entities.add(e_mention['entity_type'])
    return entities


def get_event_type_ace(ace_json, e_type):
    with open(ace_json, 'r') as f:
        for line in f:
            item = json.loads(line)
        lines = json.load(f)
        for line in lines:
            event_mentions = line['golden-event-mentions']
            for e_mention in event_mentions:
                e_type.add(e_mention['event_type'])
    return e_type


def get_argument_role_ace(ace_json, arguments=None):
    if arguments is None:
        arguments = set()
    with open(ace_json, 'r') as f:
        lines = json.load(f)
        for line in lines:
            event_mentions = line['golden-event-mentions']
            for e_mention in event_mentions:
                for i in range(len(e_mention['arguments'])):
                    arguments.add(e_mention['arguments'][i]['role'])
    return arguments



