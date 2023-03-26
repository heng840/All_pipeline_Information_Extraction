max_length = 400


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
