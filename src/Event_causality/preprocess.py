import pickle
from os.path import exists

from transformers import BertTokenizer

from read_document import make_raw_pickle


def get_sentence_number(s, all_token):
    tid = s.split('_')[0]
    for token in all_token:
        if token[0] == tid:
            return token[1]


def nth_sentence(sen_no, all_token):
    res = []
    for token in all_token:
        if token[1] == sen_no:
            res.append(token[-1])
    return res


def get_sentence_offset(s, all_token):
    positions = []
    for c in s.split('_'):
        token = all_token[int(c) - 1]
        positions.append(token[2])
    return '_'.join(positions)


def make_data_pickle(raw_pickle, data_pickle, tokenizer, debug=True):
    # make_raw_pickle()
    if not exists(raw_pickle) or debug:
        make_raw_pickle()
    with open(raw_pickle, 'rb') as f:
        documents = pickle.load(f)

    data_set = []
    count = 0
    for doc in documents:
        [all_token,
         ecb_star_events,
         ecb_star_events_plotLink,
         evaluation_data] \
            = documents[doc]

        for event1 in ecb_star_events:
            for event2 in ecb_star_events:
                if event1 == event2:  # event ID
                    continue
                offset1 = ecb_star_events[event1]
                offset2 = ecb_star_events[event2]

                rel = 'NULL'
                for elem in evaluation_data:
                    e1, e2, value = elem
                    if e1 == offset1 and e2 == offset2:
                        rel = value
                sen_s = get_sentence_number(offset1, all_token)
                sen_t = get_sentence_number(offset2, all_token)

                if abs(int(sen_s) - int(sen_t)) == 0:  # #
                    if rel != 'NULL':
                        count += 1
                    sentence_s = nth_sentence(sen_s, all_token)
                    sentence_t = nth_sentence(sen_t, all_token)
                    sen_offset1 = get_sentence_offset(offset1, all_token)
                    sen_offset2 = get_sentence_offset(offset2, all_token)

                    span1 = [int(x) for x in sen_offset1.split('_')]
                    span2 = [int(x) for x in sen_offset2.split('_')]

                    sentence_s = ['[CLS]'] + sentence_s + ['[SEP]']
                    sentence_t = ['[CLS]'] + sentence_t + ['[SEP]']

                    span1 = list(map(lambda x: x + 1, span1))
                    span2 = list(map(lambda x: x + 1, span2))

                    sentence_vec_s = []
                    sentence_vec_t = []

                    span1_vec = []
                    span2_vec = []
                    for i, w in enumerate(sentence_s):
                        tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                        xx = tokenizer.convert_tokens_to_ids(tokens)

                        if i in span1:
                            span1_vec.extend(list(range(len(sentence_vec_s), len(sentence_vec_s) + len(xx))))

                        sentence_vec_s.extend(xx)

                    for i, w in enumerate(sentence_t):
                        tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                        xx = tokenizer.convert_tokens_to_ids(tokens)

                        if i in span2:
                            span2_vec.extend(list(range(len(sentence_vec_t), len(sentence_vec_t) + len(xx))))

                        sentence_vec_t.extend(xx)

                    data_set.append([doc, sentence_vec_s, sentence_vec_t, span1_vec, span2_vec, rel])

    # print(len(data_set))
    # print(data_set[0])
    # print(count)
    with open(data_pickle, 'wb') as f:
        pickle.dump(data_set, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # todo 可以把read_document改过来
    processed_files = 'processed_files'
    raw_pickle0 = f'{processed_files}/document_raw.pickle'
    data_pickle0 = f'{processed_files}/data.pickle'
    tokenizer0 = BertTokenizer.from_pretrained('bert-base-uncased')
    make_data_pickle(raw_pickle=raw_pickle0, data_pickle=data_pickle0, tokenizer=tokenizer0)
