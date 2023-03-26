import collections
import os
import os.path
import pickle

from lxml import etree


def read_evaluation_file(fn):
    res = []
    if not os.path.exists(fn):
        return res
    for line in open(fn):
        fields = line.strip().split('\t')
        res.append(fields)
    return res


def all_tokens(filename):
    ecb_plus = etree.parse(filename, etree.XMLParser(remove_blank_text=True))
    root_ecb_plus = ecb_plus.getroot()
    root_ecb_plus.getchildren()

    all_token = []

    for elem in root_ecb_plus.findall('token'):
        temp = (elem.get('t_id'), elem.get('sentence'),
                elem.get('number'), elem.text)
        all_token.append(temp)
    return all_token


def extract_event_CAT(etreeRoot):
    """
    :param etreeRoot: ECB+/ESC XML root
    :return: dictionary with annotated events in ECB+
    """

    event_dict = collections.defaultdict(list)

    for elem in etreeRoot.findall('Markables/'):
        if elem.tag.startswith("ACTION") or elem.tag.startswith("NEG_ACTION"):
            for token_id in elem.findall('token_anchor'):  # the event should have at least one token
                event_mention_id = elem.get('m_id', 'nothing')
                token_mention_id = token_id.get('t_id', 'nothing')
                event_dict[event_mention_id].append(token_mention_id)
    return event_dict


def extract_plotLink(etreeRoot, d):
    """
    :param etreeRoot: ESC XML root
    :param d: dictionary with annotated events in ESC (event_dict)
    :return:
    """
    plot_dict = collections.defaultdict(list)
    for elem in etreeRoot.findall('Relations/'):
        if elem.tag == "PLOT_LINK":
            source_pl = elem.find('source').get('m_id', 'null')
            target_pl = elem.find('target').get('m_id', 'null')
            rel_valu = elem.get('relType', 'null')

            if source_pl in d:
                val1 = "_".join(d[source_pl])
                if target_pl in d:
                    val2 = "_".join(d[target_pl])
                    plot_dict[(val1, val2)] = rel_valu
    return plot_dict


def read_file(ecb_start_new, evaluate_file):
    ecb_star = etree.parse(ecb_start_new, etree.XMLParser(remove_blank_text=True))
    ecb_star_root = ecb_star.getroot()
    ecb_star_root.getchildren()

    ecb_star_events = extract_event_CAT(ecb_star_root)
    ecb_star_events_plotLink = extract_plotLink(ecb_star_root, ecb_star_events)
    evaluation_data = read_evaluation_file(evaluate_file)
    return ecb_star_events, ecb_star_events_plotLink, evaluation_data


def make_corpus(ecb_start_topic, evaluation_topic, datadict):
    if os.path.isdir(ecb_start_topic):
        if ecb_start_topic[-1] != '/':
            ecb_start_topic += '/'
        if evaluation_topic[-1] != '/':
            evaluation_topic += '/'

        for f in os.listdir(evaluation_topic):
            if f.endswith('plus.xml'):
                ecb_file = f
                star_file = ecb_start_topic + f + ".xml"
                evaluate_file = evaluation_topic + f
                ecb_star_events, ecb_star_events_plotLink, evaluation_data = read_file(star_file, evaluate_file)
                for key in ecb_star_events:
                    ecb_star_events[key] = '_'.join(ecb_star_events[key])

                all_token = all_tokens(star_file)
                datadict[star_file] = [all_token, ecb_star_events, ecb_star_events_plotLink, evaluation_data]


def make_raw_pickle():
    version = 'v1.0'
    ECB_star_Topic = '../../DataSets/annotated_data/' + version + '/'
    EvaluationTopic = '../../DataSets/evaluation_format/full_corpus/' + version + '/event_mentions_extended/'

    data_dict = {}
    for topic in os.listdir(f'{ECB_star_Topic}'):
        if os.path.isdir(f'{ECB_star_Topic}' + topic):
            dir1 = ECB_star_Topic + topic
            dir2 = EvaluationTopic + topic
            make_corpus(dir1, dir2, data_dict)

    processed_files = 'processed_files'
    if not os.path.exists(processed_files):
        os.makedirs(processed_files)
    with open(f'{processed_files}/document_raw.pickle', 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)


