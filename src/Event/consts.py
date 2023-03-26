import json
import os
from os.path import exists

from utils import get_event_type, write_json, process, get_argument_role, get_entity_type, \
    get_event_type_ace, get_argument_role_ace, get_entity_type_ace
from params import get_hparams
max_length = 400

NONE = 'O'
PAD = "[PAD]"
UNK = "[UNK]"

# for BERT
CLS = '[CLS]'
SEP = '[SEP]'

hp = get_hparams()
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
# trigger
if not os.path.exists('{}/event_type_list.json'.format(data_dir)):
    triggers0 = set()
    if hp.data_class == 'ace':
        triggers1 = get_event_type_ace(train_set, triggers0)
        triggers2 = get_event_type_ace(dev_set, triggers1)
        triggers3 = get_event_type_ace(test_set, triggers2)
        TRIGGERS = list(triggers3)
    else:
        triggers1 = get_event_type(train_set, triggers0)
        triggers2 = get_event_type(dev_set, triggers1)
        triggers3 = get_event_type(test_set, triggers2)
        TRIGGERS = list(triggers3)
    write_json(data=TRIGGERS, fn='{}/event_type_list.json'.format(data_dir))
else:
    with open('{}/event_type_list.json'.format(data_dir), mode='r') as f:
        TRIGGERS = json.load(f)

# argument
if not os.path.exists('{}/argument_role_list.json'.format(data_dir)):
    if hp.data_class == 'ace':
        argument = get_argument_role_ace(train_set)
        argument1 = get_argument_role_ace(dev_set, argument)
        argument2 = get_argument_role_ace(test_set, argument1)
        ARGUMENTS = list(argument2)
    else:
        argument = get_argument_role(train_set)
        argument1 = get_argument_role(dev_set, argument)
        argument2 = get_argument_role(test_set, argument1)
        ARGUMENTS = list(argument2)
    write_json(data=ARGUMENTS, fn='{}/argument_role_list.json'.format(data_dir))
else:
    with open('{}/argument_role_list.json'.format(data_dir), mode='r') as f:
        ARGUMENTS = json.load(f)

# entity
if not os.path.exists('{}/entity_type_list.json'.format(data_dir)):
    if hp.data_class == 'ace':
        entities = get_entity_type_ace(train_set)
        entities = get_entity_type_ace(dev_set, entities)
        entities = get_entity_type_ace(test_set, entities)
        ENTITIES = list(entities)
    else:
        entities, mention_type = get_entity_type(train_set)
        entities, mention_type = get_entity_type(dev_set, entities, mention_type)
        entities, mention_type = get_entity_type(test_set, entities, mention_type)
        ENTITIES = list(entities)
    write_json(data=ENTITIES, fn='{}/entity_type_list.json'.format(data_dir))
else:
    with open('{}/entity_type_list.json'.format(data_dir), mode='r') as f:
        ENTITIES = json.load(f)


# if __name__ == "__main__":
#     hp = get_hparams()
#     data_dir = None
#     if hp.data_class == 'wiki_src':
#         data_dir = 'Datasets/wiki_processed_data/source/'
#     elif hp.data_class == 'wiki_info':
#         data_dir = 'Datasets/wiki_processed_data/info/'
#     elif hp.data_class == 'ace':
#         data_dir = 'Datasets/ace2005/'
#     train_set = data_dir + 'train.json'
#     test_set = data_dir + 'test.json'
#     dev_set = data_dir + 'dev.json'
    # # 在没有预处理文件时可以使用。否则，直接创建TRIGGER AND ARGUMENTS即可
    # test_event1 = [{"id": "scenario_en_kairos_14-E3",
    #                 "event_type": "Cognitive.IdentifyCategorize.Unspecified",
    #                 "trigger": (30, 31, 'Cognitive.IdentifyCategorize.Unspecified'),
    #                 "arguments": []},
    #                {"id": "scenario_en_kairos_14-E2",
    #                 "event_type": "Cognitive.Inspection.SensoryObserve",
    #                 "trigger": (88, 89, 'Cognitive.Inspection.SensoryObserve'),
    #                 "arguments": []},
    #                {"id": "scenario_en_kairos_14-E1",
    #                 "event_type": "Cognitive.IdentifyCategorize.Unspecified",
    #                 "trigger": (166, 167, 'Cognitive.IdentifyCategorize.Unspecified'),
    #                 "arguments": []}]
    #
    # test_event2 = [{"id": "scenario_en_kairos_65-E1", "event_type": "Conflict.Attack.Unspecified",
    #                 "trigger": (50, 51, 'Conflict.Attack.Unspecified'),
    #                 "arguments": []},
    #                {"id": "scenario_en_kairos_65-E2", "event_type": "Life.Injure.Unspecified",
    #                 "trigger": (62, 63, 'Life.Injure.Unspecified'),
    #                 "arguments": [(59, 61, 'Victim')]},
    #                {"id": "scenario_en_kairos_65-E3", "event_type": "Conflict.Attack.DetonateExplode",
    #                 "trigger": (65, 66, 'Conflict.Attack.DetonateExplode'),
    #                 "arguments": []},
    #                {"id": "scenario_en_kairos_65-E4", "event_type": "Conflict.Attack.DetonateExplode",
    #                 "trigger": (417, 419, 'Conflict.Attack.DetonateExplode'),
    #                 "arguments": []},
    #                {"id": "scenario_en_kairos_65-E5", "event_type": "Conflict.Attack.DetonateExplode",
    #                 "trigger": (433, 434, 'Conflict.Attack.DetonateExplode'),
    #                 "arguments": []}]

#     process_data_fn = 'wiki_processed_data/source'
    # train_set = "../../wiki_events_dataset/info_data/train_info.jsonl"
    # dev_set = "../../wiki_events_dataset/info_data/dev_info.jsonl"
    # test_set = "../../wiki_events_dataset/info_data/test_info.jsonl"
    # process_data_fn = 'processed_data/info'
    # if not exists(process_data_fn):
    #     os.makedirs(process_data_fn)

    # trigger
    # if not os.path.exists('{}/event_type_list.json'.format(process_data_fn)):
    #     triggers0 = set()
    #     triggers1 = get_event_type(train_set, triggers0)
    #     triggers2 = get_event_type(dev_set, triggers1)
    #     triggers3 = get_event_type(test_set, triggers2)
    #     TRIGGERS = list(triggers3)
    #     write_json(data=TRIGGERS, fn='{}/event_type_list.json'.format(process_data_fn))
    # else:
    #     with open('{}/event_type_list.json'.format(process_data_fn), mode='r') as f:
    #         TRIGGERS = json.load(f)
    #
    # # argument
    # if not os.path.exists('{}/argument_role_list.json'.format(process_data_fn)):
    #     argument = get_argument_role(train_set)
    #     argument1 = get_argument_role(dev_set, argument)
    #     argument2 = get_argument_role(test_set, argument1)
    #     ARGUMENTS = list(argument2)
    #     write_json(data=ARGUMENTS, fn='{}/argument_role_list.json'.format(process_data_fn))
    # else:
    #     with open('{}/argument_role_list.json'.format(process_data_fn), mode='r') as f:
    #         ARGUMENTS = json.load(f)
    #
    # # entity
    # if not os.path.exists('{}/entity_type_list.json'.format(process_data_fn)):
    #     entities, mention_type = get_entity_type(train_set)
    #     entities, mention_type = get_entity_type(dev_set, entities, mention_type)
    #     entities, mention_type = get_entity_type(test_set, entities, mention_type)
    #     ENTITIES = list(entities)
    #     write_json(data=ENTITIES, fn='{}/entity_type_list.json'.format(process_data_fn))
    # else:
    #     with open('{}/entity_type_list.json'.format(process_data_fn), mode='r') as f:
    #         ENTITIES = json.load(f)
    #
    # processed_train = '{}/train.json'.format(process_data_fn)
    # processed_dev = '{}/dev.json'.format(process_data_fn)
    # processed_test = '{}/test.json'.format(process_data_fn)
    # chunk_test = '{}/test_chunk.json'.format(process_data_fn)
    #
    # over_write = False
    # if not os.path.exists(processed_train) or over_write:
    #     process(train_set, processed_train)
    # if not os.path.exists(processed_dev) or over_write:
    #     process(dev_set, processed_dev)
    # if not os.path.exists(processed_test) or over_write:
    #     process(test_set, processed_test)

