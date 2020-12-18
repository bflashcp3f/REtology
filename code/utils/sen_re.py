#!/usr/bin/env python

import json
import os
import torch
import sys
import argparse
import numpy as np

# from constants.tacred import *
from constants import tacred
from constants import kbp37
from constants import semeval

from collections import Counter, OrderedDict
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def mask_entities(tokens, entity_offsets, subj_entity_start, subj_entity_end,
                  obj_entity_start, obj_entity_end):
    subj_entity, obj_entity = entity_offsets

    #     print(tokens, entity_offsets, subj_entity_start, subj_entity_end,
    #                   obj_entity_start, obj_entity_end)

    if subj_entity[0] < obj_entity[0]:
        tokens = tokens[:subj_entity[0]] + [subj_entity_start] + tokens[subj_entity[0]:subj_entity[1]] + \
                 [subj_entity_end] + tokens[subj_entity[1]:obj_entity[0]] + [obj_entity_start] + \
                 tokens[obj_entity[0]:obj_entity[1]] + [obj_entity_end] + tokens[obj_entity[1]:]

        subj_entity = (subj_entity[0] + 1, subj_entity[1] + 1)
        obj_entity = (obj_entity[0] + 3, obj_entity[1] + 3)

    else:
        tokens = tokens[:obj_entity[0]] + [obj_entity_start] + tokens[obj_entity[0]:obj_entity[1]] + \
                 [obj_entity_end] + tokens[obj_entity[1]:subj_entity[0]] + [subj_entity_start] + \
                 tokens[subj_entity[0]:subj_entity[1]] + [subj_entity_end] + tokens[subj_entity[1]:]

        obj_entity = (obj_entity[0] + 1, obj_entity[1] + 1)
        subj_entity = (subj_entity[0] + 3, subj_entity[1] + 3)

    #     print(tokens, (subj_entity, obj_entity))
    return tokens, (subj_entity, obj_entity)


def convert_token(token):
    """ Convert PTB tokens to normal tokens """
    if (token.lower() == '-lrb-'):
        return '('
    elif (token.lower() == '-rrb-'):
        return ')'
    elif (token.lower() == '-lsb-'):
        return '['
    elif (token.lower() == '-rsb-'):
        return ']'
    elif (token.lower() == '-lcb-'):
        return '{'
    elif (token.lower() == '-rcb-'):
        return '}'
    return token


def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file',
                        help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args


def score(key, prediction, no_relation="no_relation", verbose=False):

    # NO_RELATION = "no_relation"

    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]

        if gold == no_relation and guess == no_relation:
            pass
        elif gold == no_relation and guess != no_relation:
            guessed_by_relation[guess] += 1
        elif gold != no_relation and guess == no_relation:
            gold_by_relation[gold] += 1
        elif gold != no_relation and guess != no_relation:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        print("")

    # Print the aggregate score
    if verbose:
        print("Final Score:")
    prec_micro = 0.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    #     if verbose:
    print("Precision (micro): {:.3%}".format(prec_micro))
    print("   Recall (micro): {:.3%}".format(recall_micro))
    print("       F1 (micro): {:.3%}".format(f1_micro))
    return prec_micro, recall_micro, f1_micro


def eval(trained_model, eval_dataloader, device, id2label, negative_label="no_relation"):

    gold_list = []
    pred_list = []

    # Evaluate data for one epoch
    for batch in eval_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels, b_subj_idx, b_obj_idx = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up dev
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = trained_model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                           subj_ent_start=b_subj_idx, obj_ent_start=b_obj_idx
                           )

        # Move logits and labels to CPU
        logits = logits[0].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        gold_list += np.argmax(logits, axis=1).tolist()
        pred_list += label_ids.tolist()

    prec, rec, f1 = score([id2label[gold_id] for gold_id in gold_list],
                          [id2label[pred_id] for pred_id in pred_list],
                          no_relation=negative_label
                          )

    return prec, rec, f1

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, sentence, label, relation, subj_ent_start, obj_ent_start, sen_len):
        self.sentence = sentence
        self.label = label
        self.relation = relation
        self.subj_ent_start = subj_ent_start
        self.obj_ent_start = obj_ent_start
        self.sen_len = sen_len


class DataProcessor(object):
    """Processor for the TACRED data set."""

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_tacred_json(self, file_name, encode_ent_type=False):
        """See base class."""
        # return load_from_json(os.path.join(self.data_dir, file_name))

        data_path = os.path.join(self.data_dir, file_name)

        feature_list = []

        with open(data_path) as f:
            line = f.readlines()
            cases = json.loads(line[0])

            for case in cases[:]:
                sen_id = case[u'id']
                token_list = [convert_token(item) for item in case[u'token']]
                # token_list = case[u'token']
                relation = case[u'relation']

                subj_start = case[u'subj_start']
                subj_end = case[u'subj_end']

                subj_type = case[u'subj_type']

                obj_start = case[u'obj_start']
                obj_end = case[u'obj_end']

                obj_type = case[u'obj_type']

                if encode_ent_type:
                    ent_offset = [[subj_start, subj_end + 1], [obj_start, obj_end + 1]]

                    subj_ent_start = "[subj-" + subj_type.lower() + "-start]"
                    subj_ent_end = "[subj-" + subj_type.lower() + "-end]"

                    obj_ent_start = "[obj-" + obj_type.lower() + "-start]"
                    obj_ent_end = "[obj-" + obj_type.lower() + "-end]"
                else:
                    # If we don't encode type info into the special token, we just treat the first
                    # entity as the subject, and use the special token "[e1-start]" and "[e1-start]"
                    # if subj_start < obj_start:
                    #     ent_offset = [[subj_start, subj_end + 1], [obj_start, obj_end + 1]]
                    # else:
                    #     ent_offset = [[obj_start, obj_end + 1], [subj_start, subj_end + 1]]
                    ent_offset = [[subj_start, subj_end + 1], [obj_start, obj_end + 1]]
                    subj_ent_start = "[subj-start]"
                    subj_ent_end = "[subj-end]"
                    obj_ent_start = "[obj-start]"
                    obj_ent_end = "[obj-end]"

                processed_token_list, _ = mask_entities(token_list, ent_offset,
                                                        subj_ent_start, subj_ent_end,
                                                        obj_ent_start, obj_ent_end,
                                                        )

                feature_list.append(InputFeatures(sentence=' '.join(processed_token_list), label=tacred.LABEL_TO_ID[relation],
                                                  relation=relation, subj_ent_start=subj_ent_start, obj_ent_start=obj_ent_start,
                                                  sen_len=len(processed_token_list)))

            #             break

            print("The number of mentions:", len(feature_list))

        return feature_list

    def load_kbp37_txt(self, file_name):
        """See base class."""

        data_path = os.path.join(self.data_dir, file_name)

        def load_from_txt(data_path, verbose=False, strip=True):
            examples = []

            with open(data_path, encoding='utf-8') as infile:
                while True:
                    line = infile.readline()
                    if len(line) == 0:
                        break

                    if strip:
                        line = line.strip()

                    examples.append(line)

            if verbose:
                print("{} examples read in {} .".format(len(examples), data_path))
            return examples

        org_data = load_from_txt(data_path)
        assert len(org_data) % 4 == 0

        feature_list = []

        for idx in range(0, len(org_data), 4):
            sid_sen_str = org_data[idx]
            relation = org_data[idx+1]

            if len(sid_sen_str.split("\t")) != 2:
                print(sid_sen_str)
                raise

            sid, sen_str = sid_sen_str.split("\t")
            assert sen_str[:2] =='" ' and sen_str[-2:] == ' "'
            sen_str = sen_str[2:-2] # Remove the prefix and suffix
            subj_ent_start = "<e1>"
            obj_ent_start = "<e2>"

            feature_list.append(InputFeatures(sentence=sen_str, label=kbp37.LABEL_TO_ID[relation],
                                              relation=relation, subj_ent_start=subj_ent_start,
                                              obj_ent_start=obj_ent_start,
                                              sen_len=len(sen_str.split())))

        print("The number of mentions:", len(feature_list))

        return feature_list

    def load_semeval_txt(self, file_name, set_name):
        """See base class."""

        data_path = os.path.join(self.data_dir, file_name)

        def load_from_txt(data_path, verbose=False, strip=True):
            examples = []

            with open(data_path, encoding='utf-8') as infile:
                while True:
                    line = infile.readline()
                    if len(line) == 0:
                        break

                    if strip:
                        line = line.strip()

                    examples.append(line)

            if verbose:
                print("{} examples read in {} .".format(len(examples), data_path))
            return examples

        org_data = load_from_txt(data_path)
        assert len(org_data) % 4 == 0

        feature_list = []

        for idx in range(0, len(org_data), 4):
            sid_sen_str = org_data[idx]
            relation = org_data[idx+1]

            if len(sid_sen_str.split("\t")) != 2:
                print(sid_sen_str)
                raise

            sid, sen_str = sid_sen_str.split("\t")
            assert sen_str.startswith('"') and sen_str.endswith('"')
            sen_str = sen_str[1:-1] # Remove the prefix and suffix

            subj_ent_start = "<e1>"
            obj_ent_start = "<e2>"

            feature_list.append(InputFeatures(sentence=sen_str, label=semeval.LABEL_TO_ID[relation],
                                              relation=relation, subj_ent_start=subj_ent_start,
                                              obj_ent_start=obj_ent_start,
                                              sen_len=len(sen_str.split())))

        if set_name == "train":
            feature_list = feature_list[:6500]
        elif set_name == "dev":
            feature_list = feature_list[6500:]

        print("The number of mentions:", len(feature_list))

        return feature_list


def convert_features_to_dataloader(args, feature_list, tokenizer, logger, file_train=False):
    """Loads a data file into a list of `InputBatch`s."""

    # if file_train:
    # print(f"{tokenizer.cls_token} {sens[0]} {tokenizer.eos_token}")
    logger.info(f"{tokenizer.cls_token} {feature_list[0].sentence} {tokenizer.sep_token}")

    # Preprocess the sequence
    tokenized_texts = [tokenizer.tokenize(f"{tokenizer.cls_token} {each_case.sentence} {tokenizer.sep_token}")
                       for each_case in feature_list]

    # if file_train:
    # print(tokenized_texts[0])
    logger.info("tokenized_texts[0]: " + " ".join(tokenized_texts[0]))

    # Pad our input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=args.max_seq_length, dtype="long", value=tokenizer.pad_token_id,
                              truncating="post", padding="post")

    # if file_train:
    # print(input_ids[0])
    # print(input_ids.shape)
    logger.info("input_ids[0]: " + " ".join([str(item) for item in input_ids[0]]))

    subj_idx_list = [tokenized_texts[sen_idx].index(feature_list[sen_idx].subj_ent_start)
                     if tokenized_texts[sen_idx].index(feature_list[sen_idx].subj_ent_start) < args.max_seq_length else args.max_seq_length - 1
                     for sen_idx in range(len(tokenized_texts))
                     ]
    logger.info(f"subj_idx_list[0]: {subj_idx_list[0]}")

    obj_idx_list = [tokenized_texts[sen_idx].index(feature_list[sen_idx].obj_ent_start)
                    if tokenized_texts[sen_idx].index(feature_list[sen_idx].obj_ent_start) < args.max_seq_length else args.max_seq_length - 1
                    for sen_idx in range(len(tokenized_texts))
                    ]
    logger.info(f"obj_idx_list[0]: {obj_idx_list[0]}")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i != tokenizer.pad_token_id) for i in seq]
        attention_masks.append(seq_mask)

    # if file_train:
    #     print(attention_masks[0])
    logger.info("attention_masks[0]: " + " ".join([str(item) for item in attention_masks[0]]))

    # Convert all of our data into torch tensors, the required datatype for our model

    inputs = torch.tensor(input_ids)
    labels = torch.tensor([item.label for item in feature_list])
    masks = torch.tensor(attention_masks)

    subj_idxs = torch.tensor(subj_idx_list)
    obj_idxs = torch.tensor(obj_idx_list)

    if file_train:
        data = TensorDataset(inputs, masks, labels, subj_idxs, obj_idxs)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size)
    else:
        data = TensorDataset(inputs, masks, labels, subj_idxs, obj_idxs)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size)

    return dataloader


def convert_examples_to_features(args, sens, labels, subj_ent_start_list, obj_ent_start_list, tokenizer,
                                 file_train=False):
    """Loads a data file into a list of `InputBatch`s."""

    # if file_train:
    # print(f"{tokenizer.cls_token} {sens[0]} {tokenizer.eos_token}")
    logger.info(f"{tokenizer.cls_token} {sens[0]} {tokenizer.sep_token}")

    # Preprocess the sequence
    tokenized_texts = [tokenizer.tokenize(f"{tokenizer.cls_token} {sent} {tokenizer.sep_token}") for sent in sens]

    # if file_train:
    # print(tokenized_texts[0])
    logger.info("tokenized_texts[0]: " + " ".join(tokenized_texts[0]))

    # Pad our input tokens
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=args.max_seq_length, dtype="long", value=tokenizer.pad_token_id,
                              truncating="post", padding="post")

    # if file_train:
    # print(input_ids[0])
    # print(input_ids.shape)
    logger.info("input_ids[0]: " + " ".join([str(item) for item in input_ids[0]]))

    subj_idx_list = [tokenized_texts[sen_idx].index(subj_ent_start_list[sen_idx])
                     if tokenized_texts[sen_idx].index(
        subj_ent_start_list[sen_idx]) < args.max_seq_length else args.max_seq_length - 1
                     for sen_idx in range(len(tokenized_texts))
                     ]
    logger.info(f"subj_idx_list[0]: {subj_idx_list[0]}")

    obj_idx_list = [tokenized_texts[sen_idx].index(obj_ent_start_list[sen_idx])
                    if tokenized_texts[sen_idx].index(
        obj_ent_start_list[sen_idx]) < args.max_seq_length else args.max_seq_length - 1
                    for sen_idx in range(len(tokenized_texts))
                    ]
    logger.info(f"obj_idx_list[0]: {obj_idx_list[0]}")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i != tokenizer.pad_token_id) for i in seq]
        attention_masks.append(seq_mask)

    # if file_train:
    #     print(attention_masks[0])
    logger.info("attention_masks[0]: " + " ".join([str(item) for item in attention_masks[0]]))

    # Convert all of our data into torch tensors, the required datatype for our model

    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    masks = torch.tensor(attention_masks)

    subj_idxs = torch.tensor(subj_idx_list)
    obj_idxs = torch.tensor(obj_idx_list)

    if file_train:
        data = TensorDataset(inputs, masks, labels, subj_idxs, obj_idxs)
        sampler = RandomSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size)
    else:
        data = TensorDataset(inputs, masks, labels, subj_idxs, obj_idxs)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=args.batch_size)

    return dataloader