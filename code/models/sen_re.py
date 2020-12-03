# import tensorflow as tf
import torch
# import pandas as pd
# import io
# import numpy as np
# import matplotlib.pyplot as plt
# import json
# import os
# import copy
# import sys
# import argparse

from constants import *
from utils import *

from transformers import BertTokenizer, BertModel, BertPreTrainedModel, BertConfig, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification

from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class BertEMES(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEMES, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                subj_ent_start=None, obj_ent_start=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # outputs = self.bert(input_ids_new, attention_mask=attention_mask_new)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Take the representation for 'SUBJ' token
        subj_ent_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_ent_start)])

        # Take the representation for 'OBJ' token
        obj_ent_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_ent_start)])

        ent_output = torch.cat([subj_ent_output, obj_ent_output], dim=1)

        bag_output = self.dropout(ent_output)

        logits = self.classifier(bag_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaEMES(BertPreTrainedModel):
    config_class = RobertaConfig
    #     pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaEMES, self).__init__(config)
        self.num_labels = config.num_labels
        self.hidden_size = config.hidden_size

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #         self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.classifier = nn.Linear(config.hidden_size * 2, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None,
                position_ids=None, head_mask=None,
                subj_ent_start=None, obj_ent_start=None):

        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # outputs = self.bert(input_ids_new, attention_mask=attention_mask_new)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        # Take the representation for 'SUBJ' token
        subj_ent_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, subj_ent_start)])

        # Take the representation for 'OBJ' token
        obj_ent_output = torch.cat([a[i].unsqueeze(0) for a, i in zip(sequence_output, obj_ent_start)])

        ent_output = torch.cat([subj_ent_output, obj_ent_output], dim=1)

        bag_output = self.dropout(ent_output)

        logits = self.classifier(bag_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        #         print("outputs: ", outputs)

        return outputs  # (loss), logits, (hidden_states), (attentions)