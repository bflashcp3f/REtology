
import torch
import numpy as np
import json
import os
import copy
import sys
import argparse
import random
import logging

# from datetime import datetime

from transformers import BertTokenizer, BertModel, BertPreTrainedModel, AdamW, BertConfig, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, RobertaForSequenceClassification

from tqdm import tqdm, trange
from collections import Counter, OrderedDict

import sys
sys.path.append("./code")

from constants.tacred import *
from utils.sen_re import *
from models.sen_re import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(n_gpu, torch.cuda.get_device_name(0))

    # Set up the random seed
    random.seed(args.random_seed)
    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.pytorch_seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.pytorch_seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))

    logger.info(args)
    logger.info("device: {}, n_gpu: {}".format(device, n_gpu))

    Tokenizer = MODELS[args.lm_model]['tokenizer']
    Config = MODELS[args.lm_model]['config']
    EMES = MODELS[args.lm_model]['emes']

    if args.do_train:
        tokenizer = Tokenizer.from_pretrained(args.lm_model)
        tokenizer.add_tokens(GRAMMER_NER_END)
        tokenizer.add_tokens(GRAMMER_NER_START)

        config = Config.from_pretrained(args.lm_model)
        config.num_labels = len(LABEL_TO_ID)

        model = EMES.from_pretrained(args.lm_model, config=config)
        model.resize_token_embeddings(len(tokenizer))

    if args.do_eval:
        tokenizer = Tokenizer.from_pretrained(args.output_dir)
        config = Config.from_pretrained(args.output_dir)
        model = EMES.from_pretrained(args.output_dir, config=config)


    processor = DataProcessor(args.data_dir)

    train_features = processor.load_tacred_json("train.json", encode_ent_type=args.encode_ent_type)
    dev_features = processor.load_tacred_json("dev.json", encode_ent_type=args.encode_ent_type)
    test_features = processor.load_tacred_json("test.json", encode_ent_type=args.encode_ent_type)

    if args.do_train:
        logger.info("Start pre-processing training data")
        train_dataloader = convert_features_to_dataloader(args=args, feature_list=train_features, tokenizer=tokenizer,
                                                          logger=logger, file_train=True)

        logger.info("Start pre-processing dev data")
        dev_dataloader = convert_features_to_dataloader(args=args, feature_list=dev_features, tokenizer=tokenizer,
                                                        logger=logger, file_train=False)

        logger.info("Start pre-processing test data")
        test_dataloader = convert_features_to_dataloader(args=args, feature_list=test_features, tokenizer=tokenizer,
                                                         logger=logger, file_train=False)

    if args.do_eval:
        if args.eval_test:
            logger.info("Start pre-processing test data")
            test_dataloader = convert_features_to_dataloader(args=args, feature_list=test_features, tokenizer=tokenizer,
                                                             logger=logger, file_train=False)
        else:
            logger.info("Start pre-processing dev data")
            dev_dataloader = convert_features_to_dataloader(args=args, feature_list=dev_features, tokenizer=tokenizer,
                                                            logger=logger, file_train=False)


    if n_gpu > 1:
        model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model.cuda()

    # param_optimizer = list(model.named_parameters())
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)

    best_f1 = -1

    if args.do_train:
        for num_epoch in trange(args.num_train_epochs, desc="Epoch"):

            # Set our model to training mode (as opposed to evaluation mode)
            model.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):

                if step % 100 == 0 and step > 0:
                    # print(step, datetime.now())
                    logger.info(f"step-{step}")

                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels, b_subj_idx, b_obj_idx = batch
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss, _ = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels,
                                subj_ent_start=b_subj_idx, obj_ent_start=b_obj_idx
                                )

                if n_gpu > 1:
                    loss = loss.mean()

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                # Update parameters and take a step using the computed gradient
                #         scheduler.step()
                optimizer.step()

                # Update tracking variables
                tr_loss += loss
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
            #         break

            # print("Train loss: {}".format(tr_loss/nb_tr_steps))
            logger.info("\nEpoch: {}".format(num_epoch))
            logger.info("Train loss: {}".format(tr_loss / nb_tr_steps))

            # Put model in evaluation mode to evaluate loss
            model.eval()

            prec_dev, rec_dev, f1_dev = eval(trained_model=model, eval_dataloader=dev_dataloader,
                                             device=device, id2label=ID_TO_LABEL, negative_label=NO_RELATION)

            # print("The performance on dev set: ")
            logger.info("\nThe performance on dev set: ")
            logger.info("Precision (micro): {:.3%}".format(prec_dev))
            logger.info("   Recall (micro): {:.3%}".format(rec_dev))
            logger.info("       F1 (micro): {:.3%}".format(f1_dev))

            prec_test, rec_test, f1_test = eval(trained_model=model, eval_dataloader=test_dataloader,
                                                device=device, id2label=ID_TO_LABEL, negative_label=NO_RELATION)

            # print("\nThe performance on test set: ")
            logger.info("\nThe performance on test set: ")
            logger.info("Precision (micro): {:.3%}".format(prec_test))
            logger.info("   Recall (micro): {:.3%}".format(rec_test))
            logger.info("       F1 (micro): {:.3%}".format(f1_test))

            # Check if the current model is the best model
            if f1_dev > best_f1:
                best_f1 = f1_dev

                self_config = vars(args).copy()

                self_config['precision_dev'] = prec_dev
                self_config['recall_dev'] = rec_dev
                self_config['f1_dev'] = f1_dev

                self_config['precision_test'] = prec_test
                self_config['recall_test'] = rec_test
                self_config['f1_test'] = f1_test

                logger.info("Get a new BEST valid performance!")
                # print(self_config, '\n')

                # Save the model
                if args.save_model:

                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)

                    if n_gpu > 1:
                        model.module.save_pretrained(save_directory=args.output_dir)
                    else:
                        model.save_pretrained(save_directory=args.output_dir)
                    tokenizer.save_pretrained(save_directory=args.output_dir)

                    # Save hyper-parameters (lr, batch_size, epoch, precision, recall, f1)
                    config_path = os.path.join(args.output_dir, 'self_config.json')
                    with open(config_path, 'w') as json_file:
                        json.dump(self_config, json_file)
                    print()

    if args.do_eval:
        model.eval()
        if args.eval_test:
            # Put model in evaluation mode to evaluate loss

            prec_test, rec_test, f1_test = eval(trained_model=model, eval_dataloader=test_dataloader,
                                                device=device, id2label=ID_TO_LABEL, negative_label=NO_RELATION)

            # print("\nThe performance on test set: ")
            logger.info("\nThe performance on test set: ")
            logger.info("Precision (micro): {:.3%}".format(prec_test))
            logger.info("   Recall (micro): {:.3%}".format(rec_test))
            logger.info("       F1 (micro): {:.3%}".format(f1_test))
        else:
            prec_dev, rec_dev, f1_dev = eval(trained_model=model, eval_dataloader=dev_dataloader,
                                             device=device, id2label=ID_TO_LABEL, negative_label=NO_RELATION)

            # print("The performance on dev set: ")
            logger.info("\nThe performance on dev set: ")
            logger.info("Precision (micro): {:.3%}".format(prec_dev))
            logger.info("   Recall (micro): {:.3%}".format(rec_dev))
            logger.info("       F1 (micro): {:.3%}".format(f1_dev))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_model", default=None, type=str, required=True)
    parser.add_argument("--gpu_ids", default=None, type=str, required=True)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument('--encode_ent_type', action='store_true', help="Encode the entity type into the special token")
    parser.add_argument("--save_model", action='store_true', help="Save trained checkpoints.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--eval_test", action="store_true", help="Whether to evaluate on final test set.")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Total batch size for training and testing.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Gradient clip.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed_set', type=int, default=0,
                        help="random seed for random library")
    parser.add_argument('--random_seed', type=int, default=1234,
                        help="random seed for random library")
    parser.add_argument('--numpy_seed', type=int, default=1234,
                        help="random seed for numpy")
    parser.add_argument('--pytorch_seed', type=int, default=1234,
                        help="random seed for pytorch")
    args = parser.parse_args()
    main(args)

