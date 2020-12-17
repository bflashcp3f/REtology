# Relation Extraction (RE) under BERTology
This repo includes the PyTorch code of BERT-based models for RE tasks and related datasets.

## Requirements
Tested verisions.
- python == 3.7.7
- torch == 1.5.0
- transformers == 3.0.2
- keras == 2.3.1
- tensorflow-gpu == 1.15.0

## Pre-trained Models
We use BERT for example but you can also use RoBERTa by simply altering 
the argument `lm_model` (with different hyper-parameters).

## Tasks
### Sentence-Level RE
In this setup, we predict the relation between two named entities in the same sentence. We use the SOTA `Entity Markers-Entity Start (EMES)` architecture 
to generate relation representation [(Soares et al., 2019)](https://arxiv.org/abs/1906.03158), where four special 
tokens such as `[E1-START]` and `[E1-END]` (`[SUBJ-START]` and `[SUBJ-END]` if we know which entity is the subject entity) are used to specify the positions of two entities, and the contextualized
embeddings of `[E1-START]` and `[E2-START]` are concatenated as input to a linear layer to predict the relation.

#### TACRED [[paper]](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf) [[data]](https://catalog.ldc.upenn.edu/LDC2018T24)

```
export TACRED_DATA_DIR=<TACRED_DATA_DIR>
export OUT_DIR=<$OUT_DIR>
python code/finetune_tasks/run_tacred.py \
  --data_dir $TACRED_DATA_DIR \
  --gpu_ids 0,1 \
  --lm_model bert-base-uncased \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --max_seq_length 128 \
  --output_dir $OUT_DIR \
  --do_train \
  --save_model
```

TACRED dataset provides entity type information, so encoding such information in the special token 
can further improve the performance. For example, if the entity type is `PERSON`, we can change the special token `[SUBJ-START]` 
to `[SUBJ-PERSON-START]` by adding `--encode_ent_type` in above command. Similar
method has been published in Zhong and Chen (2020) (didn't know this paper when the model was implemented).

The results on test set:

|                   | w/o entity type    | w/ entity type  | 
| ----------------------  | ------------- | ---------  | 
| BERT (base)             | 68.31         | 70.43      | 
| RoBERTa (base)         |  68.42        | 70.38      | 

#### KBP37 [[paper]](https://arxiv.org/pdf/1508.01006v2.pdf) [[data]](https://github.com/zhangdongxu/kbp37)
```
export KBP37_DATA_DIR=<KBP37_DATA_DIR>
export OUT_DIR=<$OUT_DIR>
python code/finetune_tasks/run_kbp37.py \
  --data_dir $KBP37_DATA_DIR \
  --gpu_ids 0,1 \
  --lm_model bert-base-uncased \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --max_seq_length 128 \
  --output_dir $OUT_DIR \
  --do_train \
  --save_model
```

|                   | Dev    | Test  | 
| ----------------------  | ------------- | ---------  | 
| BERT (base)             | 68.35         | 70.28      | 
| RoBERTa (base)         |  68.59         | 69.61      | 


#### SemEval-2010 Task 8 [[paper]](https://www.aclweb.org/anthology/S10-1006.pdf) [[data]](https://docs.google.com/leaf?id=0B_jQiLugGTAkMDQ5ZjZiMTUtMzQ1Yy00YWNmLWJlZDYtOWY1ZDMwY2U4YjFk&sort=name&layout=list&num=50)
```
export SemEval_DATA_DIR=<SemEval_DATA_DIR>
export OUT_DIR=<$OUT_DIR>
python code/finetune_tasks/run_semval.py \
  --data_dir $SemEval_DATA_DIR \
  --gpu_ids 0,1 \
  --lm_model bert-base-uncased \
  --batch_size 128 \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --max_seq_length 128 \
  --output_dir $OUT_DIR \
  --do_train \
  --save_model
```

|                   | Dev    | Test  | 
| ----------------------  | ------------- | ---------  | 
| BERT (base)             |          |       | 
| RoBERTa (base)         |           |       | 

## TODO

### Dialogue RE
- [ ] DialogRE

### Doc-level RE
- [ ] DocRED
- [ ] SciREX

### Distantly-supervised RE
- [ ] NYT-Freebase

## Contact
If you have any questions or suggestions, please contact me at `<fan.bai@cc.gatech.edu>` or create a Github issue.


## Reference
1. [Soares et al., 2019] Matching the Blanks: Distributional Similarity for Relation Learning
2. [Zhong and Chen, 2020] A Frustratingly Easy Approach for Joint Entity and Relation Extraction
