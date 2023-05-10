#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Script modified by Robert Litschko
# This script is based on:
# - https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py
# - https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/token-classification/run_ner.py

import logging
import os
import random
import multiprocessing
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import torch
from datasets import load_dataset, load_metric

import transformers
import transformers.adapters.composition as ac
from transformers import (
  AutoConfig,
  AutoModelForSequenceClassification,
  AutoTokenizer,
  DataCollatorWithPadding,
  EvalPrediction,
  HfArgumentParser,
  Trainer,
  TrainingArguments,
  default_data_collator,
  set_seed,
  MultiLingAdapterArguments
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers import AdamW

# from xlmr_colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples
# from xlmr_colbert.modeling.colbert import ColBERT

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
  """
  Arguments pertaining to what data we are going to input our model for training and eval.

  Using `HfArgumentParser` we can turn this class
  into argparse arguments to be able to specify them on
  the command line.
  """
  train_file: Optional[str] = field(
    default=None,
    metadata={"help": "An optional input train data file (tsv file)."},
  )
  validation_file: Optional[str] = field(
    default=None,
    metadata={"help": "An optional input validation data file (tsv file)."},
  )
  test_file: Optional[str] = field(
    default=None,
    metadata={"help": "An optional input test data (tsv file)."},
  )
  max_seq_length: Optional[int] = field(
    default=128,
    metadata={
      "help": "The maximum total input sequence length after tokenization. Sequences longer "
              "than this will be truncated, sequences shorter will be padded."
    },
  )
  overwrite_cache: bool = field(
    default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
  )
  pad_to_max_length: bool = field(
    default=True,
    metadata={
      "help": "Whether to pad all samples to `max_seq_length`. "
              "If False, will pad the samples dynamically when batching to the maximum length in the batch."
    },
  )
  max_train_samples: Optional[int] = field(
    default=None,
    metadata={
      "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
              "value if set."
    },
  )
  max_eval_samples: Optional[int] = field(
    default=None,
    metadata={
      "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
              "value if set."
    },
  )
  max_predict_samples: Optional[int] = field(
    default=None,
    metadata={
      "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
              "value if set."
    },
  )
  server_ip: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})
  server_port: Optional[str] = field(default=None, metadata={"help": "For distant debugging."})


@dataclass
class ModelArguments:
  """
  Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
  """

  model_name_or_path: str = field(
    default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
  )
  # language: str = field(
  #     default=None, metadata={"help": "Evaluation language. Also train language if `train_language` is set to None."}
  # )
  # train_language: Optional[str] = field(
  #     default=None, metadata={"help": "Train language if it is different from the evaluation language."}
  # )
  config_name: Optional[str] = field(
    default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
  )
  tokenizer_name: Optional[str] = field(
    default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
  )
  cache_dir: Optional[str] = field(
    default=None,
    metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
  )
  do_lower_case: Optional[bool] = field(
    default=False,
    metadata={"help": "arg to indicate if tokenizer should do lower case in AutoTokenizer.from_pretrained()"},
  )
  use_fast_tokenizer: bool = field(
    default=True,
    metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
  )
  model_revision: str = field(
    default="main",
    metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
  )
  use_auth_token: bool = field(
    default=False,
    metadata={
      "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
              "with private models)."
    },
  )

@dataclass
class MiscArguments:

  gpu: str = field(
    default=None,
    metadata={
      "help": "CUDA_VISIBLE_DEVICES"
    }
  )

  model_str: str = field(
    default=None,
    metadata={
      "help": "['colbert', 'monobert']"
    }
  )


def main():
  os.environ['CUDA_VISIBLE_DEVICES'] = os.environ[input("Specify GPU: ")]
  input("Continue?")

  # See all possible arguments in src/transformers/training_args.py
  # or by passing the --help flag to this script.
  # We now keep distinct sets of args, for a cleaner separation of concerns.
  parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, MiscArguments))
  model_args, data_args, training_args, misc_args = parser.parse_args_into_dataclasses()

  # Setup logging
  os.makedirs(training_args.output_dir, exist_ok=True)
  logfile = os.path.join(training_args.output_dir, "log.txt")
  filehandler = logging.FileHandler(filename=logfile)
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), filehandler]
  )

  log_level = training_args.get_process_log_level()
  logger.setLevel(log_level)
  datasets.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.add_handler(filehandler)
  transformers.utils.logging.set_verbosity(log_level)
  transformers.utils.logging.enable_default_handler()
  transformers.utils.logging.enable_explicit_format()

  # Log on each process the small summary:
  logger.warning(
    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
  )
  logger.info(f"Training/evaluation parameters {training_args}")

  # Detecting last checkpoint.
  last_checkpoint = None
  if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
      raise ValueError(
        f"Output directory ({training_args.output_dir}) already exists and is not empty. "
        "Use --overwrite_output_dir to overcome."
      )
    elif last_checkpoint is not None:
      logger.info(
        f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
      )

  # Set seed before initializing model.
  set_seed(training_args.seed)

  # In distributed training, the load_dataset function guarantees that only one local process can concurrently
  # download the dataset.
  # Downloading and loading prepared ms-marco dataset from disk.
  if training_args.do_train:
    if misc_args.model_str == "colbert":
      train_dataset = load_dataset("text", data_files=data_args.train_file, cache_dir=model_args.cache_dir)["train"]
    else:
      assert misc_args.model_str == "monobert"
      train_dataset = load_dataset("json", data_files=data_args.train_file, cache_dir=model_args.cache_dir)['train']

  if training_args.do_eval:
    if misc_args.model_str == "colbert":
      eval_dataset = load_dataset("text", data_files=data_args.validation_file, cache_dir=model_args.cache_dir)["train"]
    else:
      assert misc_args.model_str == "monobert"
      eval_dataset = load_dataset("json", data_files=data_args.validation_file, cache_dir=model_args.cache_dir)['train']

  if misc_args.model_str == "monobert":
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    # Labels
    num_labels = 1
    max_length = data_args.max_seq_length  # 512

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = num_labels
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config, cache_dir=model_args.cache_dir)
    # Preprocessing the datasets
    # Padding strategy
    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    data_args.pad_to_max_length = False
    max_length = min(tokenizer.max_len_sentences_pair, data_args.max_seq_length)

    # Taken from Reimers et al. (CrossEncoder.py), num_train_steps = 625000, batch_size = 32
    num_train_steps = len(train_dataset) // training_args.train_batch_size
    weight_decay = 0.01
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
      {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, **{'lr': training_args.learning_rate})
    scheduler = transformers.get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=training_args.warmup_steps,
      num_training_steps=num_train_steps
    )
    # this hack is necessary to trigger BCEWithLogitsLoss() internally, 
    # otherwise scalar outputs are treated as regression and evaluated with MSE
    model.config.problem_type = "multi_label_classification"
    model = model.to('cuda')

    def data_collator(examples):
      questions = [e['query'].strip() for e in examples]
      passages = [e['passage'].strip() for e in examples]
      tokenized_examples = tokenizer(
        *[questions, passages], padding=True, return_tensors="pt", truncation='longest_first', max_length=max_length
      )
      tokenized_examples['labels'] = torch.tensor([e['label'] for e in examples],
                                                  dtype=torch.float if config.num_labels == 1 else torch.long)
      tokenized_examples['labels'] = torch.unsqueeze(tokenized_examples['labels'], 1)
      return tokenized_examples  # input_ids, token_type_ids, attention_mask, labels
  else:
    raise NotImplemented
    # assert misc_args.model_str == "colbert"
    # n_q_tokens = 32
    # n_d_tokens = 180
    # query_tokenizer = QueryTokenizer(n_q_tokens)
    # doc_tokenizer = DocTokenizer(n_d_tokens)
    #
    # model = ColBERT.from_pretrained(
    #   "xlm-roberta-large",
    #   query_maxlen=n_q_tokens,
    #   doc_maxlen=n_d_tokens,
    #   dim=128,
    #   similarity_metric="l2",
    #   mask_punctuation=False
    # )
    #
    # model.roberta.resize_token_embeddings(len(model.tokenizer))
    # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=training_args.learning_rate, eps=1e-8)
    # scheduler = None
    #
    # def data_collator(examples):
    #   examples = [line["text"].split("\t") for line in examples]
    #   queries, positives, negatives = list(zip(*examples))
    #   # positives = [e['possitive'] for e in examples]
    #   # negatives = [e['negatives'] for e in examples]
    #   assert len(queries) == len(positives) == len(negatives)
    #   N = len(queries)
    #   Q_ids, Q_mask = query_tokenizer.tensorize(queries)
    #   D_ids, D_mask = doc_tokenizer.tensorize(positives + negatives)
    #   D_ids, D_mask = D_ids.view(2, N, -1), D_mask.view(2, N, -1)
    #
    #   # Compute max among {length of i^th positive, length of i^th negative} for i \in N
    #   maxlens = D_mask.sum(-1).max(0).values
    #
    #   # Sort by maxlens
    #   indices = maxlens.sort().indices
    #   Q_ids, Q_mask = Q_ids[indices], Q_mask[indices]
    #   D_ids, D_mask = D_ids[:, indices], D_mask[:, indices]
    #
    #   (positive_ids, negative_ids), (positive_mask, negative_mask) = D_ids, D_mask
    #   Q = (torch.cat((Q_ids, Q_ids)), torch.cat((Q_mask, Q_mask)))
    #   D = (torch.cat((positive_ids, negative_ids)), torch.cat((positive_mask, negative_mask)))
    #   # bsize = len(examples)
    #   bsize = D[0].shape[0]
    #   neg_labels = torch.zeros(bsize // 2, dtype=torch.long)
    #   pos_labels = torch.ones(bsize // 2, dtype=torch.long)
    #   labels = torch.cat((neg_labels, pos_labels))
    #   tokenized_examples = {
    #     'labels': labels,
    #     'Q': Q,
    #     'D': D
    #   }
    #   return tokenized_examples

  preprocessing_num_workers = multiprocessing.cpu_count() // 2
  data_args.preprocessing_num_workers = preprocessing_num_workers

  if training_args.do_train:
    if data_args.max_train_samples is not None:
      train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
      logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

  if training_args.do_eval:
    if data_args.max_eval_samples is not None:
      eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

  # needed because our data_collator needs access to those columns
  training_args.remove_unused_columns = False

  model = model.to('cuda')
  print("moved model to GPU")

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    data_collator=data_collator,
    tokenizer=tokenizer,
    optimizers=(optimizer, scheduler)
  )

  # Training
  if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
      checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
      checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = (
      data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


  # Evaluation
  if training_args.do_eval:
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)



if __name__ == "__main__":
  main()
