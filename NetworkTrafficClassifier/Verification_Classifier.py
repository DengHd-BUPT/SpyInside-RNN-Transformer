import sys
sys.path.append("..")
import argparse
import os
import sys
import time
import re
import pandas as pd
import csv
from sklearn.preprocessing import LabelEncoder
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import NeptuneLogger
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from Abstraction_Classifier import GPT2ClassifierDP
from Transformer4NetworkTraffic.gpt_model.classifier.model import GPT2Classifier
from Classifier.dataset import ClassificationQuantizedDataset
from Classifier.tokenizer import PacketTokenizer
from Classifier.settings import BASE_DIR, DEFAULT_PACKET_LIMIT_PER_FLOW, NEPTUNE_PROJECT
import Abstraction_Transformer as FormalNN

def main():
    start = time.perf_counter()
    parser = argparse.ArgumentParser()
    test_traffic = '/traffic-classifier/datasets/test_class.csv'
    parser.add_argument(
        '--train_dataset',
        #default=test_traffic,
        default='/traffic-classifier/datasets/train_4c93174d7808b1487aa3288084365d76_no_mawi_unswnb_iscxvpn.csv',
        help='path to preprocessed .csv dataset',
    )
    # test_traffic = '/home/dhd/DRLVerification/Transformer4NetworkTraffic/gpt_model/classifier/test2.csv'
    testpath = '/traffic-classifier/datasets/test_4c93174d7808b1487aa3288084365d76_no_mawi_unswnb_iscxvpn.csv'

    parser.add_argument(
        '--test_dataset',
        default=test_traffic,
        help='path to preprocessed .csv dataset',
    )
    parser.add_argument(
        '--test_dataset2',
        default='/traffic-classifier/datasets/test_class_dp.csv',
        help='path to preprocessed .csv dataset',
    )
    parser.add_argument(
        '--pretrained_path',
        default='/traffic-classifier/gpt-model/generator/checkpoints/gpt2_model_4epochs_classes_external',
    )
    parser.add_argument(
        '--freeze_pretrained_model',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--mask_first_token',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--batch_size',
        default=1,
    )
    parser.add_argument(
        '--es_patience',
        default=5,
        type=int,
    )
    parser.add_argument(
        '--learning_rate',
        default=None
    )
    parser.add_argument(
        '--fc_dropout',
        default=0.1,
    )
    parser.add_argument(
        '--reinitialize',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--n_layers',
        default=6,
        type=int,
        help='number of transformer layers to use, only in use when --reinitialize is provided'
    )
    parser.add_argument(
        '--log_neptune',
        dest='log_neptune',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--neptune_experiment_name',
        dest='neptune_experiment_name',
        default='gpt2_class_pretrained'
    )

    args = parser.parse_args()
    if args.learning_rate is None:
        args.learning_rate = 0.0005 if args.freeze_pretrained_model else 0.00002

    print(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = PacketTokenizer.from_pretrained(args.pretrained_path, flow_size=DEFAULT_PACKET_LIMIT_PER_FLOW)

    train_val_dataset = ClassificationQuantizedDataset(tokenizer,
                                                       dataset_path=args.train_dataset)
    print("train_val_dataset.target_encoder:")
    print(train_val_dataset.target_encoder)
    train_part_len = int(len(train_val_dataset) * 0.9)
    train_dataset, val_dataset = random_split(train_val_dataset,
                                              [train_part_len, len(train_val_dataset) - train_part_len])

    test_dataset = ClassificationQuantizedDataset(tokenizer,
                                                  dataset_path=args.test_dataset,
                                                  label_encoder=train_val_dataset.target_encoder)
    test_dataset2 = ClassificationQuantizedDataset(tokenizer,
                                                  dataset_path=args.test_dataset2,
                                                  label_encoder=train_val_dataset.target_encoder)
#
    collator = ClassificationQuantizedDataset.get_collator(mask_first_token=args.mask_first_token)

    cpu_counter = os.cpu_count()
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  drop_last=False,
                                  shuffle=False,
                                  collate_fn=collator,
                                  num_workers=cpu_counter)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                drop_last=False,
                                shuffle=False,
                                collate_fn=collator,
                                num_workers=cpu_counter
                                )

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 drop_last=False,
                                 collate_fn=collator,
                                 num_workers=cpu_counter)

    class_labels = train_val_dataset.target_encoder.classes_
    nn_classifierdp = GPT2ClassifierDP()
    model_path = '/Model/GPT2Classifier_checkpoints/epoch=11-val_loss=0.04-other_metric=0.00.ckpt'
    j = 0
    k = 0

    for i in range(5,25):
        labled = test_dataset[i]
        unlabled = labled.pop('target')
        rlt1 = re.findall("\((.*?)\)",str(unlabled))
        labled2 = test_dataset2[i]
        unlabled2 = labled2.pop('target')
        testdata = FormalNN.DeepPoly(
            lb=labled['input_ids'].to(device),
            ub=labled2['input_ids'].to(device),
            lexpr=None,
            uexpr=None,
        )
        result1 = nn_classifierdp.certify(testdata, labled['attention_mask'])
        result_l = result1.lb.max(axis=1)[1]
        result_u = result1.ub.max(axis=1)[1]
        result = str(result_l) + str(result_u)
        rlt = re.findall("\[(.*?)\]", result)
        print(rlt1, rlt)
        if rlt1[0] == rlt[0]:
            j = j+1
        if rlt1[0] == rlt[1]:
            k = k+1

    end = time.perf_counter()
    print("Running Time", end-start)
    print(j, k)

if __name__ == '__main__':
    main()
