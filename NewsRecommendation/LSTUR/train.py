from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import time
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from evaluate import evaluate
import datetime
import importlib
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval

Model = getattr(importlib.import_module("LSTUR"))
config = getattr(importlib.import_module('config'))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseDataset(Dataset):
    def __init__(self, behaviors_path, news_path):
        super(BaseDataset, self).__init__()
        assert all(attribute in [
            'category', 'subcategory', 'title', 'abstract', 'title_entities',
            'abstract_entities'
        ] for attribute in config.dataset_attributes['news'])
        assert all(attribute in ['user', 'clicked_news_length']
                   for attribute in config.dataset_attributes['record'])

        self.behaviors_parsed = pd.read_table(behaviors_path)
        self.news_parsed = pd.read_table(
            news_path,
            index_col='id',
            usecols=['id'] + config.dataset_attributes['news'],
            converters={
                attribute: literal_eval
                for attribute in set(config.dataset_attributes['news']) & set([
                    'title', 'abstract', 'title_entities', 'abstract_entities'
                ])
            })
        self.news_id2int = {x: i for i, x in enumerate(self.news_parsed.index)}
        self.news2dict = self.news_parsed.to_dict('index')
        for key1 in self.news2dict.keys():
            for key2 in self.news2dict[key1].keys():
                self.news2dict[key1][key2] = torch.tensor(
                    self.news2dict[key1][key2])
        padding_all = {
            'category': 0,
            'subcategory': 0,
            'title': [0] * config.num_words_title,
            'abstract': [0] * config.num_words_abstract,
            'title_entities': [0] * config.num_words_title,
            'abstract_entities': [0] * config.num_words_abstract
        }
        for key in padding_all.keys():
            padding_all[key] = torch.tensor(padding_all[key])

        self.padding = {
            k: v
            for k, v in padding_all.items()
            if k in config.dataset_attributes['news']
        }

    def __len__(self):
        return len(self.behaviors_parsed)

    def __getitem__(self, idx):
        item = {}
        row = self.behaviors_parsed.iloc[idx]
        if 'user' in config.dataset_attributes['record']:
            item['user'] = row.user
        item["clicked"] = list(map(int, row.clicked.split()))
        item["candidate_news"] = [
            self.news2dict[x] for x in row.candidate_news.split()
        ]
        item["clicked_news"] = [
            self.news2dict[x]
            for x in row.clicked_news.split()[:config.num_clicked_news_a_user]
        ]
        if 'clicked_news_length' in config.dataset_attributes['record']:
            item['clicked_news_length'] = len(item["clicked_news"])
        repeated_times = config.num_clicked_news_a_user - \
            len(item["clicked_news"])
        assert repeated_times >= 0
        item["clicked_news"] = [self.padding
                                ] * repeated_times + item["clicked_news"]
        return item

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better


def train():
    writer = SummaryWriter(
        log_dir=
        f"./runs/{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    )

    if not os.path.exists('Model'):
        os.makedirs('Model')
    pretrained_word_embedding = torch.from_numpy(
        np.load('./data/train/pretrained_word_embedding.npy')).float()

    model = Model(config, pretrained_word_embedding).to(device)
    print(model)

    dataset = BaseDataset('/data/train/behaviors_parsed.tsv',
                          '/data/train/news_parsed.tsv')
    print(f"Load training dataset with size {len(dataset)}.")
    dataloader = iter(
        DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True,
                   num_workers=config.num_workers,
                   drop_last=True,
                   pin_memory=True))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)

    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    step = 0
    early_stopping = EarlyStopping()
    checkpoint_dir = os.path.join('./Model')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = latest_checkpoint(checkpoint_dir)
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        early_stopping(checkpoint['early_stop_value'])
        step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()
    for i in tqdm(range(
            1,
            config.num_epochs * len(dataset) // config.batch_size + 1),
            desc="Training"):
        try:
            minibatch = next(dataloader)
        except StopIteration:
            exhaustion_count += 1
            tqdm.write(
                f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
            )
            dataloader = iter(
                DataLoader(dataset,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers,
                           drop_last=True,
                           pin_memory=True))
            minibatch = next(dataloader)

        step += 1
        y_pred = model(minibatch["user"], minibatch["clicked_news_length"],
                       minibatch["candidate_news"],
                       minibatch["clicked_news"])
        y = torch.zeros(len(y_pred)).long().to(device)
        loss = criterion(y_pred, y)
        loss_full.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            writer.add_scalar('Train/Loss', loss.item(), step)
        if i % config.num_batches_show_loss == 0:
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
            )
        if i % config.num_batches_validate == 0:
            model.eval()
            val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
                model, './data/val',
                config.num_workers, 200000)
            model.train()
            writer.add_scalar('Validation/AUC', val_auc, step)
            writer.add_scalar('Validation/MRR', val_mrr, step)
            writer.add_scalar('Validation/nDCG@5', val_ndcg5, step)
            writer.add_scalar('Validation/nDCG@10', val_ndcg10, step)
            tqdm.write(
                f"Time {time_since(start_time)}, batches {i}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
            )
            early_stop, get_better = early_stopping(-val_auc)
            if early_stop:
                tqdm.write('Early stop.')
                break
            elif get_better:
                try:
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict':
                                optimizer.state_dict(),
                            'step':
                                step,
                            'early_stop_value':
                                -val_auc
                        }, f"./Model/ckpt-{step}.pth")
                except OSError as error:
                    print(f"OS error: {error}")
def time_since(since):
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

if __name__ == '__main__':
    print('Using device:', device)
    print("Training model")
    train()