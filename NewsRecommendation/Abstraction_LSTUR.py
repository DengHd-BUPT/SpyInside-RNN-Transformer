import sys
sys.path.append("..")
import Abstraction_RNN as FormalNN
import time
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from LSTUR.news_encoder import NewsEncoder
from LSTUR.user_encoder import UserEncoder
from LSTUR.mathutils import DotProductClickPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Model = getattr(importlib.import_module("Model"))

class LSTUR(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding=None):
        super(LSTUR, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()
        assert int(config.num_filters * 1.5) == config.num_filters * 1.5
        self.user_embedding = nn.Embedding(
            config.num_users,
            config.num_filters * 3 if config.long_short_term_method == 'ini'
            else int(config.num_filters * 1.5),
            padding_idx=0)

    def forward(self, user, clicked_news_length, candidate_news, clicked_news):
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news], dim=1)
        user = F.dropout2d(self.user_embedding(
            user.to(device)).unsqueeze(dim=0),
                           p=self.config.masking_probability,
                           training=self.training).squeeze(dim=0)
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        user_vector = self.user_encoder(user, clicked_news_length,
                                        clicked_news_vector)
        click_probability = self.click_predictor(candidate_news_vector,
                                                 user_vector)
        return click_probability

    def get_news_vector(self, news):
        return self.news_encoder(news)

    def get_user_vector(self, user, clicked_news_length, clicked_news_vector):
        user = self.user_embedding(user.to(device))
        return self.user_encoder(user, clicked_news_length,
                                 clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)


class LSTURDP(LSTUR):
    def __init__(self, *args, **kwargs):
        super(LSTURDP, self).__init__(*args, **kwargs)

    def set_bound_method(self, bound_method):
        self.bound_method = bound_method

    def get_user_vector(self, user, clicked_news_length, clicked_news_vector):
        user = self.user_embedding(user.to(device))
        # user_encoder DP model
        clicked_news_length[clicked_news_length == 0] = 1
        print("\nThe original size of clicked_news_vector is " + str(clicked_news_vector.size()))
        batchdata_split = {}
        batchsizes = []
        news_vector_padded = clicked_news_vector.transpose(0, 1)
        packed_clicked_news_vector = pack_padded_sequence(
            clicked_news_vector,
            clicked_news_length,
            batch_first=True,
            enforce_sorted=False)
        padded = 0
        if padded == 1:
            for p in range(news_vector_padded.size()[0]):
                batchdata_split[p] = news_vector_padded[p]
                batchsizes.append(news_vector_padded.size()[1])
        else:
            batchsizes = packed_clicked_news_vector.batch_sizes.tolist()
            # batchdata_split = {}
            splitdata = torch.split(packed_clicked_news_vector.data, batchsizes, 0)
            for i in range(len(batchsizes)):
                batchdata_split[i] = splitdata[i]
        sum = 0
        for i in range(len(batchsizes)):
            sum = sum + batchsizes[i]
        print("sum:" + str(sum))
        layers = []
        lstm_pack = {}
        for m in range(len(batchsizes)):
            lstm_pack[m] = []
        feed = []
        chain = []
        user_encoder_lb = torch.zeros(2048, 900).to(device)
        user_encoder_ub = torch.zeros(2048, 900).to(device)

        for k in range(len(batchsizes)): # len(batchsizes) = 50
            print("The batch timestamp is: " + str(k))
            print("The batch size is: " + str(batchsizes[k]))
            batchstart = time.time()
            for i in range(batchsizes[k]):
                print("The vector number in this batch is: " + str(i))
                vectorstart = time.time()
                dpframe = FormalNN.DeepPoly(
                    batchdata_split[k].data[i].to(device),
                    batchdata_split[k].data[i].to(device),
                    None,
                    None,
                )
                lin1 = FormalNN.Linear(self.config.num_filters * 3, self.config.num_filters * 3)
                lin1.assign(torch.eye(self.config.num_filters * 3), device=device)
                lin1_out = lin1(dpframe)
                vectorlin1end = time.time()

                lstm = FormalNN.LSTMCell.convert(
                    self.user_encoder.lstm,
                    prev_layer=lin1,
                    prev_cell=None if k == 0 else lstm_pack[k-1][i],
                    method=self.bound_method,
                    device=device,
                )
                lstm_pack[k].append(lstm)
                lstm_out = lstm(lin1_out)
                # chain.append(lstm)
                vectorlstmend = time.time()

                if k == 0:
                    if i == 0:
                        user_encoder_lb = lstm_out.lb.unsqueeze(dim=0)
                        user_encoder_ub = lstm_out.ub.unsqueeze(dim=0)
                    else:
                        user_encoder_lb = torch.cat((user_encoder_lb, lstm_out.lb.unsqueeze(dim=0)), dim=0)
                        user_encoder_ub = torch.cat((user_encoder_ub, lstm_out.ub.unsqueeze(dim=0)), dim=0)
                else:
                    user_encoder_lb[i] = lstm_out.lb
                    user_encoder_ub[i] = lstm_out.ub
                # layers.append(chain)
                vectorend = time.time()
                print("lin1_time: " + str(vectorlin1end - vectorstart))
                print("lstm_time: " + str(vectorlstmend - vectorlin1end))
                print("vectortime: " + str(vectorend - vectorstart))
            batchend = time.time()
            print("The batch timestamp is: " + str(k))
            print("batchtime: " + str(batchend - batchstart))

        lb = torch.cat((user_encoder_lb, user), dim=1)
        ub = torch.cat((user_encoder_ub, user), dim=1)
        user_encoder = FormalNN.DeepPoly(
            lb,
            ub,
            lexpr=None,
            uexpr=None,
        )
        return user_encoder