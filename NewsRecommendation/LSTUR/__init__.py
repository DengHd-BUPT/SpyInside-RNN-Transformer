import torch
import torch.nn as nn
import torch.nn.functional as F
from news_encoder import NewsEncoder
from user_encoder import UserEncoder
from mathutils import DotProductClickPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        user_vector = self.user_encoder(user, clicked_news_length, clicked_news_vector)

        click_probability = self.click_predictor(candidate_news_vector, user_vector)
        return click_probability

    def get_news_vector(self, news):
        return self.news_encoder(news)

    def get_user_vector(self, user, clicked_news_length, clicked_news_vector):
        user = self.user_embedding(user.to(device))
        return self.user_encoder(user, clicked_news_length, clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
