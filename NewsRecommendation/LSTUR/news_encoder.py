import torch
import torch.nn as nn
import torch.nn.functional as F
from mathutils import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding):
        super(NewsEncoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        self.category_embedding = nn.Embedding(config.num_categories,
                                               config.num_filters,
                                               padding_idx=0)
        assert config.window_size >= 1 and config.window_size % 2 == 1
        self.title_CNN = nn.Conv2d(
            1,
            config.num_filters,
            (config.window_size, config.word_embedding_dim),
            padding=(int((config.window_size - 1) / 2), 0))
        self.title_attention = AdditiveAttention(config.query_vector_dim,
                                                 config.num_filters)

    def forward(self, news):
        # calculate category_vector
        category_vector = self.category_embedding(news['category'].to(device))
        # calculate subcategory_vector
        subcategory_vector = self.category_embedding(
            news['subcategory'].to(device))
        # calculate weighted_title_vector
        title_vector = F.dropout(self.word_embedding(news['title'].to(device)),
                                 p=self.config.dropout_probability,
                                 training=self.training)
        convoluted_title_vector = self.title_CNN(
            title_vector.unsqueeze(dim=1)).squeeze(dim=3)
        activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                           p=self.config.dropout_probability,
                                           training=self.training)
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2))
        news_vector = torch.cat(
            [category_vector, subcategory_vector, weighted_title_vector],
            dim=1)
        return news_vector
