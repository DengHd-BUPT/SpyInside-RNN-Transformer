import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.config = config
        assert int(config.num_filters * 1.5) == config.num_filters * 1.5
        self.lstm = nn.LSTM(
            config.num_filters * 3,
            config.num_filters * 1.5)

    def forward(self, user, clicked_news_length, clicked_news_vector):
        clicked_news_length[clicked_news_length == 0] = 1
        packed_clicked_news_vector = pack_padded_sequence(
            clicked_news_vector,
            clicked_news_length,
            batch_first=True,
            enforce_sorted=False)
        batchsizes = packed_clicked_news_vector.batch_sizes.tolist()
        _, last_hidden = self.lstm(packed_clicked_news_vector)
        return torch.cat((last_hidden[0].squeeze(dim=0), user), dim=1)

