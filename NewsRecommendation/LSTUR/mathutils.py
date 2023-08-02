import torch
import torch.nn as nn
import torch.nn.functional as F

class DotProductClickPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductClickPredictor, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        probability = torch.bmm(candidate_news_vector,
                                user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        return probability

class AdditiveAttention(torch.nn.Module):
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim,
                 writer=None,
                 tag=None,
                 names=None):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
        self.writer = writer
        self.tag = tag
        self.names = names
        self.local_step = 1

    def forward(self, candidate_vector):
        temp = torch.tanh(self.linear(candidate_vector))
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
                                      dim=1)
        if self.writer is not None:
            assert candidate_weights.size(1) == len(self.names)
            if self.local_step % 10 == 0:
                self.writer.add_scalars(
                    self.tag, {
                        x: y
                        for x, y in zip(self.names,
                                        candidate_weights.mean(dim=0))
                    }, self.local_step)
            self.local_step += 1
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target
