import sys
sys.path.append("..")
import logging
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import GPT2Model, GPT2Config
from transformers.optimization import AdamW

from Classifier.BaseClassifier import BaseClassifier
import Abstraction_Transformer as FormalNN

logger = logging.getLogger(__name__)
class GPT2Classifier(BaseClassifier):
    def __init__(
            self,
            config,
            class_labels,
            pretrained_model_path,
            dropout=0.1,
            freeze_pretrained_part=True,
            reinitialize=False,
            n_layers=6,
    ):
        super().__init__(config, class_labels)

        if reinitialize:
            logger.info('resetting model weights')
            config = GPT2Config.from_json_file(pretrained_model_path + '/config.json')
            config = config.to_dict()
            config['n_layer'] = n_layers
            config = GPT2Config.from_dict(config)
            self.gpt2 = GPT2Model(config)
        else:
            self.gpt2 = GPT2Model.from_pretrained(pretrained_model_path)

        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(self.gpt2.config.n_embd, self.output_dim)
        if freeze_pretrained_part:
            for param in self.gpt2.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = self.gpt2(**x)
        output = output[0]  # last hidden state (batch_size, sequence_length, hidden_size)
        # average over temporal dimension
        output = output.mean(dim=1)
        output = self.dropout(output)
        return self.fc(output)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams.es_patience // 2)
        return [optimizer], [scheduler]


class GPT2ClassifierDP():
    def __init__(self):
        super(GPT2ClassifierDP, self).__init__()
        self.dropout = torch.nn.Dropout(0.1)
        #self.config = config

    def set_bound_method(self, bound_method):
        self.bound_method = bound_method
#
    #def certify(self, input_ids, attention_mask, gt, verbose=False):
    def certify(self, input_ids, attention_mask):
        model_path = '/Model/GPT2Classifier_checkpoints/epoch=11-val_loss=0.04-other_metric=0.00.ckpt'#
        checkpoint = torch.load(model_path)
        layers = []
        dev = input_ids.device

        gpt2 = FormalNN.GPT2modeldp(prev_layer=None, pos_ids=None ,config_resid_pdrop=0.1, attention_mask=attention_mask)
        gpt2_out = gpt2(input_ids)
        gpt2_out.lb = self.dropout(gpt2_out.lb.mean(dim=1))
        gpt2_out.ub = self.dropout(gpt2_out.ub.mean(dim=1))

        #fc = FormalNN.Linear(512, 32, prev_layer=gpt2.ln_f)
        fc = FormalNN.Linear(512, 32, prev_layer=None)
        fc.assign(checkpoint["state_dict"]['fc.weight'], checkpoint["state_dict"]['fc.bias'], device=dev)
        #fc = FormalNN.Linear.convert(self.fc, prev_layer=gpt2.ln_f, device=dev)
        fc_out = fc(gpt2_out)

        return fc_out

