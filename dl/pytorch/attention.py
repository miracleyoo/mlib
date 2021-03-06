import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """ Pytorch self-attention layer code inspired from:
    Link:
        https://discuss.pytorch.org/t/self-attention-on-words-and-masking/5671/4
    Original Web Page:
        https://www.kaggle.com/dannykliu/lstm-with-attention-clr-in-pytorch
    Usage:
        In __init__():
            self.atten1 = Attention(hidden_dim*2, batch_first=True) # 2 is bidrectional
        In forward():
            x, _ = self.atten1(x, lengths)
    """

    def __init__(self, hidden_size, batch_first=True, device=None):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.batch_first = batch_first
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.att_weights = nn.Parameter(
            torch.Tensor(1, hidden_size), requires_grad=True)

        stdv = 1.0 / np.sqrt(self.hidden_size)
        for weight in self.att_weights:
            nn.init.uniform_(weight, -stdv, stdv)

    def get_mask(self):
        pass

    def forward(self, inputs, lengths):
        if self.batch_first:
            batch_size, max_len = inputs.size()[:2]
        else:
            max_len, batch_size = inputs.size()[:2]

        # apply attention layer
        weights = torch.bmm(inputs,
                            self.att_weights  # (1, hidden_size)
                            .permute(1, 0)  # (hidden_size, 1)
                            .unsqueeze(0)  # (1, hidden_size, 1)
                            # (batch_size, hidden_size, 1)
                            .repeat(batch_size, 1, 1)
                            )

        attentions = torch.softmax(F.relu(weights.squeeze()), dim=-1)

        # create mask based on the sentence lengths
        mask = torch.ones(attentions.size(), requires_grad=True).to(self.device)
        for i, l in enumerate(lengths):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0

        # apply mask and renormalize attention scores (weights)
        masked = attentions * mask
        _sums = masked.sum(-1).unsqueeze(-1)  # sums per row

        attentions = masked.div(_sums)

        # apply attention weights
        weighted = torch.mul(
            inputs, attentions.unsqueeze(-1).expand_as(inputs))

        # get the final fixed vector representations of the sentences
        representations = weighted.sum(1).squeeze()

        return representations, attentions
