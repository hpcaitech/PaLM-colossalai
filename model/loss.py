import colossalai
from torch import nn


class PaLMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = colossalai.nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
