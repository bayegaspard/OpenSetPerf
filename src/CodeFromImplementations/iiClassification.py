import torch
#https://arxiv.org/pdf/1802.04365.pdf
#I could not find pytorch code for this so I am implementing it myself.


def intra_Spread(logits:torch.Tensor):
    C = []
    for column in logits[:]:
        C.append(column.mean())
    