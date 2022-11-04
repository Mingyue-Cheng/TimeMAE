import torch
import torch.nn as nn


class CE:
    def __init__(self, model):
        self.model = model
        self.ce = nn.CrossEntropyLoss()
        self.ce_pretrain = nn.CrossEntropyLoss(ignore_index=0)

    def compute(self, batch):
        seqs, labels = batch
        outputs = self.model(seqs)  # B * N
        labels = labels.view(-1).long()
        loss = self.ce(outputs, labels)
        return loss


class Align:
    def __init__(self):
        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss()

    def compute(self, rep_mask, rep_mask_prediction):
        align_loss = self.mse(rep_mask, rep_mask_prediction)
        return align_loss


class Reconstruct:
    def __init__(self):
        self.ce = nn.CrossEntropyLoss(label_smoothing=0.2)

    def compute(self, token_prediction_prob, tokens):
        hits = torch.sum(torch.argmax(token_prediction_prob, dim=-1) == tokens)
        NDCG10 = recalls_and_ndcgs_for_ks(token_prediction_prob.view(-1, token_prediction_prob.shape[-1]),
                                          tokens.reshape(-1, 1), 10)
        reconstruct_loss = self.ce(token_prediction_prob.view(-1, token_prediction_prob.shape[-1]), tokens.view(-1))
        return reconstruct_loss, hits, NDCG10


def recalls_and_ndcgs_for_ks(scores, answers, k):
    answers = answers.tolist()
    labels = torch.zeros_like(scores).to(scores.device)
    for i in range(len(answers)):
        labels[i][answers[i]] = 1
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    cut = cut[:, :k]
    hits = labels_float.gather(1, cut)
    position = torch.arange(2, 2 + k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits * weights.to(hits.device)).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
    ndcg = (dcg / idcg).mean()
    ndcg = ndcg.cpu().item()
    return ndcg
