import torch
from args import args
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def get_rep_with_label(model, dataloader):
    reps = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            seq, label = batch
            seq = seq.to(args.device)
            labels += label.cpu().numpy().tolist()
            rep = model(seq)
            reps += rep.cpu().numpy().tolist()
    return reps, labels


def fit_lr(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=3407,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe
