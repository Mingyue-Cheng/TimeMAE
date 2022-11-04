import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.utils.data as Data
from model.TimeMAE import TimeMAE
from args import args, Test_data
from dataset import Dataset
from sklearn.manifold import TSNE

test_dataset = Dataset(device=args.device, mode='test', data=Test_data, wave_len=args.wave_length)
test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
args.data_shape = test_dataset.shape()

model = TimeMAE(args)
# state_dict = torch.load('exp/har/test/pretrain_model.pkl', map_location='cpu')
# model.load_state_dict(state_dict)
model.linear_proba = True
model.eval()

reps = []
labels = []
with torch.no_grad():
    for idx, batch in enumerate(tqdm(test_loader)):
        seqs, label = batch
        label = label.numpy()
        rep_batch = model(seqs)
        for i in range(len(rep_batch)):
            reps.append(rep_batch[i].numpy())
            labels.append(label[i])

tsne = TSNE(n_components=2, random_state=4399)
rep_new = tsne.fit_transform(reps)
plt.scatter(rep_new[:, 0], rep_new[:, 1], c=labels, s=10)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(ls='--')
plt.savefig('pic/epilepsy/random.svg', format='svg')
plt.show()
