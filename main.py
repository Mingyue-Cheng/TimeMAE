import torch
import warnings

warnings.filterwarnings('ignore')
from args import args, Test_data, Train_data_all, Train_data
from dataset import Dataset
from model.TimeMAE import TimeMAE
from process import Trainer
import torch.utils.data as Data


def main():
    torch.set_num_threads(12)
    torch.cuda.manual_seed(3407)
    train_dataset = Dataset(device=args.device, mode='pretrain', data=Train_data_all, wave_len=args.wave_length)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    args.data_shape = train_dataset.shape()
    train_linear_dataset = Dataset(device=args.device, mode='supervise_train', data=Train_data, wave_len=args.wave_length)
    train_linear_loader = Data.DataLoader(train_linear_dataset, batch_size=args.train_batch_size, shuffle=True)
    test_dataset = Dataset(device=args.device, mode='test', data=Test_data, wave_len=args.wave_length)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)

    print(args.data_shape)
    print('dataset initial ends')

    model = TimeMAE(args)

    print('model initial ends')
    trainer = Trainer(args, model, train_loader, train_linear_loader, test_loader, verbose=True)

    trainer.pretrain()
    trainer.finetune()


if __name__ == '__main__':
    main()
