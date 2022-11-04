import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from loss import CE, Align, Reconstruct
from torch.optim.lr_scheduler import LambdaLR
from classification import fit_lr, get_rep_with_label


class Trainer():
    def __init__(self, args, model, train_loader, train_linear_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(torch.device(self.device))
        # self.model = model.cuda()
        print('model cuda')

        self.train_loader = train_loader
        self.train_linear_loader = train_linear_loader
        self.test_loader = test_loader
        self.lr_decay = args.lr_decay_rate
        self.lr_decay_steps = args.lr_decay_steps

        self.cr = CE(self.model)
        self.alpha = args.alpha
        self.beta = args.beta

        self.test_cr = torch.nn.CrossEntropyLoss()
        self.num_epoch = args.num_epoch
        self.num_epoch_pretrain = args.num_epoch_pretrain
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        if self.num_epoch:
            self.result_file = open(self.save_path + '/result.txt', 'w')
            self.result_file.close()

        self.step = 0
        self.best_metric = -1e9
        self.metric = 'acc'

    def pretrain(self):
        print('pretraining')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        eval_acc = 0
        align = Align()
        reconstruct = Reconstruct()
        self.model.copy_weight()
        if self.num_epoch_pretrain:
            result_file = open(self.save_path + '/pretrain_result.txt', 'w')
            result_file.close()
            result_file = open(self.save_path + '/linear_result.txt', 'w')
            result_file.close()
        for epoch in range(self.num_epoch_pretrain):
            self.model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            loss_sum = 0
            loss_mse = 0
            loss_ce = 0
            hits_sum = 0
            NDCG_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()
                [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = self.model.pretrain_forward(batch[0])
                align_loss = align.compute(rep_mask, rep_mask_prediction)
                loss_mse += align_loss.item()
                reconstruct_loss, hits, NDCG = reconstruct.compute(token_prediction_prob, tokens)
                loss_ce += reconstruct_loss.item()
                hits_sum += hits.item()
                NDCG_sum += NDCG
                loss = self.alpha * align_loss + self.beta * reconstruct_loss
                loss.backward()
                self.optimizer.step()
                self.model.momentum_update()
                loss_sum += loss.item()
            print('pretrain epoch{0}, loss{1}, mse{2}, ce{3}, hits{4}, ndcg{5}'.format(epoch + 1, loss_sum / (idx + 1),
                                                                                       loss_mse / (idx + 1),
                                                                                       loss_ce / (idx + 1), hits_sum,
                                                                                       NDCG_sum / (idx + 1)))
            result_file = open(self.save_path + '/pretrain_result.txt', 'a+')
            print('pretrain epoch{0}, loss{1}, mse{2}, ce{3}, hits{4}, ndcg{5}'.format(epoch + 1, loss_sum / (idx + 1),
                                                                                       loss_mse / (idx + 1),
                                                                                       loss_ce / (idx + 1), hits_sum,
                                                                                       NDCG_sum / (idx + 1)),
                  file=result_file)
            result_file.close()
            if (epoch + 1) % 5 == 0:
                self.model.eval()
                train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
                test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
                clf = fit_lr(train_rep, train_label)
                acc = clf.score(test_rep, test_label)
                print(acc)
                result_file = open(self.save_path + '/linear_result.txt', 'a+')
                print('epoch{0}, acc{1}'.format(epoch, acc), file=result_file)
                result_file.close()
                if acc > eval_acc:
                    eval_acc = acc
                    torch.save(self.model.state_dict(), self.save_path + '/pretrain_model.pkl')

    def finetune(self):
        print('finetune')
        if self.args.load_pretrained_model:
            print('load pretrained model')
            state_dict = torch.load(self.save_path + '/pretrain_model.pkl', map_location=self.device)
            try:
                self.model.load_state_dict(state_dict)
            except:
                model_state_dict = self.model.state_dict()
                for pretrain, random_intial in zip(state_dict, model_state_dict):
                    assert pretrain == random_intial
                    if pretrain in ['input_projection.weight', 'input_projection.bias', 'predict_head.weight',
                                    'predict_head.bias', 'position.pe.weight']:
                        state_dict[pretrain] = model_state_dict[pretrain]
                self.model.load_state_dict(state_dict)

        self.model.eval()
        train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
        test_rep, test_label = get_rep_with_label(self.model, self.test_loader)
        clf = fit_lr(train_rep, train_label)
        acc = clf.score(test_rep, test_label)
        pred_label = np.argmax(clf.predict_proba(test_rep), axis=1)
        f1 = f1_score(test_label, pred_label, average='macro')
        print(acc, f1)
        result_file = open(self.save_path + '/linear_result.txt', 'a+')
        print('epoch{0}, acc{1}, f1{2}'.format(0, acc, f1), file=result_file)
        result_file.close()

        self.model.linear_proba = False
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        for epoch in range(self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            self.print_process(
                'Finetune epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost))
            print('Finetune train epoch:{0},loss:{1},training_time:{2}'.format(epoch + 1, loss_epoch, time_cost),
                  file=self.result_file)
            self.result_file.close()
        self.print_process(self.best_metric)
        return self.best_metric

    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_linear_loader) if self.verbose else self.train_linear_loader

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.cr.compute(batch)
            loss_sum += loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()

            self.step += 1
        # if self.step % self.eval_per_steps == 0:
        metric = self.eval_model()
        self.print_process(metric)
        self.result_file = open(self.save_path + '/result.txt', 'a+')
        print('step{0}'.format(self.step), file=self.result_file)
        print(metric, file=self.result_file)
        self.result_file.close()
        if metric[self.metric] >= self.best_metric:
            torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
            self.result_file = open(self.save_path + '/result.txt', 'a+')
            print('saving model of step{0}'.format(self.step), file=self.result_file)
            self.result_file.close()
            self.best_metric = metric[self.metric]
        self.model.train()

        return loss_sum / (idx + 1), time.perf_counter() - t0

    def eval_model(self):
        self.model.eval()
        tqdm_data_loader = tqdm(self.test_loader) if self.verbose else self.test_loader
        metrics = {'acc': 0, 'f1': 0}
        pred = []
        label = []
        test_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                ret = self.compute_metrics(batch)
                if len(ret) == 2:
                    pred_b, label_b = ret
                    pred += pred_b
                    label += label_b
                else:
                    pred_b, label_b, test_loss_b = ret
                    pred += pred_b
                    label += label_b
                    test_loss += test_loss_b.cpu().item()
        confusion_mat = self._confusion_mat(label, pred)
        self.print_process(confusion_mat)
        self.result_file = open(self.save_path + '/result.txt', 'a+')
        print(confusion_mat, file=self.result_file)
        self.result_file.close()
        if self.args.num_class == 2:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred)
            metrics['precision'] = precision_score(y_true=label, y_pred=pred)
            metrics['recall'] = recall_score(y_true=label, y_pred=pred)
        else:
            metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
            metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
        metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
        metrics['test_loss'] = test_loss / (idx + 1)
        return metrics

    def compute_metrics(self, batch):
        if len(batch) == 2:
            seqs, label = batch
            scores = self.model(seqs)
        else:
            seqs1, seqs2, label = batch
            scores = self.model((seqs1, seqs2))
        _, pred = torch.topk(scores, 1)
        test_loss = self.test_cr(scores, label.view(-1).long())
        pred = pred.view(-1).tolist()
        return pred, label.tolist(), test_loss

    def _confusion_mat(self, label, pred):
        mat = np.zeros((self.args.num_class, self.args.num_class))
        for _label, _pred in zip(label, pred):
            mat[_label, _pred] += 1
        return mat

    def print_process(self, *x):
        if self.verbose:
            print(*x)
