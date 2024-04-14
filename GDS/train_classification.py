""" Matplotlib backend configuration """
import matplotlib
matplotlib.use('PS')  # generate postscript output by default

""" Imports """
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve, precision_recall_curve, auc
from preprocess_GDS import load_dataset_classification
from GDS.model_MTL_trans_classification import TMTL



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ALZDataset(Dataset):

    def __init__(self, patients, seqs_list, labels_list):
        """
        Args:m
            df_dataset(DataFrame)
            df_label(DataFrame)
            data_idx_list(list): it shows where to devide the features for  feature classification
            seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
            labels (list): list of labels (int)
        """


        self.patients = patients



        if  len(seqs_list) != len(labels_list):
            raise ValueError("inputs and Labels have different lengths")

        self.seqs = []
        self.labels = []
        for seq, label in zip(seqs_list, labels_list):
            self.seqs.append(np.array(seq, dtype=np.float32))
            self.labels.append(np.expand_dims(label, 0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.patients[index], self.seqs[index], self.labels[index]



def visit_collate_fn(batch):
    """
    DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
    Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
    where N is minibatch size, seq is a SparseFloatTensor, and label is a LongTensor

    :returns
        seqs
        labels
        lengths
    """
    batch_patient, batch_seq, batch_label = zip(*batch)

    num_features = batch_seq[0].shape[1]
    seq_lengths = list(map(lambda patient_tensor: patient_tensor.shape[0], batch_seq))
    max_length = max(seq_lengths)

    sorted_indices, sorted_lengths = zip(*sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True))
    sorted_padded_seqs = []
    sorted_labels = []
    sorted_patients = []

    for i in sorted_indices:
        length = batch_seq[i].shape[0]

        if length < max_length:
            padded = np.concatenate(
                (batch_seq[i], np.zeros((max_length - length, num_features), dtype=np.float32)), axis=0)
        else:
            padded = batch_seq[i]

        sorted_padded_seqs.append(padded)
        sorted_labels.append(batch_label[i].flatten())
        sorted_patients.append(batch_patient[i])

    seq_tensor = np.stack(sorted_padded_seqs, axis=0)

    return sorted_patients, torch.FloatTensor(np.array(seq_tensor)), torch.FloatTensor(np.array(sorted_labels)), list(sorted_lengths), list(sorted_indices) #torch.from_numpy(seq_tensor)


class Model_trainer:
    def __init__(self, params, class_weights):

        self.params = params
        self.class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        if self.params['cuda']:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'

        #self.criterion = nn.CrossEntropyLoss(weight=self.class_weights_tensor)
        self.criterion = nn.CrossEntropyLoss()

        if self.params['cuda']:
            self.criterion = self.criterion.cuda()

    def _data_loading(self, train_p, train_seqs, train_labels, test_p, test_seqs, test_labels):

        print('===> Loading entire datasets')
        print("     ===> Construct train set")
        train_set = ALZDataset(train_p, train_seqs, train_labels)
        print("     ===> Construct test set")
        test_set = ALZDataset(test_p, test_seqs, test_labels)
        #att_set = ALZDataset(train_seqs, train_labels)

        self.train_loader = DataLoader(dataset=train_set, batch_size=self.params['batch_size'], shuffle=True,
                                  collate_fn=visit_collate_fn, num_workers=self.params['threads'])
        self.test_loader = DataLoader(dataset=test_set, batch_size=self.params['batch_size'], shuffle=False,
                                 collate_fn=visit_collate_fn, num_workers=self.params['threads'])
        #self.att_loader = DataLoader(dataset=att_set, batch_size=1, shuffle=False,collate_fn=visit_collate_fn, num_workers=self.params['threads'])

    def train(self, train_p, train_seqs, train_labels, test_p, test_seqs, test_labels):
        if self.params['threads'] == -1:
            self.params['threads'] = torch.multiprocessing.cpu_count() - 1 or 1


        print('===> Configuration')
        print(self.params)

        if self.params['cuda']:
            if torch.cuda.is_available():
                print('===> {} GPUs are available'.format(torch.cuda.device_count()))
            else:
                raise Exception("No GPU found, please run with --no-cuda")

        # Fix the random seed for reproducibility
        np.random.seed(self.params['seed'])
        torch.manual_seed(self.params['seed'])


        if self.params['cuda']:
            torch.cuda.manual_seed(self.params['seed'])

        # Data loading
        self._data_loading(train_p, train_seqs, train_labels, test_p, test_seqs, test_labels)
        print('===> Datasets loaded!')

        # Create model
        print('===> Building a Model')
        self.model = TMTL(
            dim_input=self.params['input_dim'],
            dim_output=self.params['output_dim'],
            dim_emb = self.params['emb_dim'],
            task_num=self.params['task_num'],
            dropout_rate=self.params['drop_out']
        )

        if self.params['cuda']:
            self.model = self.model.cuda()
        #print(self.model)


        print('===> Model built!')
        logFile = '../results/training.log'

        # Optimization
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.params['lr'],
                                     weight_decay=self.params['L2_norm'])  # , betas=(0.1, 0.001), eps=1e-8
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=params['lr']*0.005, momentum=0.9, weight_decay=params['L2_norm'])

        # best_valid_epoch = 0
        # best_model_wts = copy.deepcopy(self.model.state_dict())
        #best_train_loss = sys.float_info.max

        best_valid_roc = 0
        best_f1 = 0
        best_recall = 0
        best_precision = 0
        best_buf = 0
        #best_buf_macro = 0

        train_losses = []
        valid_losses = []

        #for ei in trange(self.params['epochs'], desc="Epochs"):
        for ei in range(self.params['epochs']):
            # Train
            train_y_true, train_y_pred, train_loss = self._epoch(self.train_loader, criterion=self.criterion,
                                                           optimizer=optimizer,
                                                           train=True)
            train_losses.append(train_loss)


            valid_y_true, valid_y_pred, valid_loss = self._epoch(self.test_loader, criterion=self.criterion, train=False)
            valid_losses.append(valid_loss)

            probabilities_train = torch.softmax(train_y_pred, dim=-1)
            predictions_train = torch.argmax(train_y_pred, dim=1)
            probabilities_test = torch.softmax(valid_y_pred, dim=-1)
            predictions_test = torch.argmax(valid_y_pred, dim=1)

            accuracy = accuracy_score(train_y_true, predictions_train)
            roc_auc = roc_auc_score(train_y_true, probabilities_train, multi_class='ovr', average='macro')
            recall = recall_score(train_y_true, predictions_train, average='macro')
            precision = precision_score(train_y_true, predictions_train, average='macro', zero_division=0)
            f1 = f1_score(train_y_true, predictions_train, average='macro')

            #roc_auc_macro = roc_auc_score(train_y_true, probabilities_train, multi_class='ovr', average='weighted')
            #recall_macro = recall_score(train_y_true, predictions_train, average='weighted')
            #precision_macro = precision_score(train_y_true, predictions_train, average='weighted', zero_division=0)
            #f1_macro = f1_score(train_y_true, predictions_train, average='weighted')



            accuracy_valid = accuracy_score(valid_y_true, predictions_test)
            roc_auc_valid = roc_auc_score(valid_y_true, probabilities_test, multi_class='ovr', average='macro')
            recall_valid = recall_score(valid_y_true, predictions_test, average='macro')
            precision_valid = precision_score(valid_y_true, predictions_test, average='macro', zero_division=0)
            f1_valid = f1_score(valid_y_true, predictions_test, average='macro')

            #roc_auc_valid_macro = roc_auc_score(valid_y_true, probabilities_test, multi_class='ovr', average='macro')
            #recall_valid_macro = recall_score(valid_y_true, predictions_test, average='macro')
            #precision_valid_macro = precision_score(valid_y_true, predictions_test, average='macro', zero_division=0)
            #f1_valid_macro = f1_score(valid_y_true, predictions_test, average='macro')



            buf_average = "Epoch {} - Loss_tr: {:.4f}, Loss_val: {:.4f},\n" \
                  "f1_train: {:.4f}, f1_val: {:.4f},\n " \
                  "acc_train: {:.4f}, acc_val: {:.4f},\n " \
                  "precision_train: {:.4f}, precision_val: {:.4f},\n "\
                  "recall_train: {:.4f}, recall_val: {:.4f},\n " \
                  "aucroc_train: {:.4f}, aucroc_val: {:.4f}".format(ei, train_loss, valid_loss, f1, f1_valid, accuracy, accuracy_valid, precision, precision_valid, recall, recall_valid, roc_auc, roc_auc_valid)

            #buf_macro = "Epoch {} - Loss_tr: {:.4f}, Loss_val: {:.4f},\n" \
                  #"f1_train: {:.4f}, f1_val: {:.4f},\n " \
                  #"acc_train: {:.4f}, acc_val: {:.4f},\n " \
                  #"precision_train: {:.4f}, precision_val: {:.4f},\n "\
                  #"recall_train: {:.4f}, recall_val: {:.4f},\n " \
                  #"aucroc_train: {:.4f}, aucroc_val: {:.4f}".format(ei, train_loss, valid_loss, f1_macro, f1_valid_macro, accuracy, accuracy_valid, precision_macro, precision_valid_macro, recall_macro, recall_valid_macro, roc_auc_macro, roc_auc_valid_macro)

            if best_valid_roc < roc_auc_valid:
                best_valid_roc = roc_auc_valid
                best_precision = precision_valid
                best_recall = recall_valid
                best_f1 = f1_valid
                best_buf = buf_average
                #best_buf_macro = buf_macro

            if params['print']:
                print(buf_average)
                print()
            #print2file(buf_average, logFile)
        return best_valid_roc, best_buf, best_precision, best_recall, best_f1

    def _epoch(self, loader, criterion, optimizer=None, train=False):
        if train and not optimizer:
            raise AttributeError("Optimizer should be given for training")

        if train:
            self.model.train()
            mode = 'Train'
        else:
            self.model.eval()
            mode = 'Eval'

        losses = AverageMeter()
        labels = []
        outputs = []

        # for bi, batch in enumerate(tqdm(loader, desc="{} batches".format(mode), leave=False)):
        for batch in loader:
            patients, inputs, targets, lengths, sorted_indice = batch
            if self.params['cuda']:
                inputs = inputs.cuda()
                targets = targets.cuda()

            #model_device = next(self.model.parameters()).device

            output = self.model(inputs, lengths)
            #print(output.shape)
            #print(targets.shape)
            loss = criterion(output, targets.squeeze().to(torch.int64))

            # compute gradient and do update step
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if self.params['cuda']:
                loss = loss.cpu()
                output = output.cpu()
                targets = targets.cpu()

            # sort back
            if not train:
                sort_back_indices, _ = zip(*sorted(enumerate(sorted_indice), key=lambda x: x[1]))
                sort_back_indices = list(sort_back_indices)
                output = output[sort_back_indices]
                targets = targets[sort_back_indices]

            outputs.append(output.detach())
            labels.append(targets.detach())

            # record loss
            losses.update(loss.detach().numpy(), inputs.size(0))
            # if train:
            #     break

        return torch.cat(labels, 0), torch.cat(outputs, 0), losses.avg


def print2file(buf, outFile):
    outfd = open(outFile, 'a')
    outfd.write(buf + '\n')
    outfd.close()



"""def main(params, class_weights):

    model = Model_trainer(params, class_weights)
    best_valid_roc, best_buf, best_precision, best_recall, best_f1 = model.train(train_p, train_X, train_Y, test_p, test_X, test_Y)

    print(f'best valid auroc : {best_valid_roc}')
    print(best_buf)
    return best_valid_roc, best_buf, best_precision, best_recall, best_f1


data_input_df, labels_sgd_df, patient_train, input_seqs, output_seqs, \
        patient_test, input_seqs_test, output_seqs_test, data_idx_list, class_weights = \
        load_dataset_classification('../processed_data/data_df_GDS.csv', '../processed_data/data_label_GDS_new2.csv',
                                    random_state=45, test_ratio=0.20)

train_p, train_X, train_Y = patient_train, input_seqs, output_seqs
test_p, test_X, test_Y = patient_test, input_seqs_test, output_seqs_test

params = {}
params['input_dim'] = len(train_X[0][0])
params['output_dim'] = 4
params['lr'] = 0.001
params['epochs'] = 100
params['L2_norm'] = 0.0001
params['seed'] = 45
params['threads'] = 0
params['cuda'] = True
params['print'] = False
if not torch.cuda.is_available():
    params['cuda'] = False


best_aucroc = 0
best_buf = 0
for task_num in range(1, 6):
    params['task_num'] = task_num
    for embd_dim in [10, 15, 20, 32]:
        params['emb_dim']=embd_dim
        for drpo in [0.2, 0.3, 0.4, 0.5]:
            params['drop_out'] = drpo
            for batch_size in [100, 128, 252]:
                params['batch_size'] = batch_size
                best_valid_roc, best_buf_candidate, best_precision, best_recall, best_f1 = main(params, class_weights)
                print()
                if best_valid_roc > best_aucroc:
                    best_aucroc = best_valid_roc
                    best_buf = best_buf_candidate
print('=======================================================')
print(best_aucroc)
print(best_buf)
"""


"""
if __name__ == "__main__":

    data_input_df, labels_sgd_df, patient_train, input_seqs, output_seqs,\
        patient_test, input_seqs_test, output_seqs_test, data_idx_list, class_weights =\
        load_dataset_classification('../processed_data/data_df_GDS.csv', '../processed_data/data_label_GDS_new2.csv', random_state=42, test_ratio=0.20)


    print(class_weights)
    train_p, train_X, train_Y = patient_train, input_seqs, output_seqs

    test_p, test_X, test_Y = patient_test, input_seqs_test, output_seqs_test




    params = {}
    params['input_dim'] = len(train_X[0][0])
    params['output_dim'] = 4
    params['lr'] = 0.001
    params['batch_size'] = 100
    params['epochs'] = 100
    params['L2_norm'] = 0.0001
    params['emb_dim'] = 10
    params['drop_out'] = 0.2
    params['seed'] = 1234
    params['task_num'] = 3
    params['threads'] = 0
    params['cuda'] = True
    params['print'] = True
    if not torch.cuda.is_available():
        params['cuda'] = False

    model = Model_trainer(params, class_weights)
    best_valid_roc, best_buf, best_precision, best_recall, best_f1 = model.train(train_p, train_X, train_Y, test_p, test_X, test_Y)
    print(best_valid_roc)"""







"""def main(params, class_weights, train_p, train_X, train_Y, test_p, test_X, test_Y):

    model = Model_trainer(params, class_weights)
    best_valid_roc, best_buf, best_precision, best_recall, best_f1 = model.train(train_p, train_X, train_Y, test_p, test_X, test_Y)

    print(f'best valid auroc : {best_valid_roc}')
    print(best_buf)
    return best_valid_roc, best_buf, best_precision, best_recall, best_f1


def func_seed(params):
    auroc_list = []
    f1_list = []
    recall_list = []
    precision_list = []
    for seed in [123, 321, 45, 65, 52]:
        #print(f'----------------seed: {seed}--------------')

        data_input_df, labels_sgd_df, patient_train, input_seqs, output_seqs, \
            patient_test, input_seqs_test, output_seqs_test, data_idx_list, class_weights = \
            load_dataset_classification('../processed_data/data_df_GDS.csv',
                                        '../processed_data/data_label_GDS_new2.csv',
                                        random_state=seed, test_ratio=0.20)

        train_p, train_X, train_Y = patient_train, input_seqs, output_seqs
        test_p, test_X, test_Y = patient_test, input_seqs_test, output_seqs_test

        params['input_dim'] = len(train_X[0][0])
        params['seed'] = seed
        best_valid_roc, best_buf, best_precision, best_recall, best_f1 = main(params, class_weights, train_p, train_X, train_Y, test_p, test_X, test_Y)
        print()
        auroc_list.append(best_valid_roc)
        f1_list.append(best_f1)
        recall_list.append(best_recall)
        precision_list.append(best_precision)

    print('=======================================================')
    print(f'avg auroc: {sum(auroc_list) / len(auroc_list)}')
    print(f'avg f1: {sum(f1_list) / len(f1_list)}')
    print(f'avg recall: {sum(recall_list) / len(recall_list)}')
    print(f'avg precision: {sum(precision_list) / len(precision_list)}')

    return sum(auroc_list)/len(auroc_list), sum(f1_list)/len(f1_list), sum(recall_list)/len(recall_list), sum(precision_list)/len(precision_list)


params = {}
params['output_dim'] = 4
params['lr'] = 0.001
params['epochs'] = 100
params['L2_norm'] = 0.0001
params['threads'] = 0
params['cuda'] = True
params['print'] = False
if not torch.cuda.is_available():
    params['cuda'] = False


best_aucroc = 0
best_buf = 0
for task_num in range(1, 6):
    params['task_num'] = task_num
    for embd_dim in [10, 15, 20, 32]:
        params['emb_dim']=embd_dim
        for drpo in [0.2, 0.3, 0.4, 0.5]:
            params['drop_out'] = drpo
            for batch_size in [100, 128, 252]:
                params['batch_size'] = batch_size
                print('----------------')
                print('main params')
                print(params)
                print('----------------')
                aucroc, f1, recall, precision = func_seed(params)
                if aucroc > best_aucroc:
                    best_aucroc = aucroc
                    #best_buf = best_buf_candidate
print('=======================================================')
print(best_aucroc)
#print(best_buf)"""

def main(params, class_weights):

    model = Model_trainer(params, class_weights)
    best_valid_roc, best_buf, best_precision, best_recall, best_f1 = model.train(train_p, train_X, train_Y, test_p, test_X, test_Y)

    print(f'best valid auroc : {best_valid_roc}')
    print(best_buf)
    return best_valid_roc, best_buf, best_precision, best_recall, best_f1




params = {}
#params['input_dim'] = len(train_X[0][0])
params['output_dim'] = 4
params['lr'] = 0.001
params['epochs'] = 100
params['L2_norm'] = 0.0001
params['threads'] = 0
params['cuda'] = True
params['print'] = False
if not torch.cuda.is_available():
    params['cuda'] = False

params['task_num'] =  1
params['emb_dim']= 20
params['drop_out'] = 0.20
params['batch_size'] = 128



auroc_list = []
f1_list=[]
recall_list = []
precision_list = []

for seed in [123, 321, 45, 65, 52]:
    print(f'----------------seed: {seed}--------------')

    data_input_df, labels_sgd_df, patient_train, input_seqs, output_seqs, \
        patient_test, input_seqs_test, output_seqs_test, data_idx_list, class_weights = \
        load_dataset_classification('../processed_data/data_df_GDS.csv', '../processed_data/data_label_GDS_new2.csv',
                                    random_state=seed, test_ratio=0.20)

    train_p, train_X, train_Y = patient_train, input_seqs, output_seqs
    test_p, test_X, test_Y = patient_test, input_seqs_test, output_seqs_test

    params['input_dim'] = len(train_X[0][0])
    params['seed'] = seed
    best_valid_roc, best_buf, best_precision, best_recall, best_f1= main(params, class_weights)
    print()
    auroc_list.append(best_valid_roc)
    f1_list.append(best_f1)
    recall_list.append(best_recall)
    precision_list.append(best_precision)


print('=======================================================')
print(f'avg auroc: {sum(auroc_list)/len(auroc_list)}')
print(f'avg f1: {sum(f1_list)/len(f1_list)}')
print(f'avg recall: {sum(recall_list)/len(recall_list)}')
print(f'avg precision: {sum(precision_list)/len(precision_list)}')