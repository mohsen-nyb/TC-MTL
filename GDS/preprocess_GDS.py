import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def load_dataset_classification(path_data, path_label, random_state=42, test_ratio=0.20):
    print('start data preprocessing')
    data_input_df = pd.read_csv(path_data)
    labels_sgd_df = pd.read_csv(path_label)
    data_idx_list = [4, 238, 259, 633]


    patients0 = labels_sgd_df[labels_sgd_df['GDS_class']==0]['VISITNUM'].unique()
    patients1 = labels_sgd_df[labels_sgd_df['GDS_class'] == 1]['VISITNUM'].unique()
    patients2 = labels_sgd_df[labels_sgd_df['GDS_class'] == 2]['VISITNUM'].unique()
    patients3 = labels_sgd_df[labels_sgd_df['GDS_class'] == 3]['VISITNUM'].unique()



    patient_train0, patient_test0 = train_test_split(patients0, test_size=test_ratio, random_state=random_state)
    patient_train1, patient_test1 = train_test_split(patients1, test_size=test_ratio, random_state=random_state)
    patient_train2, patient_test2 = train_test_split(patients2, test_size=test_ratio, random_state=random_state)
    patient_train3, patient_test3 = train_test_split(patients3, test_size=test_ratio, random_state=random_state)

    patient_visit_train = list(patient_train0)+list(patient_train1)+list(patient_train2)+list(patient_train3)
    patient_visit_test = list(patient_test0)+list(patient_test1)+list(patient_test2)+list(patient_test3)


    input_seqs = []
    output_seqs = []
    for visit_pat in patient_visit_train:
        output_seqs.append(labels_sgd_df[labels_sgd_df['VISITNUM'] == visit_pat]['GDS_class'].values[0])

        p_name = labels_sgd_df[labels_sgd_df['VISITNUM'] == visit_pat]['PTID'].values[0]
        max_visit_num = labels_sgd_df[labels_sgd_df['VISITNUM'] == visit_pat]['num'].values[0]
        pt_lst = []
        visitnames = labels_sgd_df[labels_sgd_df['PTID']==p_name]['VISITNUM'].values
        for i in range(1, max_visit_num+1):
            visitnum = p_name + '_' + str(i)
            if visitnum in visitnames:
                visit_seq = data_input_df[data_input_df['VISITNUM'] == visitnum].values[0, 2:].tolist()
                pt_lst.append(visit_seq)
        input_seqs.append(pt_lst)


    input_seqs_test = []
    output_seqs_test = []
    for visit_pat in patient_visit_test:
        output_seqs_test.append(labels_sgd_df[labels_sgd_df['VISITNUM'] == visit_pat]['GDS_class'].values[0])

        p_name = labels_sgd_df[labels_sgd_df['VISITNUM'] == visit_pat]['PTID'].values[0]
        max_visit_num = labels_sgd_df[labels_sgd_df['VISITNUM'] == visit_pat]['num'].values[0]
        pt_lst = []
        visitnames = labels_sgd_df[labels_sgd_df['PTID']==p_name]['VISITNUM'].values
        for i in range(1, max_visit_num+1):
            visitnum = p_name + '_' + str(i)
            if visitnum in visitnames:
                visit_seq = data_input_df[data_input_df['VISITNUM'] == visitnum].values[0, 2:].tolist()
                pt_lst.append(visit_seq)
        input_seqs_test.append(pt_lst)

    class_labels = [0, 1, 2, 3]  # List of class labels
    class_weights = compute_class_weight('balanced', classes=class_labels, y=output_seqs)

    print('data preprocessing done!')

    return data_input_df, labels_sgd_df, patient_visit_train, input_seqs, output_seqs, patient_visit_test, input_seqs_test, output_seqs_test, data_idx_list, class_weights