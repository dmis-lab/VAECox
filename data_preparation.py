import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import *
import networkx as nx
import pickle
import os
import multiprocessing as mp
import sys
import random
np.set_printoptions(threshold=sys.maxsize)

cancer_list_dict = {
    'ching': ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD'],
    'wang': ['ACC', 'BLCA', 'BRCA', 'CESC', 'UVM', 'CHOL', 'ESCA', 'HNSC', 'KIRC', 'KIRP', 
             'LGG', 'LIHC', 'LUAD', 'LUSC', 'MESO', 'PAAD', 'SARC', 'SKCM', 'STAD', 'UCEC', 'UCS'],
    'all': ['ACC', 'BLCA', 'BRCA', 'CESC', 'UVM', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM', 
            'HNSC', 'KICH', 'KIPAN', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC', 
            'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD', 'STES', 
            'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS']
}

def cancer_list(key):
    return cancer_list_dict[key]

def get_dataset_811(config):
    data_dict = {
        'train': dict(),
        'valid': dict(),
        'test': dict(),
        'coo': dict()
    }
    WHAT_OMICS = "_".join(config.omic_list)
    ORIGINAL_FILE = "./data/merged_splits/{0}_811_{1}.tsv".format(config.vae_data,WHAT_OMICS)
    MASKING_FILE = "./data/merged_splits/{0}_{1}_binary.csv".format(config.vae_data,WHAT_OMICS)
    PICKLE_PATH = "./data/merged_splits/{0}_811_{1}_{2}_{3}_{4}_{5}.pickle".format(config.vae_data, config.gcn_mode, config.feature_scaling, config.feature_selection, config.sub_graph, WHAT_OMICS)

    if not os.path.isfile(PICKLE_PATH):
        print("Making new pickle file...")
        # Missing Value Handling
        df = pd.read_csv(ORIGINAL_FILE, sep="\t", header=0, index_col=0)
        mf = pd.read_csv(MASKING_FILE, sep=",", header=0, index_col=0)
        mf = mf.reindex(df.index)
        print(df.index)
        print(mf.index)
        # mf = mf.replace(0,np.nan)
        # mf = mf.dropna(how='all',axis=0)
        # mf = mf.dropna(how='all',axis=1)

        # df = df.dropna(subset=['Cli@Days2Death', 'Cli@Days2FollowUp'], how='all')
        df.fillna(0.0, axis=1, inplace=True)

        # Dataset Split Train, Valid and Test
        df_train = df.loc[df['Fold@811'] == 0]
        df_valid = df.loc[df['Fold@811'] == 1]
        df_test = df.loc[df['Fold@811'] == 2]

        for omic in config.omic_list:
            data_dict['train'][omic] = df_train[[x for x in df_train.columns.get_values() if omic in x]]
            data_dict['valid'][omic] = df_valid[[x for x in df_valid.columns.get_values() if omic in x]]
            data_dict['test'][omic] = df_test[[x for x in df_test.columns.get_values() if omic in x]]
            data_dict['train'][omic + '_mask'] = mf.loc[df_train.index]
            data_dict['valid'][omic + '_mask'] = mf.loc[df_valid.index]
            data_dict['test'][omic + '_mask'] = mf.loc[df_test.index]


        # Dataset Clinical Data Handling
        # data_dict = clinical_handling(df_train, df_test, data_dict)

        # Dataset Feature Extraction
        if config.feature_selection is not None:
            data_dict = _feature_selection(config, data_dict)

        # Dataset Network Extraction
        if config.gcn_mode:
            data_dict = _network_extraction(config, data_dict)

        # Dataset 'Numpification'
        for omic in config.omic_list:
            data_dict['train'][omic] = np.array(data_dict['train'][omic].values).astype('float64')
            data_dict['valid'][omic] = np.array(data_dict['valid'][omic].values).astype('float64')
            data_dict['test'][omic] = np.array(data_dict['test'][omic].values).astype('float64')
            data_dict['train'][omic + '_mask'] = np.array(data_dict['train'][omic + '_mask'].values).astype('float64')
            data_dict['valid'][omic + '_mask'] = np.array(data_dict['valid'][omic + '_mask'].values).astype('float64')
            data_dict['test'][omic + '_mask'] = np.array(data_dict['test'][omic + '_mask'].values).astype('float64')

        # For Cox Survival Regression
        # data_dict = sort_by_reverse(data_dict)

        # Dataset Feature Scaling
        data_dict = _feature_scaling(config, data_dict)

        with open(PICKLE_PATH, "wb") as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(PICKLE_PATH, "rb") as handle:
        data_dict = pickle.load(handle)
    
    for o in config.omic_list:
        print('Train', o, data_dict['train'][o].shape)
        print('Valid', o, data_dict['valid'][o].shape)
        print('Test', o, data_dict['test'][o].shape)
        # _data_stats(data_dict, o)

    return data_dict


def _data_stats(data_dict, omics):
    full_data = np.vstack((data_dict['train'][omics], data_dict['valid'][omics], data_dict['test'][omics]))
    print("Entire Values | Mean: {}, Median: {}, Stdev: {}".format(np.mean(full_data), np.median(full_data), np.std(full_data)))
    entire = np.vstack((np.mean(full_data), np.median(full_data), np.std(full_data)))
    row_wise = np.vstack((np.mean(full_data, 0), np.median(full_data, 0), np.std(full_data, 0)))
    col_wise = np.vstack((np.mean(full_data, 0), np.median(full_data, 0), np.std(full_data, 0)))
    np.savetxt("./entire_values.csv", entire, delimiter=",")
    np.savetxt("./row_wise.csv", row_wise, delimiter=",")
    np.savetxt("./col_wise.csv", col_wise, delimiter=",")

def get_statistics(cancer_type='ACC' ):
    ORIGINAL_FILE = "./data/merged_splits/" + cancer_type + "_5fold.tsv"
    df = pd.read_csv(ORIGINAL_FILE, sep="\t", header=0, index_col=0)
    print(cancer_type, "Num of Censored Data:", df['Cli@Censored'].sum(), "Num of Uncensored Data:", df.shape[0] - df['Cli@Censored'].sum(), "Total:", df.shape[0])
    
    # print(cancer_type, "Number of Censored Data: ", df.loc[df['Cli@Censored'] == 1.0].sum())
    # print(cancer_type, "Number of Uncensored Data: ", df.loc[df['Cli@Censored'] == 0.0].sum()) 

def get_dataset(config, fold=0):
    pool = mp.Pool(processes=10)
    pickle_dict = dict()
    WHAT_OMICS = "_".join(config.omic_list) 
    PICKLE_PATH = "./data/merged_splits/{0}_5fold_{1}_{2}_{3}_{4}_{5}_{6}.pickle".format(config.cancer_list, config.missing_impute, config.gcn_mode, config.feature_scaling, config.augment_autoencoder, config.deseq2, WHAT_OMICS)

    if not os.path.isfile(PICKLE_PATH):
        results = []
        for cancer_type in cancer_list_dict[config.cancer_list]:
            results.append(pool.apply_async(_get_subdataset, args=(config, cancer_type, 0)))
        outputs = [p.get() for p in results]
        
        for output in outputs:
            c, d = output
            pickle_dict[c] = _combine_train_test(d)

        if config.pickle_save:
            print("Saving: ", PICKLE_PATH)
            with open(PICKLE_PATH, "wb") as handle:
                pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        with open(PICKLE_PATH, "rb") as handle:
            pickle_dict = pickle.load(handle)
    except:
        pickle_dict = pickle_dict

    for c in cancer_list_dict[config.cancer_list]:
        for o in config.omic_list:
            print(c, 'Data Size', o, pickle_dict[c][o].shape)

    return pickle_dict


def get_dataset_fold(config, fold=0):
    pool = mp.Pool(processes=10)
    pickle_dict = dict()
    WHAT_OMICS = "_".join(config.omic_list) 
    PICKLE_PATH = "./data/merged_splits/{0}_5fold_{1}_{2}_{3}_{4}_{5}_{6}_{7}.pickle".format(config.cancer_list, config.missing_impute, config.gcn_mode, config.feature_scaling, config.augment_autoencoder, config.sub_graph, fold, WHAT_OMICS)

    if not os.path.isfile(PICKLE_PATH):
        results = []
        for cancer_type in cancer_list_dict[config.cancer_list]:
            results.append(pool.apply_async(_get_subdataset, args=(config, cancer_type, fold)))
        outputs = [p.get() for p in results]
        
        for output in outputs:
            c, d = output
            pickle_dict[c] = d

        if config.pickle_save:
            print("Saving: ", PICKLE_PATH)
            with open(PICKLE_PATH, "wb") as handle:
                pickle.dump(pickle_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    try:
        with open(PICKLE_PATH, "rb") as handle:
            pickle_dict = pickle.load(handle)
    except:
        pickle_dict = pickle_dict

    for c in cancer_list_dict[config.cancer_list]:
        for o in config.omic_list:
            print(c, 'Train', o, pickle_dict[c]['train'][o].shape)
            print(c, 'Test', o, pickle_dict[c]['test'][o].shape)

    return pickle_dict

def _combine_train_test(data_dict):
    new_dict = dict()
    for key in data_dict['train'].keys():
        if 'Cli@' in key:
            new_dict[key] = np.hstack((data_dict['train'][key], data_dict['test'][key]))
        else:
            new_dict[key] = np.vstack((data_dict['train'][key], data_dict['test'][key])) 

    return new_dict

def _get_subdataset(config, cancer_type, fold):
    get_statistics(cancer_type)

    if config.deseq2:
        ORIGINAL_FILE = "./data/merged_splits/" + cancer_type + "_DESeq2.tsv"
    else: 
        ORIGINAL_FILE = "./data/merged_splits/" + cancer_type + "_5fold.tsv"
    data_dict = {'train': dict(), 'test': dict(), 'adj': dict(), 'deg': dict()}
    df = pd.read_csv(ORIGINAL_FILE, sep="\t", header=0, index_col=0)
    
    # Drop Zero Survivals
    df = _drop_zero_survival(config, df)

    # Autoencoder Augmentation
    df = _autoencoder_augmentation(config, df)

    # Missing Value Imputation
    df = _missing_imputation(config, df)

    # Dataset Split Train and Test
    df_train, df_test, data_dict = _train_test_split(config, df, fold)

    # Dataset Clinical Data Handling
    data_dict = _clinical_handling(df_train, df_test, data_dict)

    # Dataset Feature Extraction
    # Not Implemented

    # Dataset Network Extraction
    data_dict = _network_extraction(config, data_dict)

    # Dataset 'Numpification'
    for omic in config.omic_list + ['Cli@Survival']:
        data_dict['train'][omic] = np.array(data_dict['train'][omic].values).astype('float64')
        data_dict['test'][omic] = np.array(data_dict['test'][omic].values).astype('float64')

    # For Cox Survival Regression
    data_dict = _sort_by_reverse(data_dict)

    # For Estimating IPCWs
    data_dict = _get_ipc_weights(data_dict)

    # Dataset Feature Scaling
    data_dict = _feature_scaling(config, data_dict)

    assert 'Cli@IPCW' in data_dict['train'].keys()
    assert 'Cli@IPCW' in data_dict['test'].keys()

    return cancer_type, data_dict

def _drop_zero_survival(config, df):
    df = df.drop(df[df['Cli@Days2Death'] == 0.0].index)
    df = df.drop(df[df['Cli@Days2FollowUp'] == 0.0].index)

    return df

def _autoencoder_augmentation(config, df):
    if 'None' not in config.augment_autoencoder:
        WHAT_OMICS = "_".join(config.omic_list)
        ORIGINAL_FILE = "./data/merged_splits/{0}_811_{1}.tsv".format(config.vae_data,WHAT_OMICS)
        ae_df = pd.read_csv(ORIGINAL_FILE, sep="\t", header=0, index_col=0)
        ae_cols = list(ae_df.drop(columns=['Fold@811']).columns.get_values())
        df = df[ae_cols + ['Cli@Days2Death', 'Cli@Days2FollowUp', 'Cli@Censored', 'Fold@CV']]
    else:
        WHAT_OMICS = "_".join(config.omic_list)
        ORIGINAL_FILE = "./data/merged_splits/{0}_811_{1}.tsv".format(config.vae_data,WHAT_OMICS)
        ae_df = pd.read_csv(ORIGINAL_FILE, sep="\t", header=0, index_col=0)
        ae_cols = list(ae_df.drop(columns=['Fold@811']).columns.get_values())
        df = df[ae_cols + ['Cli@Days2Death', 'Cli@Days2FollowUp', 'Cli@Censored', 'Fold@CV']]

    return df

def _missing_imputation(config, df):
    for col in df.columns.values:
            if 'Cli@' not in col:
                if config.missing_impute == 'mean':
                    df[col] = df[col].fillna((df[col].mean()))
                elif config.missing_impute == 'median':
                    df[col] = df[col].fillna((df[col].median()))
                else:
                    continue
    try:
        df = df.dropna(subset=['Cli@Days2Death', 'Cli@Days2FollowUp'], how='all')
    except: 
        pass
    df.fillna(0.0, axis=1, inplace=True)

    return df

def _train_test_split(config, df, fold):
    data_dict = {'train': dict(), 'test': dict()}
    df_train = df.loc[df['Fold@CV'] != fold]
    df_test = df.loc[df['Fold@CV'] == fold]
    for omic in config.omic_list:
        data_dict['train'][omic] = df_train[[x for x in df_train.columns.get_values() if omic in x]]
        data_dict['test'][omic] = df_test[[x for x in df_test.columns.get_values() if omic in x]]
    
    return df_train, df_test, data_dict

def _clinical_handling(df_train, df_test, data_dict):
    df_train_cli = df_train[[x for x in df_train.columns.get_values() if "Cli@Days" in x]]
    df_train_cen = df_train[[x for x in df_train.columns.get_values() if "Cli@Censored" in x]]
    df_train_mask = df_train_cen.replace({0: 1, 1: 0})

    df_test_cli = df_test[[x for x in df_test.columns.get_values() if "Cli@Days" in x]]
    df_test_cen = df_test[[x for x in df_test.columns.get_values() if "Cli@Censored" in x]]
    df_test_mask = df_test_cen.replace({0: 1, 1: 0})

    df_train_cli['Cli@Days2Death'] = df_train_cli['Cli@Days2Death'].multiply(df_train_mask['Cli@Censored'])
    df_train_cli['Cli@Days2FollowUp'] = df_train_cli['Cli@Days2FollowUp'].multiply(df_train_cen['Cli@Censored'])

    df_test_cli['Cli@Days2Death'] = df_test_cli['Cli@Days2Death'].multiply(df_test_mask['Cli@Censored'])
    df_test_cli['Cli@Days2FollowUp'] = df_test_cli['Cli@Days2FollowUp'].multiply(df_test_cen['Cli@Censored'])

    df_train_cli['Cli@Survival'] = df_train_cli['Cli@Days2Death'] + df_train_cli['Cli@Days2FollowUp']
    data_dict['train']['Cli@Survival'] = df_train_cli['Cli@Survival']
    # data_dict['train']['Cli@Censored'] = np.ravel(df_train_cen.values)
    data_dict['train']['Cli@Masking'] = np.ravel(df_train_mask.values)
    data_dict['train']['Cli@Censored'] = np.ravel(df_train_cen.values)

    df_test_cli['Cli@Survival'] = df_test_cli['Cli@Days2Death'] + df_test_cli['Cli@Days2FollowUp']
    data_dict['test']['Cli@Survival'] = df_test_cli['Cli@Survival']

    # data_dict['test']['Cli@Censored'] = np.ravel(df_test_cen.values)
    data_dict['test']['Cli@Masking'] = np.ravel(df_test_mask.values)
    data_dict['test']['Cli@Censored'] = np.ravel(df_test_cen.values)

    return data_dict

def _sort_by_reverse(data_dict):
    for key in ['train', 'test']:
        sorted_indices = np.argsort(-data_dict[key]['Cli@Survival'])
        for sub_key in data_dict[key].keys():
            data_dict[key][sub_key] = data_dict[key][sub_key][sorted_indices]
    return data_dict

def _get_num_censored(surv_list, cens_list, surv_time):
    indices = [i for i, val in enumerate(surv_list) if val == surv_time]
    censored = [cens_list[i] for i in indices]
    return sum(censored)

def _get_ipc_weights(data_dict):
    for key in ['train', 'test']:
        ipcw_list, temp_list = [], []
        surv_list = list(data_dict[key]['Cli@Survival']) 
        mask_list = list(data_dict[key]['Cli@Masking'])
        cens_list = list(data_dict[key]['Cli@Censored'])
        for idx, item in enumerate(surv_list):
            _denom = sum(i <= item for i in surv_list)
            _numer = _denom - _get_num_censored(surv_list, cens_list, item)
            try:
                censored_ratio = float(_numer) / float(_denom)
            except Exception as e:
                censored_ratio = 1.0
            assert not np.isinf(censored_ratio)
            assert not np.isnan(censored_ratio)
            temp_list.append(censored_ratio)

        # temp_list = [0.0 if np.isinf(i) or np.isnan(i) else i for i in temp_list]
        temp_list = [i for i in temp_list if i > 0.0]
        for idx, item in enumerate(mask_list):
            try:
                _ggggg = np.prod(temp_list[idx + 1:])
                ipcw_list.append(float(item) / float(_ggggg))
            except Exception as e:
                ipcw_list.append(0.0)
        try:
            ipcw_list = [v / sum(ipcw_list) for v in ipcw_list]
        except:
            print("ERROR")

        data_dict[key]['Cli@IPCW'] = np.array(ipcw_list)
        assert not np.isnan(data_dict[key]['Cli@IPCW']).any()
    return data_dict

def _feature_selection(config, data_dict):
    if 'None' not in config.feature_selection:
        selected_genes = []
        for omic in config.omic_list:
            old_genes = data_dict['train'][omic].columns.get_values()
            temp_genes = [g.split("|")[0] for g in old_genes]
            print(temp_genes)
            with open('./data/{}_genes.txt'.format(config.feature_selection), "r") as fr:
                for gene in fr.readlines():
                    selected_genes.append(omic + gene.split("\n")[0])
                print(selected_genes)
                with open("./minji_vae_genes.pickle", "wb") as handle:
                    pickle.dump(selected_genes, handle, protocol=pickle.HIGHEST_PROTOCOL)
                for mode in ['train', 'valid', 'test']:
                    data_dict[mode][omic].columns = temp_genes
                    data_dict[mode][omic] = data_dict[mode][omic][selected_genes]
                    data_dict[mode][omic + '_mask'].columns = temp_genes
                    data_dict[mode][omic + '_mask'] = data_dict[mode][omic + '_mask'][selected_genes]
    return data_dict

def _feature_scaling(config, data_dict):
    if 'None' not in config.feature_scaling:
        for key in data_dict['train'].keys():
            if 'Cli@' not in key and '_mask' not in key:
                scaler = StandardScaler() 
                if config.feature_scaling == 'z':
                    scaler = StandardScaler()
                elif config.feature_scaling == 'minmax':
                    scaler = MinMaxScaler()
                else:
                    scaler = None
                data_dict['train'][key] = scaler.fit_transform(data_dict['train'][key])
                try:
                    data_dict['valid'][key] = scaler.transform(data_dict['valid'][key])
                except Exception as e:
                    pass
                data_dict['test'][key] = scaler.transform(data_dict['test'][key])
            assert not np.isnan(data_dict['train'][key]).any()
            assert not np.isnan(data_dict['test'][key]).any()
    return data_dict

def _make_coo_matrix(adj_matrix):
    adj_matrix = np.array(adj_matrix)
    coo_matrix = []
    rows, cols = adj_matrix.shape[0], adj_matrix.shape[1]
    for row in range(rows):
        # print(adj_matrix[row, :])
        edge_idx_partners = list(np.nonzero(adj_matrix[row, :])[0])
        # print(edge_idx_partners)
        for i in edge_idx_partners:
            if row <= i:
                coo_matrix.append([row, i])
    return np.array(coo_matrix).T
        
def _make_deg_matrix(adj_matrix):
    adj_matrix = np.array(adj_matrix)
    deg_matrix = []
    rows, cols = adj_matrix.shape[0], adj_matrix.shape[1]
    sums = np.sum(adj_matrix, axis=1)
    for row in range(rows):
        temp = np.zeros(shape=rows)
        temp[row] = sums[row]
        deg_matrix.append(temp)
    try:
        return np.sqrt(np.linalg.pinv(np.array(deg_matrix))).astype(float)

    except Exception as e:
        print(e)
        return np.sqrt(np.array(deg_matrix)).astype(float)

def _network_extraction(config, data_dict):
    if config.gcn_mode:
        net = pickle.load(open("./network/networkx_biogrid.pickle", "rb"))
        for omic in config.omic_list:
            new_features = []
        all_features = data_dict['train'][omic].columns.values
        if 'miRNA' not in omic:
            gene_features = [g.replace(omic, "") for g in all_features if omic in g and "?" not in g]
            if config.sub_graph != 0:
                    gene_features = random.sample(gene_features, config.sub_graph)
            if omic == 'mRNA@':
                temp_dict = dict()
                for g in gene_features:
                    temp_dict[g.split("|")[0]] = g
                gene_features = [g.split("|")[0] for g in gene_features]
                net = net.subgraph(gene_features)
                idx_name_dict = {i: name for i, name in enumerate(net.nodes())}
                adj = np.array(nx.adjacency_matrix(net).todense()).astype(dtype='float', casting='same_kind')
                data_dict['coo'][omic] = _make_coo_matrix(adj)
                # data_dict['coo'][omic] = nx.to_scipy_sparse_matrix(net, format='coo')
                # data_dict['adj'][omic] = np.array(nx.adjacency_matrix(net).todense()).astype(dtype='float',
                #                      casting='same_kind')
                # data_dict['adj'][omic] = np.add(data_dict['adj'][omic], np.identity(data_dict['adj'][omic].shape[0]))
                # data_dict['deg'][omic] = make_deg_matrix(data_dict['adj'][omic]).astype(dtype='float',
                #                         casting='same_kind')
                for idx in range(len(idx_name_dict)):
                    new_features.append(omic + temp_dict[idx_name_dict[idx]])
            else:
                net = net.subgraph(gene_features)
                idx_name_dict = {i: name for i, name in enumerate(net.nodes())}
                adj = np.array(nx.adjacency_matrix(net).todense()).astype(dtype='float', casting='same_kind')
                data_dict['coo'][omic] = _make_coo_matrix(adj)
                # data_dict['coo'][omic] = nx.to_scipy__sparse_matrix(net, format='coo')
                # data_dict['adj'][omic] = np.array(nx.adjacency_matrix(net).todense()).astype(dtype='float',
                #                      casting='same_kind')
                # data_dict['adj'][omic] = np.add(np.eye(data_dict['adj'][omic].shape[0]), data_dict['adj'][omic])
                # data_dict['deg'][omic] = make_deg_matrix(data_dict['adj'][omic]).astype(dtype='float',
                #                         casting='same_kind')
                for idx in range(len(idx_name_dict)):
                    new_features.append(omic + idx_name_dict[idx])
        else:
            new_features += [g for g in all_features if omic in g]
        data_dict['train'][omic] = data_dict['train'][omic][new_features]
        data_dict['valid'][omic] = data_dict['valid'][omic][new_features]
        data_dict['test'][omic] = data_dict['test'][omic][new_features]
        data_dict['train'][omic + '_mask'] = data_dict['train'][omic + '_mask'][new_features]
        data_dict['valid'][omic + '_mask'] = data_dict['valid'][omic + '_mask'][new_features]
        data_dict['test'][omic + '_mask'] = data_dict['test'][omic + '_mask'][new_features]
    return data_dict

def drop_partial():
    cancer_types = list(set([f.split("_")[0] for f in sorted(os.listdir('./data/merged_splits'))]))
    for c in cancer_types:
        ORIGINAL_FILE = "./data/merged_splits/" + c + "_10fold.tsv"
        DROP_PATH = "./data/partial_check/" + c + "_10fold.csv"
        df = pd.read_csv(ORIGINAL_FILE, sep="\t", header=0, index_col=0)
        df = df[['Cli@Days2Death', 'Cli@Days2FollowUp', 'Cli@Censored', 'Fold@10CV']]
        df.to_csv(DROP_PATH, sep=",", index_label='Samples', encoding="utf8", na_rep='NA')

def check_validity():
    cancer_types = list(set([f.split("_")[0] for f in sorted(os.listdir('./data/merged_splits'))]))
    for c in cancer_types:
        error_samples = 0
        ORIGINAL_FILE = "./data/merged_splits/" + c + "_10fold.tsv"
        df = pd.read_csv(ORIGINAL_FILE, sep="\t", header=0, index_col=0)
        error1, error2, error3, error4 = 0, 0, 0, 0

        for _, row in df.iterrows():
            check_error = False
            if not(np.isnan(row['Cli@Days2Death']) and not np.isnan(row['Cli@Days2FollowUp'])):
                error1 += 1
                check_error = True
            if not(not np.isnan(row['Cli@Days2Death']) and np.isnan(row['Cli@Days2FollowUp'])):
                error2 += 1
                check_error = True
            if not (np.isnan(row['Cli@Days2Death']) and row['Cli@Censored'] == 1):
                error3 += 1
                check_error = True
            if not (np.isnan(row['Cli@Days2FollowUp']) and row['Cli@Censored'] != 1):
                error4 += 1
                check_error = True
            if check_error:
                error_samples += 1

        print(c, error1, error2, error3, error4)
        print(c, error_samples)


if __name__ == '__main__':
    # get_dataset()
    get_statistics('BRCA')

    # drop_partial()
    # cancer_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'UVM',
    #    'CHOL', 'COAD', 'COADREAD', 'DLBC', 'ESCA',
    #    'GBM', 'GBMLGG', 'HNSC', 'KICH', 'KIPAN',
    #    'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC',
    #    'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD',
    #    'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM',
    #    'STAD', 'STES', 'TGCT', 'THCA', 'THYM',
    #    'UCEC', 'UCS']

    # for c in cancer_list:
    #     get_statistics(c)


    # col_size = data_pack['train']['mRNA@'].shape[1]
    #
    # fea_sel = {
    #     'Methyl@': {
    #         'method': SelectKBest,
    #         'num_or_perc': 1000,
    #         'external': None
    #     }
    # }
    #
    # temp = get_dataset(gcn_mode=True, fea_sel=fea_sel)
    # x = 0





