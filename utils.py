import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import logging
import time
import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#from maven_utils import *

class SurvivalDataLoader(object):
    def __init__(self, cancertype, mean=False, common_feat=False, gcn_feat=False):
        self.cancertype = cancertype
        self.mean = mean

        self.logger = BKLogger.logger()

        self.datadir = "./data/TCGA_Integrated_Merged_5Fold_190105"
        self.preprocessed_dir = "./data/TCGA_preprocessed/"

        filename = self.cancertype + "_5fold" + ("_mean" if self.mean else "") + ".tsv"


        """
        if common_feat:
            filepath = os.path.join(self.datadir, filename)
            print("Loading omics file..")
            self.rawdf = pd.read_csv(filepath, sep="\t", index_col=0)
            print("Preprocessing omics file.. ")
            self.rawdf = self._common_features(self.rawdf)
            if gcn_feat:
                with open("./network_genes_mRNA@.txt") as fo:
                    for line in fo.readlines():
                        gene_list = line.split("\t")
                    # Check!
                    omic = "mRNA@"
                    print("Loading Biogrid..")
                    net = pickle.load(open("./network/networkx_biogrid.pickle", "rb"))
                    gene_features = [g.replace(omic, "") for g in gene_list if omic in g and "?" not in g]
                    gene_features = [g.split("|")[0] for g in gene_features]
                    net = net.subgraph(gene_features)
                    idx_name_dict = {i: name for i, name in enumerate(net.nodes())}
                    adj = np.array(nx.adjacency_matrix(net).todense()).astype(dtype='float', casting='same_kind')
                    self.coo = make_coo_matrix(adj)
                    # Check!
                    gene_list += ['Cli@Days2Death', 'Cli@Days2FollowUp', 'Cli@Censored', 'Fold@CV']
                self.rawdf = self.rawdf[gene_list]
            self.rawdf = self._z_normalization(self.rawdf)
            # self.preprocessing()
        else:
            self.rawdf = pickle.load(open('./data/ember_omics/' + 'imputed_and_binary_{0}.pickle'.format(cancertype), 'rb'))[0]
            # self.rawdf = pickle.load(open('./data/' + 'imputed_and_binary_{0}.pickle'.format(cancertype), 'rb'))[0]
            # print(self.rawdf.shape)
            print('Embernomics ' + cancertype)
        """
        #self.rawdf = pickle.load(open('./data/ember_omics/' + 'imputed_and_binary_{0}.pickle'.format(cancertype), 'rb'))[0]
        self.rawdf = pickle.load(open('./data/' + 'imputed_and_binary_{0}.pickle'.format(cancertype), 'rb'))[0]
        # print(self.rawdf.shape)
        # data has nan ?
        #print(self.rawdf.isnull().values.any())

    def preprocessing(self):
        filename = self.cancertype + "_5fold" + ("_mean" if self.mean else "") + ".tsv"
        filepath = os.path.join(self.datadir, filename)
        targetpath = os.path.join(self.preprocessed_dir, filename)

        if os.path.exists(targetpath):
            return

        start = time.time()
        self.rawdf = pd.read_csv(filepath, sep="\t", index_col=0)
        self.logger.debug("filename:{}, shape:{}, loading time:{:.2f}s"
                .format(filename, self.rawdf.shape, time.time() - start))

        self.rawdf = self._drop_missing_features(self.rawdf)
        self.rawdf = self._z_normalization(self.rawdf)


    def _common_features(self, rawdf):
        common_df = pickle.load(open("./data/TCGA_Integrated_Merged_5Fold_190105/pickle/pan_cancer_mRNA_15%.pickle", "rb"))
        common_feat_list = list(common_df.columns)

        rawdf = rawdf.loc[:,common_feat_list]
        rawdf = rawdf.fillna(rawdf.mean(axis=1))
        return rawdf

    # def _extract_network_features(self, rawdf):

    def _drop_missing_features(self, rawdf, threshold=0.3):
        col_missingrates = rawdf.isnull().sum(axis=0) / rawdf.shape[0]
        drop_cols = []
        for c in col_missingrates.index:
            if ("Cli@" in c) or ("Fold@" in c): continue;
            if col_missingrates[c] > threshold:                     # discard features which has more than 30% missings
                drop_cols.append(c)
            if rawdf[c].std() == 0:
                drop_cols.append(c)

        rawdf = rawdf.drop(drop_cols, axis=1)
        self.logger.debug("filename:{}, shape:{}".format(filename, rawdf.shape))
        self.logger.debug("Dropped Features {}".format(len(drop_cols)))
        return rawdf

    def _z_normalization(self, rawdf):
        start = time.time()
        for c in rawdf.columns:
            if ("Cli@" in c) or ("Fold@" in c): continue;
            tmpl = rawdf[c]
            tmpl = tmpl.fillna(tmpl.mean())                         # Impute nan with feature value mean
            tmpl = (tmpl - tmpl.mean()) / tmpl.std()                # Feature wise z normalization
            rawdf[c] = tmpl

        return rawdf

    def _analysis(self):
        features = self.rawdf.columns
        feat_tokens = [v.split("@") for v in features]

        feat_type_counts = dict()
        for f in feat_tokens:                                       # f[0] is feature type (miRNA, RNA ...)
            if f[0] in feat_type_counts:
                feat_type_counts[f[0]]+=1
            else:
                feat_type_counts[f[0]]=1

        self.logger.debug(" ".join(["{}:{}".format(k, v) for k,v in feat_type_counts.items()]))

    def get_split_ember(self, seed_num):
        selected_cols = []
        for c in self.rawdf.columns:
            for pre in ["mRNA@", "Fold@", "Cli@"]:
                if pre in c:
                    selected_cols.append(c)
        df = self.rawdf[selected_cols]

        clinical_day = df.loc[:, 'Cli@Days2Death'].fillna(0).values + df.loc[:, 'Cli@Days2FollowUp'].fillna(0).values
        day = np.argsort(clinical_day)
        day_stratified = np.zeros(len(day))
        days = len(day) / 5
        day_stratified[day < days] = 0
        day_stratified[(day >= days) & (day < 2 * days)] = 1
        day_stratified[(day >= 2 * days) & (day < 3 * days)] = 2
        day_stratified[(day >= 3 * days) & (day < 4 * days)] = 3
        day_stratified[day >= 4 * days] = 4
        fold_train, fold_test = train_test_split(df, test_size=0.2, random_state = seed_num, stratify=day_stratified)

        train_data = self._get_x_y_censored_ember(fold_train)
        test_data = self._get_x_y_censored_ember(fold_test)

        train_dataset = SurvivalDataset(train_data[:3])
        test_dataset = SurvivalDataset(test_data[:3])

        return train_dataset, test_dataset, fold_train.index[train_data[3]], fold_test.index[test_data[3]], train_data[
            -1], test_data[-1]

    def get_split_dataset(self, seed_num):
        selected_cols = []
        for c in self.rawdf.columns:
            for pre in ["mRNA@", "Fold@", "Cli@"]:
                if pre in c:
                    selected_cols.append(c)
        df = self.rawdf

        # clinical_day = df.loc[:, 'Cli@Days2Death'].fillna(0).values + df.loc[:, 'Cli@Days2FollowUp'].fillna(0).values
        clinical_day = df.loc[:, 'survival'].fillna(0).values
        day = np.argsort(clinical_day)
        day_stratified = np.zeros(len(day))
        days = len(day) / 5
        day_stratified[day < days] = 0
        day_stratified[(day >= days) & (day < 2 * days)] = 1
        day_stratified[(day >= 2 * days) & (day < 3 * days)] = 2
        day_stratified[(day >= 3 * days) & (day < 4 * days)] = 3
        day_stratified[day >= 4 * days] = 4
        fold_train, fold_test = train_test_split(df, test_size=0.2, random_state = seed_num, stratify=day_stratified)

        train_data = self._get_x_y_censored_maven(fold_train)
        test_data = self._get_x_y_censored_maven(fold_test)

        train_dataset = SurvivalDataset(train_data)
        test_dataset = SurvivalDataset(test_data)

        return fold_train, train_dataset, test_dataset, fold_train, fold_test

    def get_split_df(self, seed_num):
        selected_cols = []
        for c in self.rawdf.columns:
            for pre in ["mRNA@", "Fold@", "Cli@"]:
                if pre in c:
                    selected_cols.append(c)
        df = self.rawdf[selected_cols]

        clinical_day = df.loc[:, 'Cli@Days2Death'].fillna(0).values + df.loc[:, 'Cli@Days2FollowUp'].fillna(0).values
        day = np.argsort(clinical_day)
        day_stratified = np.zeros(len(day))
        days = len(day) / 5
        day_stratified[day < days] = 0
        day_stratified[(day >= days) & (day < 2 * days)] = 1
        day_stratified[(day >= 2 * days) & (day < 3 * days)] = 2
        day_stratified[(day >= 3 * days) & (day < 4 * days)] = 3
        day_stratified[day >= 4 * days] = 4
        fold_train, fold_test = train_test_split(df, test_size=0.2, random_state = seed_num, stratify=day_stratified)

        return fold_train, fold_test

    def get_shuffle_df(self, fold_train, fold_test):
        train_data = self._get_x_y_censored(fold_train)
        test_data = self._get_x_y_censored(fold_test)

        train_dataset = SurvivalDataset(train_data)
        test_dataset = SurvivalDataset(test_data)

        return train_dataset, test_dataset

    def get_split_fold_dataset(self, dataset, foldnum_list, v):
        train_index = []
        for foldnum in range(len(foldnum_list)):
            if foldnum != v: train_index.append(foldnum_list[foldnum])
        train_index = np.hstack(train_index)

        fold_train = dataset.iloc[train_index]
        fold_valid = dataset.iloc[foldnum_list[v]]

        train_data = self._get_x_y_censored_maven(fold_train)
        valid_data = self._get_x_y_censored_maven(fold_valid)

        train_dataset = SurvivalDataset(train_data)
        valid_dataset = SurvivalDataset(valid_data)

        return train_dataset, valid_dataset

    def get_fold_dataset(self, foldnum):
        selected_cols = []
        for c in self.rawdf.columns:
            for pre in ["mRNA@", "Fold@", "Cli@"]:
                if pre in c:
                    selected_cols.append(c)
        df = self.rawdf[selected_cols]

        fold_train = df[(df["Fold@CV"] != foldnum)]
        fold_valid = df[df["Fold@CV"] == foldnum]

        self.logger.debug("Train:{}, Valid:{}".format(fold_train.shape, fold_valid.shape))
        train_data = self._get_x_y_censored(fold_train)
        valid_data = self._get_x_y_censored(fold_valid)

        train_dataset = SurvivalDataset(train_data)
        valid_dataset = SurvivalDataset(valid_data)

        return train_dataset, valid_dataset

    def _get_x_y_censored_maven(self, df):
        # print(df.shape)
        y = np.array(df['survival'])
        censored = np.array(df['censored'])
        x = np.array(df.drop(columns=['censored', 'survival']).values)
        return x, y, censored

    def _get_x_y_censored(self, df):
        censored = []
        y = []
        print(df.head())
        for d1, d2 in zip(df["Cli@Days2Death"], df["Cli@Days2FollowUp"]):
            if np.isnan(d1):                                        # Death time is not observed
                y.append(d2)                                        # Using followup time
                censored.append(1)                                  # Censored data
            else:
                y.append(d1)
                censored.append(0)

        ## original ver.
        # x = df.drop(["Cli@Days2Death", "Cli@Days2FollowUp", "Cli@Censored", "Fold@CV"], axis=1)
        x = df
        #print(x.isnull().values.any())

        usable_idx_list = [idx for idx in range(x.shape[0]) if (not np.isnan(y[idx]) and (y[idx]!=0))]

        x = x.values.astype(float)[usable_idx_list]
        y = np.array(y)[usable_idx_list]
        censored = np.array(censored)[usable_idx_list]

        return x, y, censored

    def _get_x_y_censored_ember(self, df):
        censored = []
        y = []

        for d1, d2 in zip(df["Cli@Days2Death"], df["Cli@Days2FollowUp"]):
            if np.isnan(d1):  # Death time is not observed
                y.append(d2)  # Using followup time
                censored.append(1)  # Censored data
            else:
                y.append(d1)
                censored.append(0)

        x_drop = df.loc[:, "Cli@Days2Death":"Fold@CV"]
        x = df.drop(["Cli@Days2Death", "Cli@Days2FollowUp", "Cli@Censored", "Fold@CV"], axis=1)
        # print(x.isnull().values.any())

        usable_idx_list = [idx for idx in range(x.shape[0]) if (not np.isnan(y[idx]) and (y[idx] != 0))]

        x = x.values.astype(float)[usable_idx_list]
        x_drop = x_drop.iloc[usable_idx_list, :]
        y = np.array(y)[usable_idx_list]
        censored = np.array(censored)[usable_idx_list]

        return x, y, censored, usable_idx_list, x_drop

class SurvivalDataset(Dataset):
    def __init__(self, data):
        self.x, self.y, self.c = data
        self.R = _make_R(self.y)

        self.indices = list(range(self.x.shape[0]))
        self.num_feat = self.x.shape[1]

        #print(self.x.shape, self.y.shape, self.R.shape)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # print(self.indices)
        # print(self.x.shape)
        # print(self.y.shape)
        # print(self.c.shape)
        return self.x[idx], self.y[idx], self.c[idx], self.indices[idx] # indices -> for get R matrix

def _make_R(y):
    R = np.zeros((y.shape[0], y.shape[0]))
    for i in range(y.shape[0]):
        for j in range(y.shape[0]):
            R[i,j] = (y[j] >=y[i])
    return R

class EarlyStopper():
    def __init__(self, prev=0, th=0.00000001):
        self.prev = prev
        self.th = th

    def __call__(self, ep, loss):
        a = np.abs(self.prev - loss) < self.th                      # Loss gap is smaller than threshold
        b = np.isnan(loss)                                          # Loss has nan(inf)
        c = (ep > 100) and (loss > 3*self.prev)                     # Loss is more than three times of previous loss(explode)

        self.prev = loss
        return (a or b or c)

class BKLogger(object):
    __logger = None

    @classmethod
    def __getLogger(cls):
        return cls.__logger

    @classmethod
    def logger(cls, logdir, logname):
        cls.__logger = cls.__setlogger(logdir, logname)
        cls.logger = cls.__getLogger
        return cls.__logger

    @classmethod
    def __setlogger(cls, logdir, logname):
        os.makedirs("{}/{}".format(logdir, logname), exist_ok=True)

        logger = logging.getLogger(logname)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')
        fileHandler = logging.FileHandler("{}/{}/lo.g".format(logdir, logname))
        streamHandler = logging.StreamHandler()

        fileHandler.setFormatter(formatter)
        streamHandler.setFormatter(formatter)

        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)

        return logger


class SingletonInstane(object):
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kwargs):
        cls.__instance = cls(*args, **kwargs)
        cls.instance = cls.__getInstance
        return cls.__instance



if __name__ == "__main__":
    dl = DataLoader("CHOL")
    t, v = dl.get_fold_dataset(1)
    print(t[1:4])

# import pickle
# data = pickle.load(open('imputed_and_binary_BLCA.pickle', 'rb'))[0]
# data.shape
