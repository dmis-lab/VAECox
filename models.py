import torch
import torch.nn as nn
import torch.nn.functional as F
import vae_models as torch_maven
import numpy
# from pygcn.layers import GraphConvolution
import logging
import numpy as np

class PartialNLL(nn.Module):
    def __init__(self):
        super(PartialNLL, self).__init__()

    def forward(self, theta, R, censored):
#        theta = theta.double()
#        R = R.double()
#        censored = censored.double()

        exp_theta = torch.exp(theta)
        # observed = (censored.size(0) - torch.sum(censored)).cuda()
        observed = 1 - censored

        # observed = ipcw_weights

        num_observed = torch.sum(observed).cuda()
        loss = -(torch.sum((theta.reshape(-1)- torch.log(torch.sum((exp_theta * R.t()), 0))) * observed) / num_observed)
#        loss = loss.float()

        if np.isnan(loss.data.tolist()):
            for a,b in zip(theta, exp_theta):
                print(a,b)

        return loss


class CoxRegression(nn.Module):
    def __init__(self, nfeat):
        super(CoxRegression, self).__init__()
        self.fc1 = nn.Linear(nfeat, 1)
        self.init_hidden()

    def forward(self, x, coo=None):
        x = self.fc1(x)
        return x

    def init_hidden(self):
        nn.init.xavier_normal_(self.fc1.weight)


class Coxnnet(nn.Module):
    def __init__(self, nfeat):
        super(Coxnnet, self).__init__()
        self.fc1 = nn.Linear(nfeat, int(numpy.ceil(nfeat ** 0.5)))
        self.fc2 = nn.Linear(int(numpy.ceil(nfeat ** 0.5)), 1)

    def forward(self, x, coo=None):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def init_hidden(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


class CoxMLP(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(CoxMLP, self).__init__()

        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, 1)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        return x

    def init_hidden(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)


class VAECox(nn.Module):
    def __init__(self, config, logger, dropout, gcn_mode, use_gpu, pretrained):
        super(VAECox, self).__init__()
        # Check number of features!
        # torch.load(" <file_path> ")["model_state_dict"])

        num_features = 20502
        if config.gcn_mode: num_features = 12906
        #self.vae = torch_maven.VAE(num_features=num_features, gcn_mode=gcn_mode)
        #self.vae = torch_maven.VAE(config, logger, gcn num_features=20531)
        #self.vae = torch_maven.VAE(config, logger, num_features, gcn_mode)
        self.vae = torch_maven.VAE(config, logger, num_features)
        gpu_load = torch.load(pretrained, map_location=torch.device('cpu'))
        if use_gpu: gpu_load = torch.load(pretrained)
        self.vae.load_state_dict(gpu_load["model_state_dict"])

        for param in self.vae.parameters():
            param.requires_grad = True

        # self.cox = CoxRegression(256, 0.5)
        #self.cox = CoxMLP(256, 100, dropout)
        self.cox = Coxnnet(128)

    def forward(self, x, coo=None):
        #x = self.vae.dimension_reduction(x, coo)
        #x = self.cox(x)
        h = self.vae.encode(x)
        xm = self.vae.encode_mu(h)
        x = self.cox(xm)
        return x

class DAECox(nn.Module):
    def __init__(self, config, logger, dropout, gcn_mode, use_gpu, pretrained):
        super(DAECox, self).__init__()
        # Check number of features!
        # torch.load(" <file_path> ")["model_state_dict"])
        num_features = 20502
        if config.gcn_mode: num_features = 12906

        # self.ae = torch_maven.AE(num_features=num_features, gcn_mode=gcn_mode)
        # self.ae = torch_maven.AE(config, logger, num_features, gcn_mode)
        self.dae = torch_maven.DAE(config, logger, num_features)
        gpu_load = torch.load(pretrained, map_location=torch.device('cpu'))
        if use_gpu: gpu_load = torch.load(pretrained)
        self.dae.load_state_dict(gpu_load["model_state_dict"])

        # self.cox = CoxRegression(256, 0.5)
        #self.cox = CoxMLP(256, 100, dropout)
        # self.cox = CoxMLP(128, 100, dropout)
        self.cox = Coxnnet(128)

        for param in self.dae.parameters():
            param.requires_grad = True

    def forward(self, x, coo=None):
        x = self.dae.encode(x)
        #x = self.ae.dimension_reduction(x, coo)
        x = self.cox(x)
        return x


class AECox(nn.Module):
    def __init__(self, config, logger, dropout, gcn_mode, use_gpu, pretrained):
        super(AECox, self).__init__()
        # Check number of features!
        # torch.load(" <file_path> ")["model_state_dict"])
        num_features = 20502
        if config.gcn_mode: num_features = 12906

        # self.ae = torch_maven.AE(num_features=num_features, gcn_mode=gcn_mode)
        # self.ae = torch_maven.AE(config, logger, num_features, gcn_mode)
        self.ae = torch_maven.AE(config, logger, num_features)
        gpu_load = torch.load(pretrained, map_location=torch.device('cpu'))
        if use_gpu: gpu_load = torch.load(pretrained)
        self.ae.load_state_dict(gpu_load["model_state_dict"])

        # self.cox = CoxRegression(256, 0.5)
        #self.cox = CoxMLP(256, 100, dropout)
        self.cox = CoxMLP(128, 100, dropout)

        for param in self.ae.parameters():
            param.requires_grad = True

    def forward(self, x, coo=None):
        x = self.ae.encode(x)
        #x = self.ae.dimension_reduction(x, coo)
        x = self.cox(x)
        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        #return F.log_softmax(x, dim=1)
        return x


class VAECox_test(nn.Module):
    def __init__(self, config, logger, dropout, pretrained):
        super(VAECox_test, self).__init__()

        num_features = 15574
        if config.gcn_mode: num_features = 12906

        self.vae = torch_maven.VAE(config, logger, num_features)

        for param in self.vae.parameters():
            param.requires_grad = True

        self.cox = CoxMLP(128, 100, dropout)

        self.load_state_dict(torch.load(pretrained))

    def forward(self, x, coo=None):
        h = self.vae.encode(x)
        mu = self.vae.encode_mu(h)
        x = self.cox(mu)
        return x, mu


if __name__ == "__main__":
    vae = torch_maven.VAE(num_features=15574)
    vae.load_state_dict(torch.load("../data/vae_models/pan_cancer_mRNA_15%/model_600")["model_state_dict"])
    print(vae)
