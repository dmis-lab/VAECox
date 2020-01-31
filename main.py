import os
import torch
import copy
import numpy as np
from multiprocessing import Pool
from utils import *
from models import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
# import setproctitle
import json
from tqdm import tqdm, trange
import argparse
import itertools
import random
import datetime
import logging
from os.path import exists, join
from lifelines import CoxPHFitter
from multiprocessing import set_start_method

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

torch.multiprocessing.set_start_method("spawn", force=True)
try:
    set_start_method('spawn')
except RuntimeError:
    pass

logger = None                                                       # Global Logger

def main():
    start = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('--model_struct', '-mst', type=str, default='basic')
    parser.add_argument('--cancer_object', '-co', type=str, default='BLCA')
    parser.add_argument("--cancertype", type=str, default="KIRC")
    parser.add_argument("--model", '-md', type=str, default="VAECox")
    parser.add_argument("--time_str", type=str, default="00000000")
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--random_split", '-rs', default=False, action='store_true')
    parser.add_argument('--batch_size', '-bs', default=1000, type=int)
    # find path to DAE
    parser.add_argument('--pretrained', '-pr', default='results/vae/vae_pretrained/vae/final_model', type=str)
    parser.add_argument('--weight_sparsity', type=int, default=0)
    parser.add_argument('--pool_func', '-pf', type=str, default='None')
    parser.add_argument('--eval', '-ev', default=False, action='store_true')
    parser.add_argument('--test', '-ts', default=False, action='store_true')
    parser.add_argument('--reimple', '-ri', default=False, action='store_true')
    parser.add_argument('--shuffle', '-sh', default=False, action='store_true')
    parser.add_argument('--shuffle_model', '-sl', default=0, type=int)
    parser.add_argument('--embedding_output', '-eo', default=False, action='store_true')
    parser.add_argument('--embedding_train', '-et', default=False, action='store_true')
    parser.add_argument('--common_feat', '-cf', default=False, action='store_true')

    parser.add_argument('--hidden_nodes', '-hn', type=int, default=4096)
    parser.add_argument('--gcn_func', '-gcf', default='None', type=str)
    parser.add_argument('--cancer_list', '-cl', type=str, default='coxnnet')
    parser.add_argument('--omic_list', '-ol', nargs='+', type=str)
    parser.add_argument('--missing_impute', '-mi', type=str, default='mean')
    parser.add_argument('--exclude_impute', '-xi', default=False, action='store_true')
    parser.add_argument('--feature_scaling', '-fc', type=str, default='None')
    parser.add_argument('--feature_selection', '-fs', type=str, default='None')
    parser.add_argument('--gcn_mode', '-gcn', default=False, action='store_true')
    parser.add_argument('--ipcw_mode', '-ipcw', default=False, action='store_true')
    parser.add_argument('--device_type', '-dv', type=str, default='cuda')
    parser.add_argument('--cuda_device', '-cd', type=str, default='0')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6)
    parser.add_argument('--max_epochs', '-mx', type=int, default=500)
    parser.add_argument('--model_optimizer', '-mo', type=str, default='Adam')
    parser.add_argument('--dropout_rate', '-dr', type=float, default=0.0)
    parser.add_argument('--multi_task', '-mu', default=False, action='store_true')
    parser.add_argument('--mt_regularization', '-mr', default='None', type=str)
    parser.add_argument('--num_clusters', '-nc', default=8, type=int)
    parser.add_argument('--augment_autoencoder', '-aug', default='None', type=str)
    parser.add_argument('--deseq2', '-deseq', default=False, action='store_true')
    parser.add_argument('--acti_func', '-af', default="Tanh", type=str)
    # parser.add_argument('--file_version', '-fv', type=str, default='15%')
    parser.add_argument('--hp_search', '-hs', default=False, action='store_true')
    parser.add_argument('--vae_data', '-vd', default='ember_libfm_190511', type=str)
    parser.add_argument('--test_mode', '-tm', default=False, action='store_true')
    parser.add_argument('--model_type', '-mt', default='ae', type=str)
    parser.add_argument('--save_mode', '-sm', default=False, action='store_true')
    parser.add_argument('--checkpoint_dir', '-cp', default='./results/', type=str)
    parser.add_argument('--session_name', '-sn', default='test', type=str)
    parser.add_argument('--pickle_save', '-ps', default=False, action='store_true')
    #parser.add_argument('--gcn_func', '-gf', default='ChebConv', type=str)

    config = parser.parse_args()

    cancertype = config.cancertype
    model = config.model
    dropout_rate = config.dropout_rate
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    use_gpu = config.use_gpu
    epochs = config.epochs
    gcn_mode = config.gcn_mode
    test = config.test

    os.makedirs("./results/", exist_ok=True)
    os.makedirs("./saved_model/", exist_ok=True)

    ''' Ember Edition '''

    exp_code = config.time_str + "/" + "-".join([cancertype, model])

    global logger
    logger = BKLogger.logger(logdir="results", logname=exp_code)


    if not config.eval:
        ten_cancer = ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD']
        param_dict = {'dropout_rate': [0.0, 0.3, 0.5], 'learning_rate': [1e-4, 1e-3],
                      'weight_decay': [1e-3, 1e-4, 1e-5]}

        do_search(ten_cancer, param_dict, config, model, use_gpu, epochs, gcn_mode, config.batch_size, test)

    else:
        pretrained = config.pretrained_file
        if len(config.pretrained_file)==0:
            print("No pretrained file")
            exit()
        do_test(pretrained, config, use_gpu, gcn_mode)

"""
    # python main.py -rs
    if config.random_split:

        ten_cancer = ['BLCA', 'BRCA', 'HNSC', 'KIRC', 'LGG', 'LIHC', 'LUAD', 'LUSC', 'OV', 'STAD']
        # ten_cancer = ten_cancer[-4:]
        # ten_cancer = ten_cancer[4:6]
        # ten_cancer = [config.cancer_object]
        param_dict = {'dropout_rate': [0.0, 0.3, 0.5], 'learning_rate': [1e-4, 1e-3],
                      'weight_decay': [1e-3, 1e-4, 1e-5]}

        do_search(ten_cancer, param_dict, config, model, use_gpu, epochs, gcn_mode, config.batch_size, test)

    # reimplement the best model
    # python main.py -ri
    # or
    # python main.py -ri -eo (-et)
    elif config.reimple:

        pretrained_path = './saved_model/pan_vae_best_noember_new/'
        best_model_list = os.listdir(pretrained_path)
        best_model_list = [data for data in best_model_list if 'pan_' in data]
        best_model_list.sort()
        pre_model = best_model_list[config.shuffle_model]
        pretrained = pretrained_path + pre_model

        # python main.py -ri -sh -pr 0
        if config.shuffle:

            print('Preparing shuffle test')
            f = open('./results/shuffle_test/shuffle_pan_vae_{}.tsv'.format(config.shuffle_model), 'a')
            f.write('\n' + pre_model)
            f.close()
            do_test(pretrained, config, use_gpu, gcn_mode)

        else:

            print('Preparing reimplementation')
            best_model_listt = best_model_list
            for i in range(len(best_model_listt)):
                pretrained = pretrained_path + best_model_listt[i]
                if 'pan_' in pretrained.split('/')[-1]: do_test(pretrained, config, use_gpu, gcn_mode)

    # else:
    #
    #     dl = SurvivalDataLoader(cancertype, common_feat=config.common_feat, gcn_feat=gcn_mode)
    #     if dl is None:
    #         return
    #
    #     multiargs = [
    #         (v, config, cancertype, *dl.get_fold_dataset(v), exp_code, model, dropout_rate, learning_rate, weight_decay, batch_size, use_gpu, epochs, gcn_mode) for v in range(5)]
    #
    #     resultdf = do_5cv(multiargs, exp_code, model, epochs)
    #
    #     gcn_check = '_'
    #     if gcn_mode: gcn_check = '_gcn_'
    #     resultdf_name = '-'.join([cancertype, model]) + gcn_check + '_'.join(
    #         [str(epochs), str(learning_rate), str(weight_decay)])
    #     resultdf.to_csv('results/' + resultdf_name + '.tsv', sep='\t')
    #     print(resultdf.tail(1))

    #bbs = sunkyu.BeaverBotSender()
    #bbs.send("Multiomics - survival analysis", "Experiment : {} \n{}\nEND\nTime:{:.4f}min".format(exp_code, str(resultdf.iloc[-1]), (time.time()-start)/60))
"""


def do_search(ten_cancer, param_dict, config, model, use_gpu, epochs, gcn_mode, batch_size, test):

    it = 10
    if test:
        it = 1
        param_dict = {'dropout_rate': [0.0], 'learning_rate': [1e-4], 'weight_decay': [1e-3]}

    param_combi = list(itertools.product(*(param_dict[Key] for Key in sorted(param_dict))))
    param_combi_count = len(param_combi)

    for cancer in ten_cancer:

        dl = SurvivalDataLoader(cancer, common_feat=config.common_feat, gcn_feat=gcn_mode)
        if dl is None:
            return

        '''
        if cancer == 'BLCA': param_combi = [(0.3, 1e-4, 1e-4),(0.5, 1e-4, 1e-4),(0.3, 1e-4, 1e-3),(0.5, 1e-4, 1e-3)]
        '''

        cancertype = cancer
        exp_code = config.time_str + "/" + "-".join([cancertype, model])

        #logger = sunkyu.BKLogger.logger(logdir="results", logname=exp_code)

        if model == 'coxnnet':
            model_name = '1.coxnnet'
        elif model == 'coxLasso':
            model_name = '1.coxlasso'
        elif model == 'coxRidge':
            model_name = '1.coxridge'
        elif model == 'coxMLP':
            model_name = '2.coxMLP'
        elif model == 'AECox':
            model_name = '3.AECox'
        elif model == 'VAECox':
            model_name = '4.VAECox'
        elif model == 'DAECox':
            model_name = '5.DAECox'

        coo = 0
        if gcn_mode:
            coo_list = []
            for b in range(256):
                temp = (b * 256) + torch.tensor(dl.coo).long().to(torch.device('cuda'))
                coo_list.append(temp)
            coo =torch.cat(coo_list, 1)

        mean_highest_ci, mean_last_ci = 0, 0
        for i in range(it):
            train_set, train_dataset, test_dataset, fold_train, fold_test = dl.get_split_dataset(seed_num=i)  # split the data
            cv_list = []
            if ('AECox' in model):
                fold_train.to_csv('./saved_model/{}_{}_train.tsv'.format(cancertype, i), sep='\t')
                fold_test.to_csv('./saved_model/{}_{}_test.tsv'.format(cancertype, i), sep='\t')

            # split for 5CV by KFold
            cv = KFold(5, shuffle=True, random_state=i)
            for idx, (idx_train, idx_test) in enumerate(cv.split(train_set)):
                cv_list.append(idx_test)
# rawdf
            # hyperparamter fitting
            best_ci, combi_count = 0, 0
            for param in param_combi:
                dropout_rate, learning_rate, weight_decay = param[0], param[1], param[2]
                combi_count += 1

                dl.logger.debug(
                    "cancer: {}, count:{} of {}, param_combi_count:{} of {}".format(cancertype, i + 1, it, combi_count,
                                                                                    param_combi_count))
                dl.logger.debug("dr:{}, lr:{}, wd:{}".format(dropout_rate, learning_rate, weight_decay))

                multiargs = [
                    (v, i, config, cancertype, *dl.get_split_fold_dataset(train_set, cv_list, v), exp_code, model,
                     dropout_rate, learning_rate, weight_decay, batch_size, use_gpu, epochs, gcn_mode, coo) for v in
                    range(5)]

                resultdf = do_5cv(multiargs, exp_code, model, epochs)
                highest_valid_ci = resultdf['valid_ci'].max()

                if best_ci < highest_valid_ci:
                    best_ci, best_param, best_result = highest_valid_ci, param, resultdf

            # test
            dropout_rate, learning_rate, weight_decay = best_param[0], best_param[1], best_param[2]
            test_args = [
                (-1, i, config, cancertype, train_dataset, test_dataset, exp_code, model, dropout_rate, learning_rate,
                 weight_decay, batch_size, use_gpu, epochs, gcn_mode, coo)]

            dl.logger.debug(
                "Test {}, cancer: {}, dropout:{}, lr:{}, wd:{}".format(i + 1, cancertype, dropout_rate, learning_rate,
                                                                       weight_decay))

            resultdf = do_5cv(test_args, exp_code, model, epochs)
            last_test_ci = resultdf['valid_ci'][epochs]

            highest_result = resultdf[resultdf['valid_ci'].max() == resultdf['valid_ci']]['valid_ci']
            highest_ep = highest_result.index[0]
            highest_test_ci = highest_result.values[0]

            mean_highest_ci += highest_test_ci
            mean_last_ci += last_test_ci

            with open('./results/test_ci_dae.tsv'.format(cancertype), 'a') as f:
                f.write('\n' + datetime.datetime.now().strftime("%Y%m%d") + '\t' + cancertype + '\t' + model_name)
                f.write('\t' + str(gcn_mode) + '\t' + str(round(highest_test_ci, 3)) + '\t' + str(highest_ep))
                f.write('\t' + str(round(last_test_ci, 3)) + '\t' + str(epochs))
                f.write('\t' + str(dropout_rate) + '\t' + str(learning_rate) + '\t' + str(weight_decay))
                f.write('\t' + str(i) + '\t' + 'True' + '\t' + config.pretrained)

        mean_highest_ci /= it
        mean_last_ci /= it

        with open('./results/test_ci_dae.tsv', 'a') as f:
            f.write('\n' + datetime.datetime.now().strftime("%Y%m%d") + '\t' + cancertype + '\t' + model_name)
            f.write('\t' + str(gcn_mode) + '\t' + str(round(mean_highest_ci, 3)) + '\t')
            f.write('\t' + str(round(mean_last_ci, 3)) + '\t\t\t\t\t'+ '-1'+ '\t' + 'True' + '\t' + config.pretrained + '\n')

        # remove others without the best model data
        if ('AECox' in model):
            try:
                best_model = [saved_model for saved_model in os.listdir('./saved_model') if
                          '{}_{}'.format(model_name, cancertype) in model][0]

                data_list = [data for data in os.listdir('./saved_model') if
                         (cancertype in data) and ('.tsv' in data) and (str(best_model.split('_')[-6]) not in data)]
                for data in data_list: os.remove('./saved_model/' + data)

            except: pass


def do_5cv(multiargs, exp_code, model, epochs):

    if model == "baseline":
        for arg in multiargs[:5]:
            do_baseline(*arg)

    if model == "VAECox" or model == "AECox" or model == "DAECox": pool = Pool(2)
    else: pool = Pool(20)

    pool.starmap(do_one_fold, multiargs)
    pool.close()

    # for arg in multiargs:
    #     do_one_fold(*arg)

    if len(multiargs) == 1:
        resultdf = get_test_score(exp_code, epochs)
    else:
        resultdf = get_score(exp_code, epochs)

    return resultdf


def do_one_fold(foldnum, it, config, cancertype, train, valid, exp_code, model_str, dropout_rate,
        learning_rate, weight_decay, batch_size, use_gpu, epochs, gcn_mode, coo=None):

    # setproctitle.setproctitle("COX_{}_{}".format(exp_code, foldnum))
    coo = None

    pgbar = trange(epochs, position=foldnum, desc="Fold{}".format(foldnum))
    train_dl = DataLoader(dataset=train, batch_size=config.batch_size, shuffle=True)
    valid_dl = DataLoader(dataset=valid, batch_size=10000, shuffle=False)
    LOGGER = logging.getLogger()

    # load pretrained model
    pretrained = config.pretrained

    lasso = False
    if (model_str == "coxRG") or (model_str == 'coxRidge'):
        model = CoxRegression(train.num_feat)
    elif model_str == 'coxLasso':
        model = CoxRegression(train.num_feat)
        lasso = True
    elif model_str == "coxMLP":
        model = CoxMLP(train.num_feat, 100, dropout=dropout_rate)
    elif model_str == "coxnnet":
        model = Coxnnet(train.num_feat)
    elif model_str == "VAECox":
        model = VAECox(config=config, logger=LOGGER, dropout=dropout_rate, gcn_mode=gcn_mode, use_gpu=use_gpu, pretrained=pretrained)
    elif model_str == "AECox":
        model = AECox(config=config, logger=LOGGER, dropout=dropout_rate, gcn_mode=gcn_mode, use_gpu=use_gpu, pretrained=pretrained)
    elif model_str == "DAECox":
        model = DAECox(config=config, logger=LOGGER, dropout=dropout_rate, gcn_mode=gcn_mode, use_gpu=use_gpu, pretrained=pretrained)
    else:
        raise Exception

    if use_gpu == True: model = model.cuda()

    model = model.double()

    loss_fn = PartialNLL()
    if lasso: weight_decay = 0
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


    train_result = dict()
    valid_result = dict()

    stopper = EarlyStopper()

    last_ep = 0
    for ep in range(epochs+1):
        train_result[ep] = model_train(ep, model=model, dataloader=train_dl,
                                      loss_function=loss_fn, optimizer=optim, get_ci=True, use_gpu=use_gpu, coo=coo,
                                      lasso=lasso)
        valid_result[ep] = model_valid(ep, model=model, dataloader=valid_dl,
                                       loss_function=loss_fn, get_ci=True, use_gpu=use_gpu, coo=coo)

        score = train_result[ep]['c-index']
        loss = train_result[ep]['loss']

        last_ep = ep
        # if stopper(ep, loss):
        #     break

        # valid_ci = valid_result[ep]['valid c-index']
        # if ep >= 9:
        #     if valid_ci - np.mean(valid_ci_list[int(round(len(valid_ci_list)/2)):]) <= 0.01: break
        # valid_ci_list.append(valid_ci)

        pgbar.set_postfix(score=score, loss=loss)
        pgbar.update()


    # get the last results
    valid_result[last_ep] = model_valid(last_ep, model=model, loss_function=loss_fn, dataloader=valid_dl, get_ci=True, coo=coo, use_gpu=use_gpu)

    # save the model
    if (foldnum == -1) and ('AECox' in model_str):

        if it == 0:
            PATH = "./saved_model/{}_{}_{}_{}_{}_{}_{}".format(cancertype, it,
                                                                     epochs,
                                                                     dropout_rate, learning_rate, weight_decay, round(valid_result[last_ep]['valid c-index'], 3))
            torch.save(model.state_dict(), PATH)

        else:
            try:
                model_list = [model for model in os.listdir('./saved_model') if '{}_{}'.format(model_str, cancertype) in model]

                if float(model_list[0].split('_')[-1]) < float(valid_result[last_ep]['valid c-index']):
                    os.remove('./saved_model/' + model_list[0])

                    data_list = [data for data in os.listdir('./saved_model') if (cancertype in data) and ('.tsv' in data) and (str(it) not in data)]
                    for data in data_list: os.remove('./saved_model/' + data)

                    PATH = "./saved_model/{}_{}_{}_{}_{}_{}_{}".format(cancertype, it, epochs,
                                                                         dropout_rate, learning_rate, weight_decay, round(valid_result[last_ep]['valid c-index'], 3))
                    torch.save(model.state_dict(), PATH)

            except: pass

    train_path = "results/{}".format(exp_code)
    if not os.path.isdir(train_path): os.mkdir(train_path)

    with open(train_path + "/train_{}.json".format(foldnum), "w") as fw:
        json.dump(train_result, fw)
    with open(train_path + "/valid_{}.json".format(foldnum), "w") as fw:
        json.dump(valid_result, fw)


def model_train(ep, model, dataloader, loss_function, optimizer, coo, use_gpu=True, print_every=10, get_ci=False,
                lasso=False):
    model.train()
    total_loss = 0.0
    total = 0.0

    for it, batch_data in enumerate(dataloader):
        batch_x, batch_y, batch_censored, batch_idx = batch_data
        try: batch_R = dataloader.dataset.R[batch_idx,:][:,batch_idx]
        except:
            print(batch_R.shape)
            print(batch_idx)

#        batch_R = torch.FloatTenQsor(batch_R)
#
#        batch_x = batch_x.type(torch.FloatTensor)
#        batch_y = batch_y.type(torch.FloatTensor)
#        batch_censored = batch_censored.type(torch.FloatTensor)
        batch_R = torch.DoubleTensor(batch_R)

        batch_x = batch_x.type(torch.DoubleTensor)
        batch_y = batch_y.type(torch.DoubleTensor)
        batch_censored = batch_censored.type(torch.DoubleTensor)
        if use_gpu:
            batch_x = Variable(batch_x.cuda())
            batch_y = Variable(batch_y.cuda())
            batch_R = batch_R.cuda()
            batch_censored = batch_censored.cuda()

        optimizer.zero_grad()
        model.batch_size = len(batch_x)
        theta = model(batch_x)

        loss = loss_function(theta, batch_R, batch_censored).cpu()
        if lasso:
            L1 = torch.nn.L1Loss()
            model_param = torch.cat([x.view(-1) for x in model.fc1.parameters()])
            loss = loss.cuda() + L1(model_param.cuda(), torch.zeros(model_param.size()).cuda().double())

        loss.backward()
        optimizer.step()

        total_loss += loss.data.tolist()
        total += len(batch_y)

    ci = None
    #if ep%print_every == 0:
    #logger.info("====Ep {}".format(ep))
    #logger.info("Training loss:\t{:.3e}".format(total_loss/total))
    event_observed = np.array([1 if v==0 else 0 for v in batch_censored])
    if ep%1000==0 or get_ci:
        batch_y = batch_y.cpu()
        theta = -theta.reshape(-1).data.cpu()
        ci = concordance_index(batch_y, theta, event_observed)

     #logger.info("Training C-index:\t{:.4f}".format(ci))
    return {"loss":total_loss, "c-index":ci}

def model_valid(ep, model, dataloader, loss_function, coo, use_gpu=True, get_ci=False):
    model.eval()
    total_loss = 0.0
    total = 0.0

    for it, batch_data in enumerate(dataloader):
        batch_x, batch_y, batch_censored, batch_idx = batch_data
        batch_R = dataloader.dataset.R[batch_idx,:][:,batch_idx]
#        batch_R = torch.FloatTensor(batch_R)
#
#        batch_x = batch_x.type(torch.FloatTensor)
#        batch_y = batch_y.type(torch.FloatTensor)
#        batch_censored = batch_censored.type(torch.FloatTensor)
        batch_R = torch.DoubleTensor(batch_R)

        batch_x = batch_x.type(torch.DoubleTensor)
        batch_y = batch_y.type(torch.DoubleTensor)
        batch_censored = batch_censored.type(torch.DoubleTensor)

        if use_gpu:
            batch_x = Variable(batch_x.cuda())
            batch_y = Variable(batch_y.cuda())
            batch_R = batch_R.cuda()
            batch_censored = batch_censored.cuda()

        model.batch_size = len(batch_x)
        theta = model(batch_x, coo)

        if type(theta) == tuple:
            theta = theta[0]
        loss = loss_function(theta, batch_R, batch_censored).cpu()
        total_loss += loss.data.tolist()
        total += len(batch_y)

    ci = None
    #if ep%print_every == 0:
    #logger.info("Testing loss:\t{:.3e}".format(total_loss/total))
    event_observed = np.array([1 if v==0 else 0 for v in batch_censored])

    if get_ci:
        batch_y = batch_y.cpu()
        theta = -theta.reshape(-1).data.cpu()
        ci = concordance_index(batch_y, theta, event_observed)
    #logger.info("Testing C-index:\t{:.4f}".format(ci))
    return {"valid loss": total_loss, "valid c-index": ci}


def model_valid_embedding(model, dataloader, coo, use_gpu=True):
    model.eval()

    for it, batch_data in enumerate(dataloader):
        batch_x, batch_y, batch_censored, batch_idx = batch_data
        batch_x = batch_x.type(torch.DoubleTensor)
        if use_gpu:
            batch_x = Variable(batch_x.cuda())

        model.batch_size = len(batch_x)
        theta, vector = model(batch_x, coo)

        return theta, vector


def get_score(exp_code, epochs, num_cv=5):
    results_dir = "results/{}/".format(exp_code)
    train_dict = dict()  # keys = num of iteration, values = scores
    valid_dict = dict()

    min_ep = epochs + 1
    for i in range(num_cv):
        print("")
        train_dict[i] = json.load(open(results_dir + "train_{}.json".format(i)))
        valid_dict[i] = json.load(open(results_dir + "valid_{}.json".format(i)))

    #        if min_ep > len(train_dict[i]):
    #            min_ep = len(train_dict[i])
    #        if min_ep > len(valid_dict[i]):
    #            min_ep = len(valid_dict[i])

    print("")
    result_ep = dict()
    for ep in trange(epochs+1, desc="Result aggregate"):
        train_loss = []
        train_ci = []
        valid_loss = []
        valid_ci = []

        for cv in range(num_cv):
            train_cv = train_dict[cv][str(ep)] if str(ep) in train_dict[cv] else train_dict[cv][
                str(len(train_dict[cv]) - 1)]
            valid_cv = valid_dict[cv][str(ep)] if str(ep) in valid_dict[cv] else valid_dict[cv][
                str(len(valid_dict[cv]) - 1)]

            train_loss.append(train_cv['loss'])
            train_ci.append(train_cv['c-index'])
            valid_loss.append(valid_cv['valid loss'])
            valid_ci.append(valid_cv['valid c-index'])

        train_loss = np.array(train_loss)
        train_ci = np.array(train_ci)
        valid_loss = np.array(valid_loss)
        valid_ci = np.array(valid_ci)

        result_dict = dict()

        if np.isnan(train_loss).any():
            break
        else:
            result_dict["train_loss"] = train_loss.mean()
            result_dict["valid_loss"] = valid_loss.mean()

        if train_ci[0] is not None:
            result_dict["train_ci"] = train_ci.mean()

        if valid_ci[0] is not None:
            result_dict["valid_ci"] = valid_ci.mean()

        result_ep[ep] = result_dict

    df = pd.DataFrame(result_ep).transpose()

    df.to_csv(os.path.join(results_dir, "result.tsv"), sep="\t", index_label="epochs")

    return df

def get_test_score(exp_code, epochs):
    results_dir = "results/{}/".format(exp_code)

    train_dict = dict()  # keys = num of iteration, values = scores
    valid_dict = dict()

    min_ep = epochs + 1
    print("")
    train_dict[0] = json.load(open(results_dir + "train_-1.json"))
    valid_dict[0] = json.load(open(results_dir + "valid_-1.json"))

    print("")
    result_ep = dict()
    for ep in trange(epochs+1, desc="Result aggregate"):
        train_loss = []
        train_ci = []
        valid_loss = []
        valid_ci = []

        train_cv = train_dict[0][str(ep)] if str(ep) in train_dict[0] else train_dict[0][
            str(len(train_dict[0]) - 1)]
        valid_cv = valid_dict[0][str(ep)] if str(ep) in valid_dict[0] else valid_dict[0][
            str(len(valid_dict[0]) - 1)]

        train_loss.append(train_cv['loss'])
        train_ci.append(train_cv['c-index'])
        valid_loss.append(valid_cv['valid loss'])
        valid_ci.append(valid_cv['valid c-index'])

        result_dict = dict()

        result_dict["train_loss"] = np.array(train_loss).mean()
        result_dict["valid_loss"] = np.array(valid_loss).mean()
        result_dict["train_ci"] = np.array(train_ci).mean()
        result_dict["valid_ci"] = np.array(valid_ci).mean()

        result_ep[ep] = result_dict

    df = pd.DataFrame(result_ep).transpose()

    df.to_csv(os.path.join(results_dir, "test_result.tsv"), sep="\t", index_label="epochs")

    return df

def do_baseline(foldnum, train, valid, exp_code, model_str):
    cph = CoxPHFitter()
    df = pd.DataFrame(train.x)
    print(df.shape)
    df['duration'] = train.y
    df['event'] = [1 if v == 0 else 0 for v in train.c]

    df = df.fillna(df.mean())
    cph.fit(df, 'duration', event_col="event")

    cph.print_summary()

    valid_df = pd.DataFrame(valid.x)
    valid_df = valid_df.fillna(valid_df.mean())
    print(cph.predict_log_partial_hazard(valid_df))


def do_test(pretrained, config, use_gpu, gcn_mode):

    model_str = pretrained.split('_')[-8]
    cancertype = pretrained.split('_')[-7]
    it = int(pretrained.split('_')[-6])
    exp_code = config.time_str + "/" + "-".join([cancertype, model_str])

    if model_str == 'coxnnet':
        model_name = '1.coxnnet'
    elif model == 'coxLasso':
        model_name = '1.coxlasso'
    elif model == 'coxRidge':
        model_name = '1.coxridge'
    elif model_str == 'coxMLP':
        model_name = '2.coxMLP'
    elif model_str == 'AECox':
        model_name = '3.AECox'
    elif model_str == 'VAECox':
        model_name = '4.VAECox'
    elif model_str == 'DAECox':
        model_name = '5.DAECox'

    dl = SurvivalDataLoader(cancertype, common_feat=config.common_feat, gcn_feat=gcn_mode)
    if dl is None:
        return

    coo = 0
    if gcn_mode:
        coo_list = []
        for b in range(256):
            temp = (b * 256) + torch.tensor(dl.coo).long().to(torch.device('cuda'))
            coo_list.append(temp)
        coo =torch.cat(coo_list, 1)

    foldnum = -1
    # setproctitle.setproctitle("COX_{}_{}".format(exp_code, foldnum))

    LOGGER = logging.getLogger()

    model = VAECox_test(config=config, logger=LOGGER, dropout=0, pretrained=pretrained)

    print(pretrained)
    if use_gpu == True: model = model.cuda()

    model = model.double()

    loss_fn = PartialNLL()
    valid_result = dict()
    last_ep = 0

    if config.shuffle:
        dl.logger.debug("Shuffle Test {}, cancer: {}".format(it + 1, cancertype))

        fold_train, fold_test = dl.get_split_df(seed_num=it)  # split the data
        feature_num = fold_test.shape[1]
        pgbar = trange(feature_num-4, position=foldnum, desc="Fold{}".format(foldnum))

        for col_num in range(feature_num-4):
            fold_train_col, fold_test_col = copy.deepcopy(fold_train), copy.deepcopy(fold_test)

            fold_train_col.iloc[:, col_num] = np.random.permutation(fold_train_col.iloc[:, col_num].values)
            fold_test_col.iloc[:, col_num] = np.random.permutation(fold_test_col.iloc[:, col_num].values)
            train_dataset, test_dataset = dl.get_shuffle_df(fold_train_col, fold_test_col)

            if config.embedding_train:
                data = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=False)
            else:
                data = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

            valid_result[col_num] = model_valid(col_num, model=model, dataloader=data,
                                                loss_function=loss_fn, get_ci=True, use_gpu=use_gpu, coo=coo)
            score = valid_result[col_num]['valid c-index']

            pgbar.set_postfix(score=score)
            pgbar.update()

            f = open('./results/shuffle_test/shuffle_pan_vae_{}.tsv'.format(config.shuffle_model), 'a')
            f.write('\t' + str(score))
            f.close()

    elif config.embedding_output:
        dl.logger.debug("Embedding output {}, cancer: {}".format(it + 1, cancertype))

        train_dataset, test_dataset, train_index, test_index, train_drop, test_drop = dl.get_split_ember(it)

        if config.embedding_train:
            data_loaded = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=False)
            index = train_index.values
            x_drop = train_drop
        else:
            data_loaded = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)
            index = test_index.values
            x_drop = test_drop

        cox_output, embedding_vector = model_valid_embedding(model=model, dataloader=data_loaded, use_gpu=use_gpu, coo=coo)
        embedding_vector = pd.DataFrame(embedding_vector.cpu().numpy())
        embedding_vector.index = index
        embedding_vector = pd.merge(embedding_vector, x_drop, left_index=True, right_index=True)

        cox_output = pd.DataFrame(cox_output.detach().cpu().numpy())
        cox_output.index = index

        save_path = './results/embedding/' + pretrained.split('/')[-1]
        embedding_vector.to_csv(save_path + '_vector' + '.tsv', sep='\t')
        cox_output.to_csv(save_path + '_cox_output' + '.tsv', sep='\t')

    else:
        dl.logger.debug("Test {}, cancer: {}".format(it + 1, cancertype))

        fold_train, train_dataset, test_dataset, fold_train, fold_test = dl.get_split_dataset(it)
        train_dl = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=False)
        valid_dl = DataLoader(dataset=test_dataset, batch_size=10000, shuffle=False)

        # get the last results
        test_score = model_valid(last_ep, model=model, loss_function=loss_fn, dataloader=valid_dl,
                                            get_ci=True, coo=coo, use_gpu=use_gpu)

        score = round(test_score['valid c-index'], 3)
        model_spec = pretrained.split('/')[-1]

        print(score)
        f = open('./results/model_test.tsv', 'a')
        f.write('\n' + datetime.datetime.now().strftime("%Y%m%d") + '\t' + cancertype + '\t' + model_name)
        f.write('\t' + model_spec + '\t' + str(gcn_mode) + '\t' + str(score) + '\t' + str(it))
        f.close()


if __name__ == "__main__":
    main()
