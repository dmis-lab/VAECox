from vae_models import *
import argparse
import logging
import sys
import statistics

parser = argparse.ArgumentParser()

parser.add_argument('--cancer_list', '-cl', type=str, default='coxnnet')
parser.add_argument('--omic_list', '-ol', nargs='+', type=str)
parser.add_argument('--missing_impute', '-mi', type=str, default='mean')
parser.add_argument('--exclude_impute', '-xi', default=False, action='store_true')
parser.add_argument('--feature_scaling', '-fc', type=str, default='None')
parser.add_argument('--feature_selection', '-fs', type=str, default='None')
parser.add_argument('--gcn_mode', '-gcn', default=False, action='store_true')
parser.add_argument('--gcn_func', '-gcf', default='None', type=str)
parser.add_argument('--ipcw_mode', '-ipcw', default=False, action='store_true')
parser.add_argument('--device_type', '-dv', type=str, default='cuda')
parser.add_argument('--cuda_device', '-cd', type=str, default='0')
parser.add_argument('--hidden_nodes', '-hn', type=int, default=2048)
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
parser.add_argument('--weight_sparsity', '-ws', type=float, default=1e-6)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-6)
parser.add_argument('--max_epochs', '-mx', type=int, default=500)
parser.add_argument('--model_optimizer', '-mo', type=str, default='SGD')
parser.add_argument('--dropout_rate', '-dr', type=float, default=0.0)
parser.add_argument('--acti_func', '-af', default="ReLU", type=str)
# Graph Convolution
parser.add_argument('--sub_graph', '-sg', default=0, type=int)
parser.add_argument('--batch_size', '-bs', default=64, type=int)
parser.add_argument('--pool_func', '-pf', default='Single', type=str)
parser.add_argument('--topk_pooling', '-tp', default=0.5, type=float)
# Cox Regression
parser.add_argument('--multi_task', '-mu', default=False, action='store_true')
parser.add_argument('--mt_regularization', '-mr', default='None', type=str)
parser.add_argument('--num_clusters', '-nc', default=8, type=int)
parser.add_argument('--augment_autoencoder', '-aug', default='None', type=str)
parser.add_argument('--deseq2', '-deseq', default=False, action='store_true')
# parser.add_argument('--file_version', '-fv', type=str, default='15%')
parser.add_argument('--hp_search', '-hs', default=False, action='store_true')
parser.add_argument('--vae_data', '-vd', default='ember_libfm_190507', type=str)
parser.add_argument('--test_mode', '-tm', default=False, action='store_true')
parser.add_argument('--model_struct', '-ms', default='basic', type=str)
parser.add_argument('--model_type', '-mt', default='coxrgmt', type=str)
parser.add_argument('--save_mode', '-sm', default=False, action='store_true')
parser.add_argument('--checkpoint_dir', '-cp', default='./results/', type=str)
parser.add_argument('--session_name', '-sn', default='test', type=str)
parser.add_argument('--pickle_save', '-ps', default=False, action='store_true')
config = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=config.cuda_device
device = torch.device(config.device_type)
LOGGER = logging.getLogger()

def init_logging(config):
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)
    # For logfile writing
    logfile = logging.FileHandler(
        config.checkpoint_dir + 'logs/' + ' '.join(sys.argv) + '.txt', 'w')
    logfile.setFormatter(fmt)
    LOGGER.addHandler(logfile)

def init_seed(seed=None):
    if seed is None:
        seed = int(round(time.time() * 1000)) % 10000
    LOGGER.info("Using seed={}, pid={}".format(seed, os.getpid()))
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def run_session(config):
    result_list = []
    if config.model_type == 'vae':
        train, valid, test, num_cols = torchify_vaeserin(config)
        maven = VAE(config=config, logger=LOGGER, num_features=num_cols)
        maven.init_layers()
        test_loss = maven.fit_predict(train, valid, test)
        LOGGER.info('==============Final Results==============')
        LOGGER.info('Metric\tTraining\tValidation\tTesting') 
        LOGGER.info('RMSE Loss\t{0:0.4f}\t{1:0.4f}\t{2:0.4f}'.format(math.sqrt(maven.global_train_loss), math.sqrt(maven.global_valid_loss), math.sqrt(test_loss)))
        print('\n')

    elif config.model_type == 'ae':
        train, valid, test, num_cols = torchify_vaeserin(config)
        maven = AE(config=config, logger=LOGGER, num_features=num_cols)
        maven.init_layers()
        test_loss = maven.fit_predict(train, valid, test)
        LOGGER.info('==============Final Results==============')
        LOGGER.info('Metric\tTraining\tValidation\tTesting') 
        LOGGER.info('RMSE Loss\t{0:0.4f}\t{1:0.4f}\t{2:0.4f}'.format(math.sqrt(maven.global_train_loss), math.sqrt(maven.global_valid_loss), math.sqrt(test_loss)))
        print('\n')

    elif config.model_type == 'dae':
        train, valid, test, num_cols = torchify_vaeserin(config)
        maven = DAE(config=config, logger=LOGGER, num_features=num_cols)
        maven.init_layers()
        test_loss = maven.fit_predict(train, valid, test)
        LOGGER.info('==============Final Results==============')
        LOGGER.info('Metric\tTraining\tValidation\tTesting') 
        LOGGER.info('RMSE Loss\t{0:0.4f}\t{1:0.4f}\t{2:0.4f}'.format(math.sqrt(maven.global_train_loss), math.sqrt(maven.global_valid_loss), math.sqrt(test_loss)))
        print('\n')

    else:
        return NotImplemented

if __name__ == "__main__":
    init_logging(config)
    LOGGER.info('COMMAND: {}'.format(' '.join(sys.argv)))
    run_session(config)
    