import argparse
import copy
import datetime
import models
import numpy as np
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from fed import Federation
from metrics import Metric
from utils_new import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger
import csv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
for k in cfg:
    cfg[k] = args[k]
if args['control_name']:
    cfg['control'] = {k: v for k, v in zip(cfg['control'].keys(), args['control_name'].split('_'))} \
        if args['control_name'] != 'None' else {}
cfg['control_name'] = '_'.join([cfg['control'][k] for k in cfg['control']])
cfg['pivot_metric'] = 'Global-Accuracy'
cfg['pivot'] = -float('inf')
cfg['metric_name'] = {'train': {'Local': ['Local-Loss', 'Local-Accuracy']},
                      'test': {'Local': ['Local-Loss', 'Local-Accuracy'], 'Global': ['Global-Loss', 'Global-Accuracy']}}


def main():
    torch.cuda.set_device(0)
    device = torch.cuda.current_device()
    print(device)
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['subset'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    seed = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    dataset = fetch_dataset(cfg['data_name'], cfg['subset'])
    process_dataset(dataset)
    model = eval('models.{}(model_rate=cfg["global_model_rate"]).to(cfg["device"])'.format(cfg['model_name']))
    optimizer = make_optimizer(model, cfg['lr'])
    scheduler = make_scheduler(optimizer)
    COMM = 60
    COMP = 2
    if cfg['resume_mode'] == 1:
        last_epoch, data_split, label_split, model, optimizer, scheduler, logger = resume(model, cfg['model_tag'],
                                                                                          optimizer, scheduler)
    elif cfg['resume_mode'] == 2:
        last_epoch = 1
        _, data_split, label_split, model, _, _, _ = resume(model, cfg['model_tag'])
        logger_path = os.path.join('output', 'runs', '{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    else:
        last_epoch = 1
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
        logger_path = os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag']))
        logger = Logger(logger_path)
    if data_split is None:
        data_split, label_split = split_dataset(dataset, cfg['num_users'], cfg['data_split_mode'])
    global_parameters = model.state_dict()
    maxxx = 0
    MAX_TIME=[]
    Accuracy=[]
    LOSS1 = np.ones(cfg['num_users']) # 
    LOSS2 = np.ones(cfg['num_users']) # 
    ACC = np.ones(cfg['num_users'])
    s = 1 
    C = np.ones(cfg['num_users']) #   
    P = np.ones(cfg['num_users']) #
    stack = np.zeros(cfg['num_users']) #
    lb_level = np.zeros(cfg['num_users'])
    lb = np.zeros(cfg['num_users'])
    ub = np.zeros(cfg['num_users'])
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)
        for i in range(cfg['num_users']):
           #cfg['comm_time'][i] = np.random.randint(0,60)
           cfg['comm_time'][i] = COMM
           #cfg['comp_time'][i] = COMP
           C[i] = 1-cfg['comm_time'][i]/60
        for i in range(cfg['num_users']):
                if lb_level[i] == 0:
                    lb[i] = 0.0625
                elif lb_level[i] == 1:
                    lb[i] = 0.125
                elif lb_level[i] == 2:
                    lb[i] = 0.25
                elif lb_level[i] == 3:
                    lb[i] = 0.5
                else:
                    lb[i] = 1.0
                if cfg['comp_time'][i] > 16:#raspberry pi
                    ub[i] = 0.0625
                elif cfg['comp_time'][i] > 12.8:#NANO
                    ub[i] = 0.125
                elif cfg['comp_time'][i] > 10.24:#Tx2
                    ub[i] = 0.25
                elif cfg['comp_time'][i] > 8:#XAVIER
                    ub[i] = 0.5
                else:#LAPTOP
                    ub[i] = 1.0
                if lb[i] >= ub[i]:
                    P[i] = ub[i]
                else:
                    if C[i] < lb[i]:
                        P[i] = lb[i]
                    elif C[i] > ub[i]:
                        P[i] > ub[i]
                    else: 
                        P[i] = C[i]
                if P[i] <= 0.0625:
                    cfg['model_rate'][i] = 0.0625
                    cfg['model_comp'][i] = 0.615
                elif P[i] <= 0.125:
                    cfg['model_rate'][i] = 0.125
                    cfg['model_comp'][i] = 0.69
                elif P[i] <= 0.25:
                    cfg['model_rate'][i] = 0.25
                    cfg['model_comp'][i] = 0.76
                elif P[i] <= 0.5:
                    cfg['model_rate'][i] = 0.5
                    cfg['model_comp'][i] = 0.84
                else:
                    cfg['model_rate'][i] = 1
                    cfg['model_comp'][i] = 1

        federation = Federation(global_parameters, cfg['model_rate'], label_split)
        print(cfg['model_rate'])
        LOSS1 = LOSS2.copy()
        print(LOSS1)
        LOSS2 = train(dataset['train'], data_split['train'], label_split, federation, model, optimizer, logger, epoch, maxxx, MAX_TIME, LOSS2)
        print(LOSS2)
        SLOPE = (abs(LOSS2-LOSS1))
        print(SLOPE)
        for i in range(cfg['num_users']):
            if SLOPE[i] < 0.05:
                stack[i] = stack[i] + 1
                if stack[i] == 3:
                    stack[i] = 0
                    lb_level[i] = lb_level[i] + 1
        print(lb_level)
        test_model = stats(dataset['train'], model)
        test(dataset['test'], data_split['test'], label_split, test_model, logger, epoch, Accuracy)
        if cfg['scheduler_name'] == 'ReduceLROnPlateau':
            scheduler.step(metrics=logger.mean['train/{}'.format(cfg['pivot_metric'])])
        else:
            scheduler.step()
        logger.safe(False)
        model_state_dict = model.state_dict()
        save_result = {
            'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'label_split': label_split,
            'model_dict': model_state_dict, 'optimizer_dict': optimizer.state_dict(),
            'scheduler_dict': scheduler.state_dict(), 'logger': logger}
        save(save_result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if cfg['pivot'] < logger.mean['test/{}'.format(cfg['pivot_metric'])]:
            cfg['pivot'] = logger.mean['test/{}'.format(cfg['pivot_metric'])]
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    with open('iid_WAFL_same_comm_diff_comp.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(MAX_TIME)
        spamwriter.writerow(Accuracy)
    logger.safe(False)
    return


def train(dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch, maxxx, MAX_TIME, LOSS):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation)
    num_active_users = len(local)
    lr = optimizer.param_groups[0]['lr']
    s = 1
    max_time = 0
    for m in range(num_active_users):
        local_parameters[m],LOSS[user_idx[m]] = copy.deepcopy(local[m].train(local_parameters[m], lr, logger))
        if s == 1:
            local_time = cfg['model_comp'][user_idx[m]]*cfg['comp_time'][user_idx[m]] + cfg['model_rate'][user_idx[m]]*cfg['comm_time'][user_idx[m]]
        else:
            local_time = cfg['comp_time'][user_idx[m]] + cfg['comm_time'][user_idx[m]]
        if local_time > max_time:
            max_time = local_time
        if m % int((num_active_users * cfg['log_interval']) + 1) == 0:
            #epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_users - m - 1))
            #exp_finished_time = epoch_finished_time + datetime.timedelta(
                #seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_users))
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_users),
                             'ID: {}({}/{})'.format(user_idx[m], m+1, num_active_users),
                             #'Learning rate: {}'.format(lr),
                             'Rate: {}'.format(federation.model_rate[user_idx[m]]),
                             'Training Time: {}'.format(local_time)]}
                             #'Epoch Finished Time: {}'.format(epoch_finished_time),
                             #'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            logger.write('train', cfg['metric_name']['train']['Local'])
    maxxx = maxxx + max_time
    MAX_TIME.append(maxxx)
    federation.combine(local_parameters, param_idx, user_idx)
    global_model.load_state_dict(federation.global_parameters)
    return LOSS


def stats(dataset, model):
    with torch.no_grad():
        test_model = eval('models.{}(model_rate=cfg["global_model_rate"], track=True).to(cfg["device"])'
                          .format(cfg['model_name']))
        test_model.load_state_dict(model.state_dict(), strict=False)
        data_loader = make_data_loader({'train': dataset})['train']
        test_model.train(True)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input = to_device(input, cfg['device'])
            test_model(input)
    return test_model


def test(dataset, data_split, label_split, model, logger, epoch, Accuracy):
    with torch.no_grad():
        metric = Metric()
        model.train(False)
        for m in range(cfg['num_users']):
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])})['test']
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(label_split[m])
                input = to_device(input, cfg['device'])
                output = model(input)
                output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                evaluation = metric.evaluate(cfg['metric_name']['test']['Local'], input, output)
                logger.append(evaluation, 'test', input_size)
        data_loader = make_data_loader({'test': dataset})['test']
        Sum = 0
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['img'].size(0)
            input = to_device(input, cfg['device'])
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(cfg['metric_name']['test']['Global'], input, output)
            Sum = Sum + evaluation['Global-Accuracy']
            logger.append(evaluation, 'test', input_size)
        Accuracy.append(Sum/(i+1))
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
    return


def make_local(dataset, data_split, label_split, federation):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    local_parameters, param_idx = federation.distribute(user_idx)
    local = [None for _ in range(num_active_users)]
    for m in range(num_active_users):
        model_rate_m = federation.model_rate[user_idx[m]]
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, data_split[user_idx[m]])})['train']
        local[m] = Local(model_rate_m, data_loader_m, label_split[user_idx[m]])
    return local, local_parameters, user_idx, param_idx


class Local:
    def __init__(self, model_rate, data_loader, label_split):
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split
        #self.accuracy = evaluation

    def train(self, local_parameters, lr, logger):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        Loss = 0
        optimizer = make_optimizer(model, lr)
        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, input in enumerate(self.data_loader):
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(self.label_split)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                logger.append(evaluation, 'train', n=input_size)
                Loss += evaluation['Local-Loss']
        Loss = Loss/((i+1)*cfg['num_epochs']['local'])
        local_parameters = model.state_dict()
        return local_parameters, Loss



if __name__ == "__main__":
    main()
