#python gjx_train_CNN_fisher_ll2.py --data_name MNIST --model_name conv --control_name 1_20_1_non-iid-2_fix_a2-b8_bn_1_1 --id 20240719fisherll2_delay10 --quanti_mode linear --linear_arg 50 --beta_T 1 --T_set 20 --delay_epoch 10  --onepoint 0
#python gjx_train_CNN_fisher_ll2.py --data_name MNIST --model_name conv --control_name 1_20_1_non-iid-2_fix_a2-b8_bn_1_1 --id 20240719fisherll2_delay10_onepoint2 --quanti_mode linear --linear_arg 50 --beta_T 1 --T_set 20 --delay_epoch 10  --onepoint 2

#python gjx_train_CNN_fisher_ll3.py --data_name MNIST --model_name conv --control_name 1_20_1_non-iid-2_fix_a2-b8_bn_1_1 --id 20240720fisherll3_avg10 --quanti_mode linear --linear_arg 50 --beta_T 1 --T_set 20 --delay_epoch 5  --onepoint 5

#python gjx_train_CNN_fisher_ll2.py --data_name CIFAR10 --model_name resnet18 --control_name 1_20_1_non-iid-5_fix_a2-b8_bn_1_1 --id 20240719fisherll2_delay10_resnet18 --quanti_mode linear --linear_arg 50 --beta_T 1 --T_set 20 --delay_epoch 10  --onepoint 0
#python gjx_train_CNN_fisher_ll2.py --data_name CIFAR10 --model_name resnet18 --control_name 1_20_1_non-iid-5_fix_a2-b8_bn_1_1 --id 20240719fisherll2_delay10_onepoint2_resnet18 --quanti_mode linear --linear_arg 50 --beta_T 1 --T_set 20 --delay_epoch 10  --onepoint 2


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
#from fed_roll import Federation
#from fed_random import Federation
from metrics import Metric
from utils_new import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import Logger
import csv
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
parser.add_argument('--id', default=None, type=str)
parser.add_argument('--linear_arg',default=None, type=float)
parser.add_argument('--quanti_mode',default=None, type=str)
parser.add_argument('--beta_T',default=None, type=float)
parser.add_argument('--T_set',default=None, type=float)
parser.add_argument('--delay_epoch',default=None, type=int)
parser.add_argument('--onepoint',type=int)

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
cfg['K']=10
device_model_time={}
with open("./device_list/"+cfg['model_name']+"_device_model_time.csv","r") as f:
    reader=csv.reader(f)
    reader_row=next(reader)
    for row in reader:
        device_model_time[str(row[0])+str(row[1])]=float(row[2])*int(cfg['batch_size']['train'])*int(cfg['K'])
print(device_model_time)
device_list_device=[]
device_list_modelsize=[]
device_list_commrate=[]
device_list_commtime=[]
with open("./device_list/"+cfg['model_name']+"_device_list.csv","r") as f:
    reader=csv.reader(f)
    reader_row=next(reader)
    for row in reader:
        device_list_device.append(row[1])
        device_list_modelsize.append(float(row[2]))
        device_list_commrate.append(float(row[3]))
        device_list_commtime.append(float(row[4]))


cfg['id']=args['id']
linear_arg=args['linear_arg']
quanti_mode=args['quanti_mode']
beta_T=args['beta_T']
T_set=args['T_set']
delay_epoch=args['delay_epoch']
onepoint=args['onepoint']

print(cfg['batch_size']['train'])
print(cfg)
print(device_model_time)

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
    
def deltaloss_1(loss1,loss2,lossth,a,b):
    #print(loss2)
    #print(loss1)
    #print(a)
    #print(b)
    if loss2>=lossth:
        return (1+abs(loss2-loss1))**(a+b)
    else:
        return (1+abs(loss2-loss1))**b
        
def deltaloss_2(loss1,loss2,deltalossth,c,tlastmin,t):
    if abs(loss2-loss1)<=deltalossth:
        return c*tlastmin/t
    else:
        return 0
        
def loss_compare_1(loss,losstarget,alpha):
    if loss>=alpha*losstarget:
        return 1
    return 0
    
def loss_compare_2(deltaloss,deltalossth):
    if deltaloss<=deltalossth:
        return 1
    return 0
        
def sigmoid(x):
    return 1/(1+np.exp(-(x-13)))+0.5
    
def sig(x):
    return 1-(x/max(x)-1)**6
    
def sig_linear(x,linear_arg):
    ans=np.zeros(len(x))
    for i in range(0,len(x)):
        if x[i]>=0:
          ans[i]=x[i]/linear_arg
        else:
          ans[i]=-abs(x[i])/linear_arg
    return x/linear_arg;
    
def quantization(W,N):
    maxW=max(W)
    minW=min(W)
    ans=np.zeros(N)
    for n in range(N):
        if (W[n]-minW)/(maxW-minW)>=0.8:
            ans[n]=1
        if (W[n]-minW)/(maxW-minW)<0.8 and (W[n]-minW)/(maxW-minW)>=0.6:
            ans[n]=0.5
        if (W[n]-minW)/(maxW-minW)<0.6 and (W[n]-minW)/(maxW-minW)>=0.4:
            ans[n]=0.25
        if (W[n]-minW)/(maxW-minW)<0.4 and (W[n]-minW)/(maxW-minW)>=0.2:
            ans[n]=0.125
        if (W[n]-minW)/(maxW-minW)<0.2:
            ans[n]=0.0625
    return ans
        
def quantization_new(W,N):
    ans=np.zeros(N)
    for n in range(N):
        if W[n]>=0.8:
            ans[n]=1
        if W[n]<0.8 and W[n]>=0.6:
            ans[n]=0.5
        if W[n]<0.6 and W[n]>=0.4:
            ans[n]=0.25
        if W[n]<0.4 and W[n]>=0.2:
            ans[n]=0.125
        if W[n]<0.2:
            ans[n]=0.0625
    return ans

def W2abs(x,beta):
    if x>=0:
        return x**beta
    return -(abs(x)**beta)

def deletenan(W,W_past):
    if math.isnan(W)==True:
        return W_past
    return W
    
    
def update_model_rate(model_rate, epoch,fisher_record, delay_epoch,data_split_len,cfg,T_set,beta_T,linear_arg,device_model_time, device_list_device,device_list_commtime,onepoint):
    last_model_rate=model_rate
    last_T=np.ones(cfg['num_users'])
    for n in range(0,cfg['num_users']):
        if abs(last_model_rate[n]-0.0625)<0.01:
            last_T[n]=device_model_time[device_list_device[n]+"0.0625"]+device_list_commtime[n]
        if abs(last_model_rate[n]-0.125)<0.01:
            last_T[n]=device_model_time[device_list_device[n]+"0.125"]+device_list_commtime[n]
        if abs(last_model_rate[n]-0.25)<0.01:
            last_T[n]=device_model_time[device_list_device[n]+"0.25"]+device_list_commtime[n]
        if abs(last_model_rate[n]-0.5)<0.01:
            last_T[n]=device_model_time[device_list_device[n]+"0.5"]+device_list_commtime[n]
        if abs(last_model_rate[n]-1)<0.01:
            last_T[n]=device_model_time[device_list_device[n]+"1.0"]+device_list_commtime[n]
    W=np.zeros(cfg['num_users'])
#    if onepoint==0:
#        for n in range(0,cfg['num_users']):
#            if epoch-2-delay_epoch<0:
#                W[n]=data_split_len[n]*(0.00001)*T_set/((last_T[n])**beta_T)
#            else:
#                W[n]=data_split_len[n]*(fisher_record[epoch-2-delay_epoch][n]**0.5)*(T_set/(last_T[n]))**beta_T
#    else:
    for n in range(0,cfg['num_users']):
        if epoch-2-delay_epoch-onepoint<0:
            W[n]=data_split_len[n]*(0.00001)*(T_set/last_T[n])**beta_T
        else:
            temp_avg=0
            for record_i in range(epoch-2-delay_epoch-onepoint,epoch-2-delay_epoch+onepoint+1):
                temp_avg=temp_avg+fisher_record[record_i][n]
                temp_avg=temp_avg/(2*onepoint+1)
                W[n]=data_split_len[n]*(temp_avg**0.5)*(T_set/last_T[n])**beta_T
    print("W")
    print(W)
    W=sig_linear(W,linear_arg)
    print("sig_W")
    print(W)
    model_rate=quantization_new(W,cfg['num_users'])
    return model_rate

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
    
    data_split_temp=data_split["train"]
    data_split_len=[0 for n in range(0,cfg['num_users'])]
    for i in range(0,cfg['num_users']):
        data_split_len[i]=len(data_split_temp[i])
    #print(data_split_len)
    
    maxxx = 0
    MAX_TIME=[]
    Accuracy=[]
    LOSS1 = np.ones(cfg['num_users']) # 
    LOSS2 = np.ones(cfg['num_users']) # 
    LOSS_util = np.ones(cfg['num_users'])
    LOSS_record = []
    fisher_record = []
    fisher = np.zeros(cfg['num_users']) 
    delta_loss_record = []
    delta_loss_record_sum = 0
    ACC = np.ones(cfg['num_users'])
    s = 1 
    C = np.ones(cfg['num_users']) #   
    P = np.ones(cfg['num_users']) #
    stack = np.zeros(cfg['num_users']) #
    lb_level = np.zeros(cfg['num_users'])
    lb = np.zeros(cfg['num_users'])
    ub = np.zeros(cfg['num_users'])
    W =  np.zeros(cfg['num_users'])
    W_past = np.zeros(cfg['num_users'])
    W_record1 = np.zeros(cfg['num_users'])
    W_record2 = np.zeros(cfg['num_users'])
    W_record3 = np.zeros(cfg['num_users'])
    TE_past = np.zeros(cfg['num_users'])
    T =  np.zeros(cfg['num_users'])
    T_comp = np.zeros(cfg['num_users'])
    T_comm = np.zeros(cfg['num_users'])
    modelrate=[0.0625 for n in range(cfg['num_users'])]
    Modelsize_record = np.zeros(cfg['num_users'])
    os.mkdir('./log/'+cfg['id'])
    for n in range(cfg['num_users']):
        T[n]=device_model_time[device_list_device[n]+"1.0"]+device_list_commtime[n]
        T_comp[n]=device_model_time[device_list_device[n]+"1.0"]
        T_comm[n]=device_list_commtime[n]
    Tlastmin=min(T)
    print(T)
    for epoch in range(last_epoch, cfg['num_epochs']['global'] + 1):
        logger.safe(True)
        modelrate=update_model_rate(modelrate, epoch,fisher_record, delay_epoch,data_split_len,cfg,T_set,beta_T,linear_arg,device_model_time, device_list_device,device_list_commtime,onepoint)
        with open('./log/'+cfg['id']+'/'+cfg['id']+'_modelrate.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(modelrate)
        #####################base
        #for n in range(cfg['num_users']):
        #    modelrate[n]=1.0
        
        #####################
        for n in range(cfg['num_users']):
            T[n]=device_model_time[device_list_device[n]+str(modelrate[n])]+device_list_commtime[n]*(modelrate[n])**2
            T_comp[n]=device_model_time[device_list_device[n]+str(modelrate[n])]
            T_comm[n]=device_list_commtime[n]*(modelrate[n])**2
            Modelsize_record[n]=device_list_modelsize[n]*(modelrate[n])**2
        print("TTTTTTTTTTTTTTT")
        print(T)
        print("TTTTTTTTTTTTTTT")
        with open('./log/'+cfg['id']+'/'+cfg['id']+'_T.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(T)
        with open('./log/'+cfg['id']+'/'+cfg['id']+'_Tcomp.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(T_comp)
        with open('./log/'+cfg['id']+'/'+cfg['id']+'_Tcomm.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(T_comm)
        with open('./log/'+cfg['id']+'/'+cfg['id']+'_modelsize.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(Modelsize_record)
        Tlastmin=min(T)
        print(modelrate)
        federation = Federation(global_parameters, modelrate, label_split)
        print(modelrate)
        LOSS1 = LOSS2.copy()
        #print(LOSS1)
        LOSS2, maxxx, fisher, LOSS_util = train(dataset['train'], data_split['train'], label_split, federation, model, optimizer, logger, epoch, maxxx, MAX_TIME, LOSS2, T, cfg['K'],fisher,cfg,LOSS_util)
        fisher_record.append(fisher)
        LOSS_record.append(LOSS_util)
        with open('./log/'+cfg['id']+'/'+cfg['id']+'_fisher.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(fisher)
        print("Current loss:")
        print(LOSS2)
        print("Delta loss:")
        print(abs(LOSS2-LOSS1))
        delta_loss_record.append(np.mean(abs(LOSS2-LOSS1)))
        delta_loss_record_sum+=np.mean(abs(LOSS2-LOSS1))
        test_model = stats(dataset['train'], model)
        thisacc=test(dataset['test'], data_split['test'], label_split, test_model, logger, epoch, Accuracy)
        with open('./log/'+cfg['id']+'/'+cfg['id']+'.csv', 'a', newline='') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow([maxxx, thisacc,sum(LOSS2)/len(LOSS2)])
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
    logger.safe(False)
    return


def train(dataset, data_split, label_split, federation, global_model, optimizer, logger, epoch, maxxx, MAX_TIME, LOSS, T, K,fisher, cfg, LOSS_util):
    global_model.load_state_dict(federation.global_parameters)
    global_model.train(True)
    local, local_parameters, user_idx, param_idx = make_local(dataset, data_split, label_split, federation, K)
    num_active_users = len(local)
    lr = optimizer.param_groups[0]['lr']
    s = 1
    max_time = 0
    for m in range(num_active_users):
        local_parameters[m],LOSS[user_idx[m]],fisher[user_idx[m]], LOSS_util[user_idx[m]] = copy.deepcopy(local[m].train(local_parameters[m], lr, logger, cfg))
        local_time = T[m]
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
    return LOSS, maxxx, fisher, LOSS_util


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
        thisacc=Accuracy[-1]
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        logger.write('test', cfg['metric_name']['test']['Local'] + cfg['metric_name']['test']['Global'])
    return thisacc


def make_local(dataset, data_split, label_split, federation, K):
    num_active_users = int(np.ceil(cfg['frac'] * cfg['num_users']))
    user_idx = torch.arange(cfg['num_users'])[torch.randperm(cfg['num_users'])[:num_active_users]].tolist()
    local_parameters, param_idx = federation.distribute(user_idx)
    local = [None for _ in range(num_active_users)]
    for m in range(num_active_users):
        model_rate_m = federation.model_rate[user_idx[m]]
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, data_split[user_idx[m]])})['train']
        local[m] = Local(model_rate_m, data_loader_m, label_split[user_idx[m]], K)
    return local, local_parameters, user_idx, param_idx


class Local:
    def __init__(self, model_rate, data_loader, label_split, K):
        self.model_rate = model_rate
        self.data_loader = data_loader
        self.label_split = label_split
        self.K =K
        #self.accuracy = evaluation

    def train(self, local_parameters, lr, logger, cfg):
        metric = Metric()
        model = eval('models.{}(model_rate=self.model_rate).to(cfg["device"])'.format(cfg['model_name']))
        model.load_state_dict(local_parameters)
        model.train(True)
        Loss = 0
        Loss_util = 0
        optimizer = make_optimizer(model, lr)
        cnt = 0
        fisher = []

        for local_epoch in range(1, cfg['num_epochs']['local'] + 1):
            for i, input in enumerate(self.data_loader):
                cnt += 1
                input = collate(input)
                input_size = input['img'].size(0)
                input['label_split'] = torch.tensor(self.label_split)
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                output = model(input)
                output['loss'].backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_square_mean = torch.mean(param.grad ** 2).item() / input_size
                        fisher.append(float(grad_square_mean))
                        #print(f'Epoch {local_epoch}, Batch {i+1}, {name} gradient square mean: {grad_square_mean:.6f}')

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                evaluation = metric.evaluate(cfg['metric_name']['train']['Local'], input, output)
                logger.append(evaluation, 'train', n=input_size)
                Loss += evaluation['Local-Loss']
                Loss_util += evaluation['Local-Loss']**2
                if cnt >= self.K:
                    break
            if cnt >= self.K:
                break

        Loss = Loss / self.K
        Loss_util=Loss_util/(cfg['batch_size']['train']*self.K)
        local_parameters = model.state_dict()
        max_fisher=max(fisher)
        return local_parameters, Loss, max_fisher, Loss_util



if __name__ == "__main__":
    main()
