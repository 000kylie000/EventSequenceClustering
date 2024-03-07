import numpy as np
import itertools
# from clustering_generate_data_v1 import Logic_Model_Generator
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch
import torch.optim as optim
import datetime
import time
# from utils import redirect_log_file, Timer

##################################################################
class Logic_Model(nn.Module): 
    # Assumption: body predicates: X1 ~ X5;  head predicate: X6
    # rules(related to X1~X5): 
    # f0: background; 
    # f1: X1 AND X2 AND X3, X1 before X2; 

    def __init__(self):

        ### the following parameters are used to manually define the logic rules
        self.num_predicate = 11
        self.num_formula = 2

        self.body_predicate_set = list(np.arange(0,(self.num_predicate-1),1)) 
        self.head_predicate_set = [self.num_predicate-1] 
        
        self.empty_pred = 2 # append 2 empty predicates
        self.k = 3 # relaxedtopk parameter topk 
        self.sigma = 0.1 # laplace kernel parameter
        self.sigma2 = 0.1
        self.temp = 0.1
        self.tolerance = 0.1
        self.prior = torch.tensor([0.1,0.9], dtype=torch.float64, requires_grad=True)

        self.weight = (torch.ones((self.num_formula-1), (len(self.body_predicate_set)+self.empty_pred)) * 0.000000001).double()
        self.weight[0,[0,1,2]] = 1000

        self.weight = F.normalize(self.weight, p=1, dim = 1)
        self.weight.requires_grad = True

        self.relation = {}
        for i in range(self.num_formula-1):
            self.relation[str(i)] = {}
            
        self.model_parameter = {}
        head_predicate_idx = self.head_predicate_set[0]
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] = torch.ones(1)*0.02 
        self.model_parameter[head_predicate_idx]['weight'] = torch.autograd.Variable((torch.ones(self.num_formula - 1) * 0.5).double(), requires_grad=True)


    def log_p_star(self, head_predicate_idx, t, pi, data_sample, A, add_relation = True, relation_grad = True):
        # background
        cur_intensity = self.model_parameter[head_predicate_idx]['base']
        log_p_star_0 = torch.log(cur_intensity.clone()) + (- t * cur_intensity) + torch.log(pi[0])
        body_predicate_indicator = data_sample <= t
        body_predicate_indicator = torch.cat([body_predicate_indicator,torch.ones(len(t),self.empty_pred)],dim=1)
        body_predicate_indicator = body_predicate_indicator.repeat(1, A.shape[0])
        body_predicate_indicator = body_predicate_indicator.reshape(len(t), A.shape[0], A.shape[1])

        feature_formula = torch.exp(-abs(torch.sum(body_predicate_indicator * A, dim=2) - self.k) / self.sigma)
        relation_used = {}
        if add_relation:
            topk_idx = A.sort(1,descending=True)[1][:,:self.k]
            topk_idx = topk_idx.sort(1,descending=False)[0]
            topk_val = A.sort(1,descending=True)[0][:,:self.k]
            relation_features = []

            for i in range(self.num_formula-1): # iterates rules
                relation_used[str(i)]=[]
                rule_relation_features = []
                idx = topk_idx[i,:]
                select_body = list(idx.numpy() < (self.num_predicate-1))
                if select_body[-1] == 0:
                    body_idx = idx[:select_body.index(0)]
                else:
                    body_idx = idx

                if len(body_idx) > 1: # only when the num of body pred larger than 1 , there will be temporal relation
                    body_idx2 = (body_idx.repeat(1,2)).reshape(2,-1)
                    idx_comb = np.array(list(itertools.product(*body_idx2)))
                    idx_comb = np.delete(idx_comb, np.arange(0, len(body_idx)**2, len(body_idx)+1), axis=0)
                    delete_set = []
                    for j in range(len(body_idx)-1):
                        delete_set = delete_set + list(np.arange(j+(len(body_idx)-1)*(j+1), len(body_idx)**2-len(body_idx), len(body_idx)-1))
                    idx_comb = np.delete(idx_comb, delete_set, axis=0)
                    for k in range(idx_comb.shape[0]):
                        relation_used[str(i)].append(str(list(idx_comb[k,:])))
                        if str(list(idx_comb[k,:])) in self.relation[str(i)]:
                            self.relation[str(i)][str(list(idx_comb[k,:]))].requires_grad = True
                            prob = self.relation[str(i)][str(list(idx_comb[k,:]))]
                        else:
                            self.relation[str(i)][str(list(idx_comb[k,:]))] = F.normalize(torch.ones(4)*torch.tensor([1,1,1,1]), p=1, dim = 0) 
                            self.relation[str(i)][str(list(idx_comb[k,:]))].requires_grad = True
                            prob = self.relation[str(i)][str(list(idx_comb[k,:]))]
                        
                        time_diff = data_sample[:,idx_comb[k,0]] - data_sample[:,idx_comb[k,1]]
                        time_binary_indicator = torch.zeros(len(time_diff),4)  

                        time_binary_indicator[:,0] = time_diff > self.tolerance # after
                        time_binary_indicator[:,1] = abs(time_diff) < self.tolerance # equal
                        time_binary_indicator[:,2] = time_diff < -self.tolerance # before
                        time_binary_indicator[:,3] = 1-prob[0]*time_binary_indicator[:,0].clone()-prob[1]*time_binary_indicator[:,1].clone()-prob[2]*time_binary_indicator[:,2].clone() 
                        
                        rule_relation_features.append(self.softmax(time_binary_indicator * prob))   
                else:
                    continue
                rule_relation_feature = self.softmin(torch.stack(rule_relation_features,dim=1))
                relation_features.append(rule_relation_feature)
            # get soft min of relation features
            if len(relation_features) != 0:
                relation_feature = torch.stack(relation_features,dim=1)
            else:
                relation_feature = torch.ones(500,self.num_formula-1)
            feature_formula = feature_formula * relation_feature

 
        data_sample = data_sample.repeat(1, A.shape[0])
        data_sample = data_sample.reshape(len(t), A.shape[0], -1)
        max_body_time = torch.max(body_predicate_indicator[:,:,0:(self.num_predicate-1)] * data_sample[:,:,0:(self.num_predicate-1)] * A[:,0:(self.num_predicate-1)], dim=2)[0] 
        t = t.repeat(1, A.shape[0])
        t = t.reshape(len(t),A.shape[0])
        sigm = torch.sigmoid(t - max_body_time) # max body time < t: 1;   max body time > t: 0
        pre_intensity = self.model_parameter[head_predicate_idx]['base']
        cur_intensity = self.model_parameter[head_predicate_idx]['base'] + sigm * feature_formula * self.model_parameter[head_predicate_idx]['weight'] 
        log_p_star = torch.log(cur_intensity.clone()) + (- t * cur_intensity + sigm * (- max_body_time * pre_intensity + max_body_time * cur_intensity)) + torch.log(pi[1:])
        return torch.cat([log_p_star_0, log_p_star],dim=1), relation_used

    def softmin(self, x):
        exp_x = torch.exp(-x/self.temp)
        return torch.sum(exp_x * x, dim = 1) / torch.sum(exp_x, dim = 1)
    
    def softmax(self, x):
        exp_x = torch.exp(x/self.temp)
        return torch.sum(exp_x * x, dim = 1) / torch.sum(exp_x, dim = 1)

    def reparametrization(self, avg_num, tau):
        weight = self.weight.expand(avg_num,self.weight.shape[0],self.weight.shape[1])
        EPSILON = torch.from_numpy(np.array(np.finfo(np.float32).tiny))
        scores = torch.log(weight.clone())
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g
        khot = torch.zeros_like(scores)
        onehot_approx = torch.zeros_like(scores)
        for i in range(self.k):
            khot_mask = torch.max(1.0 - onehot_approx, EPSILON)
            scores = scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(scores / tau, dim=2)
            khot = khot + onehot_approx
        A =  khot
        return torch.mean(A, axis=0)

    def optimize_EM_single(self, body_time, head_time, batch_size, head_predicate_idx, optimizer, pi, tau, num_samples):
        qz = [] # E step
        num_batch = num_samples / batch_size
        EPSILON = torch.from_numpy(np.array(np.finfo(np.float32).tiny))

        dict = {}
        # 1. update rule content： self.weight
        self.model_parameter[head_predicate_idx]['weight'].requires_grad = False

        self.relation["0"]["[0, 1]"] = F.normalize(torch.ones(4)*torch.tensor([1,1,1000,1]),p=1,dim=0)
        self.relation["0"]["[0, 2]"] = F.normalize(torch.ones(4)*torch.tensor([1,1,1,1000]),p=1,dim=0)
        self.relation["0"]["[1, 2]"] = F.normalize(torch.ones(4)*torch.tensor([1,1,1,1000]),p=1,dim=0)

        # 2. update model parameters: self.model_parameter[head_predicate_idx]['weight'],use second half of data
        indices = np.arange(body_time.shape[0])
        np.random.shuffle(indices)
        body_time = body_time[indices,:]
        head_time = head_time[indices]

        self.weight.requires_grad = False
        for batch_idx in np.arange(num_batch/2, num_batch, 1): # iterate batches
            tau = torch.ones(1)*0.1
            sample_ID_batch = np.arange(batch_idx*batch_size, (batch_idx+1)*batch_size, 1)
            self.model_parameter[head_predicate_idx]['weight'].requires_grad = True
            optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
            avg_num = 10
            A = self.reparametrization(avg_num, tau)

            # iterate over a batch
            log_likelihood = torch.tensor([0], dtype=torch.float64)
            data_sample = body_time[sample_ID_batch,:]
            t = head_time[sample_ID_batch,:]
            log_p_star, relation_used = self.log_p_star(head_predicate_idx, t, pi, data_sample, A)
            log_avg = log_p_star          # batch_size x num rules
            qzi = F.normalize(torch.exp(log_avg),p=1,dim=1)
            log_likelihood = torch.sum(qzi * log_avg) 
           
            loss = -log_likelihood 
            loss.backward(retain_graph=True)
            # update weight
            with torch.no_grad():
                grad_weight = self.model_parameter[head_predicate_idx]['weight'].grad
                self.model_parameter[head_predicate_idx]['weight'] -= grad_weight * 0.0002
                self.model_parameter[head_predicate_idx]['weight'] = torch.maximum(self.model_parameter[head_predicate_idx]['weight'], EPSILON)

        # 3. get E step
        for batch_idx in np.arange(0, num_batch, 1): # iterate batches
            tau = torch.ones(1)*0.1
            sample_ID_batch = np.arange(batch_idx*batch_size, (batch_idx+1)*batch_size, 1)
            z_p_star = torch.zeros(batch_size,self.num_formula)
            
            avg_num = 10
            A = self.reparametrization(avg_num, tau)
            log_likelihood = torch.tensor([0], dtype=torch.float64)
            data_sample = body_time[sample_ID_batch,:]
            t = head_time[sample_ID_batch,:]
            log_p_star, relation_used = self.log_p_star(head_predicate_idx, t, pi, data_sample, A)
            log_avg = log_p_star          # batch_size x num rules
            z_p_star = torch.exp(log_avg)
            qz.append(F.normalize(z_p_star, p=1, dim=1))
            
        qz = torch.cat(qz, dim=0)
        pi = torch.max(F.normalize(torch.sum(qz, dim=0), p=1, dim=0),torch.ones(1)*0.00001)
        pi = F.normalize(pi, p=1, dim=0)
            
        # save result to a txt file
        dict["A"] = A
        print(dict["A"])
        dict["WEIGHT"] = self.weight
        dict["pi"] = pi
        dict["weight"] = self.model_parameter[head_predicate_idx]['weight']
        print("----- A -----")
        print(dict["A"])
        print("----- WEIGHT -----")
        print(dict["WEIGHT"])
        print("----- pi -----")
        print(dict["pi"])
        print('----- weight -----')
        print(dict["weight"])
        return pi.detach(),self.model_parameter[head_predicate_idx]['weight'].numpy()

 
print('---------- key tunable parameters ----------')
print('sigma = 0.1')
print('tau = 20')
print('batch_size = 500')
print('lr = 0.1')
print('pernalty C = 5')
print('--------------------')
num_samples = 20000
iter_nums = 1000
# time_horizon = 5
batch_size = 500
num_batch = num_samples / batch_size

data = np.load('data_1rule11pred_20000.npy', allow_pickle='TRUE').item()
body_time = np.zeros((num_samples, 10))
head_time = np.zeros((num_samples, 1))
for i in range(num_samples):
    body_time[i,:] = data[i]['body_predicates_time']
    head_time[i,:] = data[i]['head_predicate_time'][0]
body_time = torch.from_numpy(body_time)
head_time = torch.from_numpy(head_time)

head_pred = 10
logic_model = Logic_Model()
torch.autograd.set_detect_anomaly(True)
optimizer = optim.Adam([{'params':logic_model.model_parameter[head_pred]['weight']},{'params': logic_model.weight}], lr=0.1)
prior = torch.from_numpy(np.array([0.1, 0.9]))

tau = 20 * torch.ones(1) 

record = []
##### print key model information
for iter in range(iter_nums):
    prior,weight = logic_model.optimize_EM_single(body_time, head_time, batch_size, head_pred, optimizer, prior, tau, num_samples)
    if iter == 0:
        prev_weight = weight
    else:
        dist = np.sqrt((prev_weight-weight)**2)
        print(dist)
        if dist < 0.01:
            break
        else:
            prev_weight = weight

       