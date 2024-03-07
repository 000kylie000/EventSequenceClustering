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
torch.manual_seed(2)
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
        self.temp = 0.07 # 0.05,0.01, 0.15
        self.tolerance = 0.1
        self.prior = torch.tensor([0.01,0.99], dtype=torch.float64, requires_grad=True)
        

        self.weight = (torch.ones((self.num_formula-1), (len(self.body_predicate_set)+self.empty_pred)) * 0.000000001).double()
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

        # use laplace kernel to calculate feature    
        feature_formula = torch.exp(-abs(torch.sum(body_predicate_indicator * A, dim=2) - self.k) / self.sigma)
        relation_used = {}
        if add_relation:
            topk_idx = A.sort(1,descending=True)[1][:,:self.k]
            topk_idx = topk_idx.sort(1,descending=False)[0]
            topk_val = A.sort(1,descending=True)[0][:,:self.k]
            relation_features = []

            relation_feature = torch.ones(500,self.num_formula-1)
            for i in range(self.num_formula-1): # iterates rules
                relation_used[str(i)]=[]
                rule_relation_features = []
                idx = topk_idx[i,:]
                # find body predicates, exclude dummy variables
                select_body = list(idx.numpy() < (self.num_predicate-1))
                if select_body[-1] == 0:
                    body_idx = idx[:select_body.index(0)]
                else:
                    body_idx = idx

                if len(body_idx) > 1: # only when the num of body pred larger than 1 , there will be temporal relation
                    body_idx2 = (body_idx.repeat(1,2)).reshape(2,-1)
                    idx_comb = np.array(list(itertools.product(*body_idx2)))
                    # find k(k-1)/2 combination for a rule
                    idx_comb = np.delete(idx_comb, np.arange(0, len(body_idx)**2, len(body_idx)+1), axis=0)
                    delete_set = []
                    for j in range(len(body_idx)-1):
                        delete_set = delete_set + list(np.arange(j+(len(body_idx)-1)*(j+1), len(body_idx)**2-len(body_idx), len(body_idx)-1))
                    idx_comb = np.delete(idx_comb, delete_set, axis=0)
                    # get prob from the dict and time indicator
                    for k in range(idx_comb.shape[0]):
                        relation_used[str(i)].append(str(list(idx_comb[k,:])))
                        if str(list(idx_comb[k,:])) in self.relation[str(i)]:
                            self.relation[str(i)][str(list(idx_comb[k,:]))].requires_grad = True
                            prob = self.relation[str(i)][str(list(idx_comb[k,:]))]
                        else:
                            self.relation[str(i)][str(list(idx_comb[k,:]))] = F.normalize(torch.ones(4)*torch.tensor([1,1,1,10]), p=1, dim = 0)
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
                relation_feature[:,i] = rule_relation_feature
            
            feature_formula = feature_formula * relation_feature

        # get soft max body time (our method)
        data_sample = data_sample.repeat(1, A.shape[0])
        data_sample = data_sample.reshape(len(t), A.shape[0], -1)
        max_body_time = torch.max(body_predicate_indicator[:,:,0:(self.num_predicate-1)] * data_sample[:,:,0:(self.num_predicate-1)] * A[:,0:(self.num_predicate-1)], dim=2)[0] 
        
        t = t.repeat(1, A.shape[0])
        t = t.reshape(len(t),A.shape[0])
        sigm = torch.sigmoid((t - max_body_time)) # max body time < t: 1;   max body time > t: 0
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
        # print(self.weight)
        m = torch.distributions.gumbel.Gumbel(torch.zeros_like(scores), torch.ones_like(scores))
        g = m.sample()
        scores = scores + g
        # continuous top k
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
        # 1. update rule content： self.weight and relation
        self.model_parameter[head_predicate_idx]['weight'].requires_grad = False
        
        n=1
        for i in range(n):
            indices = np.arange(body_time.shape[0])
            np.random.shuffle(indices)
            body_time = body_time[indices,:]
            head_time = head_time[indices]
            
            for batch_idx in np.arange(0, num_batch, 1): # iterate batches
                # annealing
                if batch_idx % 10 == 0:
                    tau = torch.max(tau / 2, torch.ones(1) * 0.1)
                sample_ID_batch = np.arange(batch_idx*batch_size, (batch_idx+1)*batch_size, 1)
                self.weight.requires_grad = True
                optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
                avg_num = 10
                A = self.reparametrization(avg_num, tau)
                # iterate over a batch
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
                    grad_Weight = self.weight.grad
                    self.weight -= grad_Weight * 0.0001
                    self.weight = torch.maximum(self.weight, EPSILON)
                    self.weight = torch.minimum(self.weight, torch.from_numpy(np.array(np.finfo(np.float32).max)))
                    self.weight = F.normalize(self.weight, p=1, dim = 1)
                   

        m = 1
        for i in range(m):
            np.random.shuffle(indices)
            body_time = body_time[indices,:]
            head_time = head_time[indices]
            
            for batch_idx in np.arange(0, num_batch/2, 1): # iterate batches
                tau = torch.ones(1)*0.1
                sample_ID_batch = np.arange(batch_idx*batch_size, (batch_idx+1)*batch_size, 1)
                self.weight.requires_grad = True
                optimizer.zero_grad()  # set gradient zero at the start of a new mini-batch
                avg_num = 10
                A = self.reparametrization(avg_num, tau)
                # iterate over a batch
                data_sample = body_time[sample_ID_batch,:]
                t = head_time[sample_ID_batch,:]
                # calculate expectation               return matrix: batch_size x num rules
                log_p_star, relation_used = self.log_p_star(head_predicate_idx, t, pi, data_sample, A)
                log_avg = log_p_star          # batch_size x num rules 
                qzi = F.normalize(torch.exp(log_avg),p=1,dim=1)
                log_likelihood = torch.sum(qzi * log_avg) 
                loss = -log_likelihood  
                loss.backward(retain_graph=True)
                with torch.no_grad():
                    # update relation
                    for k,v in relation_used.items():
                        if len(v) > 0:
                            for j in range(len(v)):
                                grad_relation = self.relation[k][v[j]].grad
                                self.relation[k][v[j]] -= grad_relation * 0.0035 #0.25，0.3， 0.35
                                self.relation[k][v[j]] = torch.maximum(self.relation[k][v[j]], EPSILON)
                                self.relation[k][v[j]] = F.normalize(self.relation[k][v[j]], p=1, dim=0)
        

      
        # 2. update model parameters: self.model_parameter[head_predicate_idx]['weight']
        
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
            # take average
            avg_num = 10
            A = self.reparametrization(avg_num, tau)

            # iterate over a batch
            log_likelihood = torch.tensor([0], dtype=torch.float64)
            data_sample = body_time[sample_ID_batch,:]
            t = head_time[sample_ID_batch,:]
            # calculate expectation               
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
            # iterate over a batch
            log_likelihood = torch.tensor([0], dtype=torch.float64)
            data_sample = body_time[sample_ID_batch,:]
            t = head_time[sample_ID_batch,:]
            # calculate expectation               
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
        return pi.detach()

 
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
prior = torch.from_numpy(np.array([0.01, 0.99]))



tau = 20 * torch.ones(1) 
appearance = {}
record = []
record_single = []
count = 0
##### print key model information
for iter in range(iter_nums):
    prior = logic_model.optimize_EM_single(body_time, head_time, batch_size, head_pred, optimizer, prior, tau, num_samples)
    if iter == 0:
        prev_weight = torch.ones_like(logic_model.weight.clone())
    else:
        diff = torch.norm(prev_weight-logic_model.weight.clone(), p=1)
        prev_weight = logic_model.weight.clone()

    if count != 0:
        count = 0

    A_m =  logic_model.weight.clone()
    max_ind = torch.sort(A_m,descending=True)[1][:,:logic_model.k]
    print(A_m,max_ind)
    
    valid = np.sort(max_ind, axis=1)  
    for i in range(valid.shape[0]):  
        count = 0
        for j in range(valid.shape[1]):  
            if valid[i, j] >= head_pred:  
                count += 1
                valid[i, j] = head_pred + count - 1

    record.append(valid)
    print(record)
    print(appearance)

    # stopping rule 1
    if len(record) >= 4:
        for i in range(-4,-1,1):
            equal = True
            equal = np.array_equal(record[-1],record[i])
            if equal:
                count += 1
        print(count)
        if count == 3:
            break
    
    # stopping rule 2
    # single = np.zeros((len(record[-1]),1))
    # if len(record) >= 3:
    #     for i in range(-3,-1,1):
    #         for j in range(len(record[-1])):
    #             exist = []
    #             for k in range(len(record[-1])):
    #                 exist.append(record[-1][j] == record[i][k])
    #             if sum(exist) != 0:
    #                 single[j] += 1
    
    #     print(single)
    #     if sum(single>=1) != 0:
    #         ind_rule=[]
    #         for i in range(len(single)):
    #             if single[i]  >= 1:
    #                 ind_rule.append(i)
    #                 # break
    #         for ind_rule_i in ind_rule:
    #             element = record[-1][ind_rule_i]
    #             if str(element) in appearance.keys():
    #                 appearance[str(element)] += 1
    #             else:
    #                 appearance[str(element)] = 1

        # if sum(single==3) != 0:
        #     ind_rule = 0
        #     for i in range(len(single)):
        #         if single[i] == 3:
        #             ind_rule = i
        #             break
        #     rule = record[-1][ind_rule]
        #     in_rec = False
        #     for s in range(len(record_single)):
        #         if rule.issubset(record_single[s]):
        #             in_rec = True
        #     if in_rec == False:
        #         record_single.append(rule)
        # print(record_single)
        # if len(record_single) == len(record[-1]):
        #     break
        
    record = record[-10:]
                 