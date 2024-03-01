import numpy as np
import itertools
import random

##################################################################
np.random.seed(1)
class Logic_Model_Generator:
    # Assumption: body predicates: X1 ~ X5;  head predicate: X6
    # rules(related to X1~X5): 
    # f0: background; 
    # f1: X1 AND X2 AND X3, X1 before X2; 
    
    
    # (select rule based on prior distribution)
   
    def __init__(self):

        ### the following parameters are used to manually define the logic rules
        self.num_predicate = 11  
        self.num_rule_pred = 3
        self.num_formula = 2
        self.body_predicate_set = list(np.arange(0,(self.num_predicate-1),1)) 
        self.head_predicate_set = [self.num_predicate-1] 
        # the probability of each rule (f0, f1)
        self.prior=[0.2, 0.8]  # background f0: 0.02
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.Time_tolerance = 0.02

        ### the following parameters are used to generate synthetic data
        ### for the learning part, the following is used to claim variables
        self.model_parameter = {}

        head_predicate_idx = self.num_predicate - 1 # set the last predicate as the head
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] =  0.02 # b0
        formula_idx = 0 # background
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0 
        formula_idx = 1
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.4  
        

        self.logic_template = self.logic_rule()

    def logic_rule(self):
        # encode rule information
        logic_template = {}

        head_predicate_idx = self.num_predicate - 1 # set the last predicate as the head
        logic_template[head_predicate_idx] = {} 

        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = []
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = []
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = []


        formula_idx = 1
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0, 1, 2]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 1]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]

        
        return logic_template


    def intensity(self, cur_time, head_predicate_idx, history, rule_cur):
        formula_idx = rule_cur
        weight_formula = self.model_parameter[head_predicate_idx][formula_idx]['weight'] # r_k
        # check rule 
        feature_1 = np.prod(history[self.logic_template[head_predicate_idx][formula_idx]['body_predicate_idx']] <= cur_time)
        feature_2 = 1 
        for idx, temporal_relation in enumerate(self.logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx']):
            time_difference = history[temporal_relation[0]]-history[temporal_relation[1]]
            if self.logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'][idx] == 'BEFORE':
                feature_2 = feature_2 * (time_difference < -self.Time_tolerance)
            if self.logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'][idx] == 'AFTER':
                feature_2 = feature_2 * (time_difference > self.Time_tolerance)
            if self.logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'][idx] == 'EQUAL':
                feature_2 = feature_2 * (abs(time_difference) <= self.Time_tolerance)
        feature_formula = feature_1 * feature_2
        intensity = self.model_parameter[head_predicate_idx]['base'] + weight_formula * feature_formula
        return intensity


    def generate_data(self, num_sample, time_horizon):
        data={}

        for sample_ID in np.arange(0, num_sample, 1):  
            # initialize intensity function for body predicates
            body_intensity = np.append(np.array(random.choices([round(i,2) for i in list(np.arange(1,5,0.1))], k=self.num_rule_pred)), np.array(random.choices([round(i,2) for i in list(np.arange(0.01,0.8,0.02))], k=self.num_predicate-1-self.num_rule_pred)))
            
            data[sample_ID] = {}
            rule_set=[]
            data[sample_ID]["intensity"]=body_intensity

            # generate data (head predicates)
            head_predicate_idx = self.head_predicate_set[0]
            data[sample_ID]["head_predicate_time"]=[]
            # select a rule based on thr prior probability
            rule = random.choices(np.arange(0,self.num_formula,1),weights=self.prior)[0]
            rule_set.append(rule)
            # set the time of body predicates defined by selected rule as their original time, and other predicates as a large enough
            # time so that they won't affect the head predicate
            body_pred = np.array([time_horizon*10000]*self.num_rule_pred)  
            body_pred[self.logic_template[head_predicate_idx][rule]['body_predicate_idx']] = 1
            body_pred = np.append(body_pred, np.array([1]*(self.num_predicate-1-self.num_rule_pred)))

            data[sample_ID]["body_predicates_time"] = np.random.exponential(scale=1.0 / body_intensity) * body_pred
            
            # generate head predicate via accept and reject  （Ogata’s modified thinning algorithm）
            t = 0
            while True:
                intensity_max=self.model_parameter[self.num_predicate-1]["base"]+self.model_parameter[self.num_predicate-1][rule]["weight"]
                time_to_event = np.random.exponential(scale=1.0/intensity_max)
                t = t + time_to_event
                ratio = min(self.intensity(t, head_predicate_idx, data[sample_ID]["body_predicates_time"] , rule) / intensity_max, 1)
                flag = np.random.binomial(1, ratio)     # if flag = 1, accept, if flag = 0, regenerate
                if flag == 1: # accept
                    data[sample_ID]["head_predicate_time"].append(t)
                    break
            data[sample_ID]["real_rule"]=rule_set
        return data


if __name__ == "__main__":
    logic_model_generator = Logic_Model_Generator()
    data = logic_model_generator.generate_data(num_sample=20000, time_horizon=5)
    np.save('data_1rule11pred_20000.npy', data)
    print("done")
