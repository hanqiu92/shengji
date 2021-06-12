import numpy as np
import random
import math

def transform_action_key(action):
    return tuple(sorted(action[0]))

MAX_THRES = 0.5

class MCTSNode:
    def __init__(self,level,father,prev_action,father_direc,prob_sa,c_puct=20):
        self.level = level
        self.father = father
        self.prev_action = prev_action
        # self.prob_sa = prob_sa
        # self.c_puct = c_puct
        self.pc = prob_sa * c_puct

        # self.father_direc = father_direc
        self.father_direc_bool = father_direc >= 0

        self.is_root = self.is_leaf = self.is_expanded = False
        self.is_evaluating = self.is_evaluated = False
        self.predict_value = np.nan
        self.children = None
        self.sim_count = self.sim_value = self.total_sim_value = 0
        self.select_value_tuple = self.get_select_value_tuple()

    def __repr__(self):
        return 'Node({},{:.2f},{:.2f})'.format(
            self.sim_count,self.sim_value,self.select_value_tuple[1])

    def set_root(self):
        self.is_root = True
        self.father = None

    def update_value(self,new_value):
        self.total_sim_value += new_value
        # self.sim_count += 1
        self.sim_value = self.total_sim_value / self.sim_count
        self.select_value_tuple = self.get_select_value_tuple()

    def update_count(self):
        self.sim_count += 1

    def get_select_value_tuple(self):
        if self.is_leaf:
            if self.father_direc_bool:
                return (self.predict_value,0)
            else:
                return (-self.predict_value,0)
        else:
            if self.father_direc_bool:
                return (self.sim_value,self.pc / (self.sim_count + 1))
            else:
                return (-self.sim_value,self.pc / (self.sim_count + 1))

    def get_select_value(self,father_sim_count_sqrt):
        return self.select_value_tuple[0] + self.select_value_tuple[1] * father_sim_count_sqrt

    def select_action_by_value_old(self):
        sim_count_sqrt = math.sqrt(self.sim_count)
        max_action_codes,max_value = (-1,),-1e8
        for action_codes,child in self.children.items():
            tmp_value = child.get_select_value(sim_count_sqrt)
            if tmp_value > max_value:
                max_action_codes,max_value = action_codes,tmp_value
        return max_action_codes,self.children[max_action_codes]

    def select_action_by_value(self):
        sim_count_sqrt = math.sqrt(self.sim_count)
        max_action_codes_list,max_value = [],-1e8
        for action_codes,child in self.children.items():
            tmp_value = child.get_select_value(sim_count_sqrt)
            if tmp_value > max_value + MAX_THRES:
                max_action_codes_list,max_value = [action_codes],tmp_value
            elif tmp_value > max_value - MAX_THRES:
                max_action_codes_list.append(action_codes)
        if len(max_action_codes_list) > 1:
            max_action_codes = max_action_codes_list[np.random.choice(len(max_action_codes_list))]
        else:
            max_action_codes = max_action_codes_list[0]
        return max_action_codes,self.children[max_action_codes]

    def select_action_by_value_root(self,new_prob_values,explore_alpha):
        sim_count_sqrt = math.sqrt(self.sim_count)
        max_action_codes,max_value = (-1,),-1e8
        for new_prob_value,(action_codes,child) in zip(new_prob_values,self.children.items()):
            tmp_value = self.select_value_tuple[0] + ((1 - explore_alpha) * self.select_value_tuple[1] + new_prob_value / (child.sim_count + 1)) * sim_count_sqrt
            if tmp_value > max_value:
                max_action_codes,max_value = action_codes,tmp_value
        return max_action_codes,self.children[max_action_codes]

    def select_action_by_count(self,temp=1):
        assert len(self.children) > 0
        if temp < 1e-3:
            values = []
            for action in self.state.action_set:
                action_codes = tuple(sorted(action[0]))
                values.append(self.children[action_codes].sim_count)
            values = np.array(values)
            a_idx = np.argmax(values)
            probs = np.zeros(len(values),)
            probs[a_idx] = 1
        else:
            inv_temp = 1 / temp
            values = []
            for action in self.state.action_set:
                action_codes = tuple(sorted(action[0]))
                values.append(self.children[action_codes].sim_count ** inv_temp)
            values = np.array(values)
            # values = np.array([child.sim_count ** temp for child in self.children])
            probs = values/np.sum(values)
            a_idx = np.random.choice(len(values),p=probs)
        action = self.state.action_set[a_idx]
        return a_idx,action,probs
