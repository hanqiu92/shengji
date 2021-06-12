import numpy as np
import random
from states import rotate_state
from mcts import transform_action_key,OldMCTSNode

class RlAgent(Agent):
    def __init__(self,idx,model):
        self.idx = idx
        self.reset_cards()

        self.action_model = model
        self.dataset = []
        self.curr_state = None
        self.curr_action = None
        self.debug_flag = False
        
    def get_state_vector(self,global_info):
        if global_info.house == self.idx:
            state_vector = global_info.state_vec_house
        else:
            state_vector = global_info.state_vec
        return state_vector
        
    def get_reward_vector(self,global_info,stack_score=0):
        if len(global_info.past_best_player) > 0:
            best_player,round_score = global_info.past_best_player[-1],global_info.past_round_score[-1]
            reward = (round_score + stack_score) * (1 if (best_player - self.idx) % 2 == 0 else -1)
            is_new_round = False
        else:
            reward,is_new_round = 0,True
        return reward,is_new_round
        
    def update_dataset(self,global_info,action_options=None,stack_score=0,is_terminated=False):
        ruler = global_info.ruler
        if not is_terminated:
            state = self.get_state_vector(global_info)
            actions = np.zeros((len(action_options),55))
            for i,a in enumerate(action_options):
                for code in a[0]:
                    actions[i,code] += 1
                    actions[i,-1] += ruler.codes_score[code]
        else:
            state,actions = None,None
        past_reward,is_new_round = self.get_reward_vector(global_info,stack_score)
        if self.curr_state is not None and self.curr_action is not None and not is_new_round:
            self.dataset.append([self.curr_state,self.curr_action,past_reward,is_terminated,state,actions])
            self.action_model.add_experience(self.dataset[-1])
        return state,actions
        
        
    def get_action(self,global_info,action_options):
        ruler = global_info.ruler
        state,actions = self.update_dataset(global_info,action_options)
        q_values,probs,action_samples = self.action_model.predict(state,actions)
        if self.debug_flag:
            sort_idxs = np.argsort(probs)[::-1]
            debug_str = ';'.join(
                [' {},{:.2f},{:.2f}'.format(ruler.get_codes_repr(action_options[i][0]),q_values[i],probs[i]) 
                 for i in sort_idxs if probs[i] > 1e-3])
            print(debug_str)
        a_idx = action_samples
        self.curr_state = state
        self.curr_action = actions[a_idx]
        return action_options[a_idx]

class OldMCTSAgent(Agent):
    def __init__(self,model,sim_env,N_search=1000,max_depth=10,c_puct=20,temp=1,explore_alpha=0.25):
        self.action_model = model
        self.sim_env = sim_env
        self.N_search = N_search
        self.max_depth = max_depth
        self.c_puct = c_puct
        self.temp = temp
        self.explore_alpha = explore_alpha
        self.root_node = None
        
        self.debug_flag = False
        self.infer_flag = False
        self.sample_collect_flag = False
        self.last_observation_id = 0
        self.policy_samples_cache = []
        self.value_samples_cache = []
        self.value_samples_partial_cache = []

    def cache_samples(self):
        for sample in self.policy_samples_cache:
            self.action_model.add_experience(sample,exp_pool_id=0)
        self.policy_samples_cache = []
        for sample in self.value_samples_cache:
            self.action_model.add_experience(sample,exp_pool_id=1)
        self.value_samples_cache = []

    def observe(self,observation_id,prev_action,state,start_flag,end_flag):
        if observation_id != self.last_observation_id:
            self.last_observation_id = observation_id
            if start_flag:
                self.init_mcts_tree(state)
            else:
                self.update_mcts_tree(prev_action,state,end_flag)

            ## check if the game ends
            if self.sample_collect_flag and end_flag:
                ## value samples: Q(state,action) -> final reward - current reward
                ## update the value samples here
                for sample in self.value_samples_partial_cache:
                    if len(sample) == 3:
                        self.value_samples_cache.append((sample[0],sample[1],state.eval_score-sample[2]))
                self.value_samples_partial_cache = []

                self.cache_samples()

    def act(self,state):
        return self.mcts_search(state)

    def get_predict_values(self,state):
        if (state.actions_vec.shape[1]) > 0:
            q_values,probs,_ = self.action_model.predict(*state.get_vecs())
            return q_values + state.eval_score,probs
        else:
            return None,None

    def init_mcts_tree(self,state):
        self.root_node = OldMCTSNode(level=0,father=None,prev_action=None,father_direc=0,predict_value=state.eval_score,prob_sa=1,c_puct=self.c_puct)
        q_values,probs = self.get_predict_values(state)
        self.root_node.expand(state=state,end_flag=False,q_values=q_values,probs=probs,explore_alpha=self.explore_alpha,c_puct=self.c_puct)
        self.root_node.set_root()

    def update_mcts_tree(self,prev_action,state,end_flag):
        if not end_flag:
            if self.root_node is None or prev_action is None:
                self.init_mcts_tree(state)
            else:
                prev_codes = tuple(sorted(prev_action[0]))
                if prev_codes in self.root_node.children:
                    self.root_node = self.root_node.children[prev_codes]
                    if not self.root_node.is_expanded:
                        q_values,probs = self.get_predict_values(state)
                        self.root_node.expand(state=state,end_flag=end_flag,q_values=q_values,probs=probs,explore_alpha=self.explore_alpha,c_puct=self.c_puct)
                    self.root_node.set_root()
                else:
                    self.init_mcts_tree(state)

    def mcts_search(self,state):
        action_set = state.action_set
        Na = len(action_set)
        if Na == 1:
            action = action_set[0]
            if self.sample_collect_flag:
                ## value samples: Q(state,action) -> final reward - current reward
                ## only init parts here; append the final reward at the end of the game
                state_vec,actions_vec,direc = state.get_vecs()
                self.value_samples_partial_cache.append((state_vec,actions_vec[:,0],state.eval_score))
        else:
            if self.root_node is None:
                self.init_mcts_tree(state)

            max_search_level = self.root_node.level + self.max_depth

            for iter_ in range(max(self.N_search,Na)):
                curr_node = self.root_node

                if self.debug_flag:
                    if len(self.root_node.children) <= 5:
                        print(iter_,[child for child in self.root_node.children.values()])

                while curr_node.is_expanded and not curr_node.is_leaf and curr_node.level < max_search_level:
                    _,curr_node = curr_node.select_action_by_value()

                if not curr_node.is_expanded:
                    _,end_flag,state_tmp = self.sim_env.step(curr_node.father.state.copy(),curr_node.prev_action,if_display=False)
                    ## error check, remove later
                    if not end_flag and (state_tmp.action_set is None or len(state_tmp.action_set) == 0):
                        print(state_tmp.codes_by_player,state_tmp.action_set)
                    q_values,probs = self.get_predict_values(state_tmp)
                    curr_node.expand(state=state_tmp,end_flag=end_flag,q_values=q_values,probs=probs,explore_alpha=self.explore_alpha,c_puct=self.c_puct)

                predict_value = curr_node.predict_value
                curr_node.update_value(predict_value)
                while (curr_node.father is not None and not curr_node.is_root):
                    curr_node = curr_node.father
                    curr_node.update_value(predict_value)

            if self.debug_flag:
                ruler = state.ruler
                values = [child.get_value() for child in self.root_node.children.values()]
                sort_idxs = np.argsort(values)[::-1]
                debug_str = ';'.join(
                    [' {},{:.2f}'.format(ruler.get_codes_repr(action_set[i][0]),values[i]) 
                    for i in sort_idxs])
                return debug_str

            a_idx,action,a_probs = self.root_node.select_action_by_count(temp=self.temp)

            if self.sample_collect_flag:
                state_vec,actions_vec,direc = self.root_node.state.get_vecs()
                ## policy samples: P(state,actions,direc) -> action_probs
                self.policy_samples_cache.append((state_vec,actions_vec,direc,a_probs))
                ## value samples: Q(state,action) -> final reward - current reward
                ## only init parts here; append the final reward at the end of the game
                self.value_samples_partial_cache.append((state_vec,actions_vec[:,a_idx],state.eval_score))
            
        return action

