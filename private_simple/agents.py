import numpy as np
import random
from states import rotate_state
from mcts import transform_action_key,MCTSNode

import asyncio
import ipywidgets as widgets
from IPython.display import display,clear_output

class Agent:
    def __init__(self):
        self.last_observation_id = 0

    def cache_samples(self):
        pass

    def observe(self,observation_id,prev_action,state,start_flag,end_flag,**kwargs):
        pass

    def act(self,state,**kwargs):
        action_set = state.action_set
        a_idx = np.random.choice(len(action_set))
        return action_set[a_idx]

    def act_probs(self,state,**kwargs):
        action_set = state.action_set
        return (np.ones((len(action_set),),dtype=float) / len(action_set))

    def update_train_meta(self):
        pass

class RandomAgent(Agent):
    def __init__(self):
        super().__init__()

class IpyAgent(Agent):
    def act(self,state,**kwargs):
        if_display_private = kwargs.get('if_display_private',True)
        if_display_public = kwargs.get('if_display_public',False)
        joint_header = '*' * 5 + ' ' * 3

        ## first, print out relevant infos
        print('*' * 50)
        suit_rule = ' / '.join([state.ruler.get_code_repr(state.ruler.suit_enc_cum_size[i])[0] for i in range(4)])
        score_rule = dict([(state.ruler.get_card_repr((1,key))[1],value) for key,value in state.ruler.score_map.items()])
        print(joint_header+'Interactive player: ({}) ({})'.format(suit_rule,score_rule))
        if len(state.best_codes) > 0:
            round_best_play = state.ruler.get_codes_repr(state.best_codes)
        else:
            round_best_play = '-' * 3
        stack_info = state.ruler.get_codes_repr(state.stack)
        if not if_display_public and state.curr_player != state.house:
            stack_info = '?'
        print(joint_header+'Game major {}. Round best play {}, round total score {}. Current game score {}. Stack {}.'.format(
            state.ruler.get_code_repr(-3),
            round_best_play,
            sum([state.ruler.get_codes_score(codes) for _,codes in state.round_plays],0),
            state.game_score,
            stack_info))
        print('*' * 50)

        headers,curr_plays,remain_plays = [],[],[]
        for player_idx in range(4):
            target_player = (state.curr_player - len(state.round_plays) + player_idx) % 4
            
            header = ''
            if target_player == state.house:
                header += '△ '
            if player_idx == len(state.round_plays):
                header += '--> '
            target_header = '{}Player {} '.format(header,target_player)
            
            if player_idx < len(state.round_plays):
                target_curr_play = state.ruler.get_codes_repr(state.round_plays[player_idx][1])
            else:
                target_curr_play = '-' * 3

            target_remain_structs = state.structs_by_player[target_player]
            target_remain_play = []
            for struct in target_remain_structs:
                struct_codes = state.ruler.get_codes_repr(sum([component[-1] for component in struct[1]],()))
                if struct_codes == '':
                    struct_codes = '-' * 3
                target_remain_play.append(struct_codes)
            target_remain_play = ' / '.join(target_remain_play)

            if not if_display_public and player_idx != len(state.round_plays):
                target_remain_play = '?'

            headers.append(target_header)
            curr_plays.append(target_curr_play)
            remain_plays.append(target_remain_play)

        header_max = np.max([len(header) for header in headers])
        curr_play_max = np.max([len(curr_play) for curr_play in curr_plays])
        for header,curr_play,remain_play in zip(headers,curr_plays,remain_plays):
            print(joint_header+'{} | this round: {} | remain cards: {}.'.format(
                    header.ljust(header_max),curr_play.ljust(curr_play_max),remain_play))
        print('*' * 50)

        ## now provide the options
        action_set = state.action_set
        label0,value0 = 'Please select from below:',([],None,False)
        options = [(label0,value0)] + \
                [(state.ruler.get_codes_repr(action[0]),action) for action in action_set]
        if len(action_set[0][0]) > 2 and not state.is_first:
            options += [('other combinations',None)] ## allow for combination of single cards
        dropbox = widgets.Dropdown(options=options,index=0,value=value0,label=label0)
        display(dropbox)
        future = asyncio.Future()
        def getvalue(change):
            future.set_result(change.new)
            dropbox.close()
        dropbox.observe(getvalue, 'value')
        return future
    
    def act_single(self,state):
        struct = state.structs_by_player[state.curr_player]
        codes = []
        for suit_enc in range(4):
            for component in struct[suit_enc][1]:
                codes.extend(component[-1])
        codes = sorted(codes)
        label0,value0 = 'Please select from below:',-1
        options = [(label0,value0)] + \
            [(state.ruler.get_code_repr(code),code) for code in codes] + \
            [('confirmed',-2)]
        dropbox = widgets.SelectMultiple(options=options,index=[0],value=[value0],label=[label0])
        display(dropbox)
        future = asyncio.Future()
        def getvalue(change):
            if 'confirmed' in dropbox.label:
                if len(dropbox.label) != state.round_num+1:
                    print('please recheck size of your play!',end='\r')
                else:
                    future.set_result(change.new)
                    dropbox.close()
            else:
                print(','.join([item for item in dropbox.label if item != label0]),end='\r')
        dropbox.observe(getvalue, 'value')
        return future

class RlAgent(Agent):
    def __init__(self,model):
        self.action_model = model
        
        self.debug_flag = False
        self.infer_flag = False
        self.sample_collect_flag = False
        self.last_observation_id = 0
        self.value_bootstrap_partial_sample = None
        self.value_bootstrap_samples_cache = []

    def cache_samples(self):
        for sample in self.value_bootstrap_samples_cache:
            self.action_model.add_experience(sample)
        self.value_bootstrap_samples_cache = []

    def observe(self,observation_id,prev_action,state,start_flag,end_flag,**kwargs):
        if observation_id != self.last_observation_id:
            self.last_observation_id = observation_id
            if self.sample_collect_flag:
                state_vec,actions_vec,direc = state.get_vecs()
                if start_flag:
                    self.value_bootstrap_partial_sample = None

                if end_flag:
                    ## end of game 
                    if self.value_bootstrap_partial_sample is not None and len(self.value_bootstrap_partial_sample) == 3:
                        prev_state_vec,prev_action_vec,prev_score = self.value_bootstrap_partial_sample
                        self.value_bootstrap_samples_cache.append((prev_state_vec,prev_action_vec,state.eval_score-prev_score,True,direc,state_vec,actions_vec))
                        self.value_bootstrap_partial_sample = None

                    self.cache_samples()
        
    def act(self,state,**kwargs):
        state_vec,actions_vec,direc = state.get_vecs()
        action_set = state.action_set
        Na = len(action_set)
        if Na == 1:
            a_idx = 0
        else:
            ## rotation
            r = np.random.choice(4)
            rotate_direc = (1 if r % 2 == 0 else -1)
            direc_tmp = direc * rotate_direc
            state_vec_tmp = rotate_state(state_vec,r)
            q_values,probs,action_samples = self.action_model.predict(state_vec_tmp,actions_vec,direc_tmp)
            q_values = q_values * rotate_direc
            
            # q_values,probs,action_samples = self.action_model.predict(state_vec,actions_vec,direc)
            if self.debug_flag:
                ruler = state.ruler
                sort_idxs = np.argsort(q_values)[::-1]
                debug_str = ';'.join(
                    [' {},{:.2f}'.format(ruler.get_codes_repr(action_set[i][0]),q_values[i]) 
                    for i in sort_idxs])
                print(debug_str)
                
                # for r in range(4):
                #     state_vec_tmp = rotate_state(state_vec,r)
                #     direc_tmp = direc * (1 if r % 2 == 0 else -1)
                #     qs_tmp,p_tmp,a_tmp = self.action_model.predict(state_vec_tmp,actions_vec,direc_tmp)
                #     print(r,qs_tmp,a_tmp)
                
            if self.infer_flag or random.random() < 0.9:
                a_idx = action_samples
            else:
                a_idx = np.random.choice(Na)

        if self.sample_collect_flag:
            ## value bootstrap samples: Q(state,action) -> next reward - current reward + (1 - end) * V(next_state,next_action,next_direc)
            ## manage the previous partial sample (if any) and init the next partial sample
            if self.value_bootstrap_partial_sample is not None and len(self.value_bootstrap_partial_sample) == 3:
                prev_state_vec,prev_action_vec,prev_score = self.value_bootstrap_partial_sample
                self.value_bootstrap_samples_cache.append((prev_state_vec,prev_action_vec,state.eval_score-prev_score,False,direc,state_vec,actions_vec))
            self.value_bootstrap_partial_sample = (state_vec,actions_vec[:,a_idx],state.eval_score)

        return action_set[a_idx]

class PGAgent(Agent):
    def __init__(self,model):
        self.action_model = model
        
        self.debug_flag = False
        self.infer_flag = False
        self.sample_collect_flag = False
        self.last_observation_id = 0
        self.policy_partial_sample = None
        self.policy_samples_cache = []

    def cache_samples(self):
        self.action_model.add_experience(self.policy_samples_cache)
        self.policy_samples_cache = []

    def observe(self,observation_id,prev_action,state,start_flag,end_flag,**kwargs):
        if observation_id != self.last_observation_id:
            self.last_observation_id = observation_id
            if self.sample_collect_flag:
                state_vec,actions_vec,direc = state.get_vecs()
                if start_flag:
                    self.policy_partial_sample = None

                if end_flag:
                    ## end of game 
                    if self.policy_partial_sample is not None and len(self.policy_partial_sample) == 5:
                        prev_state_vec,prev_actions_vec,prev_action_idx,prev_score,direc = self.policy_partial_sample
                        self.policy_samples_cache.append((prev_state_vec,prev_actions_vec,prev_action_idx,state.eval_score-prev_score,direc,True))
                        self.policy_partial_sample = None

                    self.cache_samples()
        
    def act(self,state,**kwargs):
        state_vec,actions_vec,direc = state.get_vecs()
        action_set = state.action_set
        Na = len(action_set)
        if Na == 1:
            a_idx = 0
        else:
            ## rotation
            r = np.random.choice(4)
            rotate_direc = (1 if r % 2 == 0 else -1)
            direc_tmp = direc * rotate_direc
            state_vec_tmp = rotate_state(state_vec,r)
            q_values,probs,action_samples = self.action_model.predict(state_vec_tmp,actions_vec,direc_tmp)
            a_idx = np.random.choice(Na,p=probs)

        if self.sample_collect_flag:
            ## policy samples: sequence of (state,actions,action_idx,return)
            ## manage the previous partial sample (if any) and init the next partial sample
            if self.policy_partial_sample is not None and len(self.policy_partial_sample) == 5:
                prev_state_vec,prev_actions_vec,prev_action_idx,prev_score,direc = self.policy_partial_sample
                self.policy_samples_cache.append((prev_state_vec,prev_actions_vec,prev_action_idx,state.eval_score-prev_score,direc,False))
            self.policy_partial_sample = (state_vec,actions_vec,a_idx,state.eval_score,direc)

        return action_set[a_idx]

class CFRAgent(Agent):
    def __init__(self,regret_models,policy_model,explore_alpha=0.1,explore_player=0):
        self.regret_models = regret_models
        self.policy_model = policy_model
        self.explore_alpha = explore_alpha
        self.explore_player = explore_player
        self.weight_const = 1
        
        self.rotation_flag = False
        self.debug_flag = False
        self.infer_flag = False
        self.sample_collect_flag = False
        self.last_observation_id = 0
        self.cfr_samples_partial_cache = []
        self.cfr_samples_cache = []

    def cache_samples(self):
        for sample in self.cfr_samples_cache:
            if (sample[1].shape[1]) == 1:
                continue
            self.regret_models[self.explore_player].add_experience(sample)
            self.policy_model.add_experience(sample)
        self.cfr_samples_cache = []

    def observe(self,observation_id,prev_action,state,start_flag,end_flag,**kwargs):
        if observation_id != self.last_observation_id:
            self.last_observation_id = observation_id
            if self.sample_collect_flag:
                if start_flag:
                    self.cfr_samples_partial_cache = []

                if end_flag:
                    ## end of game 
                    for sample in self.cfr_samples_partial_cache:
                        if len(sample) == 11:
                            state_vec,full_state_vec,actions_vec,action_idx,prev_score,values_baseline,probs,probs_sample,weight_r,weight_p,direc = sample
                            Na = actions_vec.shape[1]
                            curr_value = ((state.eval_score - prev_score) - values_baseline[action_idx]) / probs_sample[action_idx]
                            values = values_baseline.copy()
                            values[action_idx] += curr_value
                            regrets = values * direc
                            regrets -= np.sum(regrets * probs)
                            self.cfr_samples_cache.append((state_vec,actions_vec,values,regrets,probs,weight_r,weight_p,direc))

                    self.cfr_samples_partial_cache = []
                    self.cache_samples()

    def update_train_meta(self):
        self.weight_const += 1

    def get_predict_values(self,state):
        state_vec,actions_vec,direc = state.get_vecs()
        if (actions_vec.shape[1]) > 0:
            if self.rotation_flag:
                r = np.random.choice(4)
                rotate_direc = (1 if r % 2 == 0 else -1)
                direc_tmp = direc * rotate_direc
                state_vec_tmp = rotate_state(state_vec,r)
                regrets,values,avg_probs = self.regret_models[state.curr_player].predict(state_vec_tmp,actions_vec,direc_tmp)
                regrets = regrets * rotate_direc
            else:
                regrets,values,avg_probs = self.regret_models[state.curr_player].predict(state_vec,actions_vec,direc)
            return regrets,values,avg_probs
        else:
            return None,None,None

    def get_predict_values_pnet(self,state):
        state_vec,actions_vec,direc = state.get_vecs()
        if (actions_vec.shape[1]) > 0:
            if self.rotation_flag:
                r = np.random.choice(4)
                rotate_direc = (1 if r % 2 == 0 else -1)
                direc_tmp = direc * rotate_direc
                state_vec_tmp = rotate_state(state_vec,r)
                regrets,values,avg_probs = self.policy_model.predict(state_vec_tmp,actions_vec,direc_tmp)
                regrets = regrets * rotate_direc
            else:
                regrets,values,avg_probs = self.policy_model.predict(state_vec,actions_vec,direc)
            return regrets,values,avg_probs
        else:
            return None,None,None

    def get_policy(self,state,regrets=None):
        if regrets is None:
            regrets = self.get_predict_values(state)[0]
        Na = len(regrets)
        regrets_p = np.maximum(regrets,0)
        if np.all(regrets <= 1e-8):
            if np.any(regrets <= -1e-8):
                probs = np.zeros((Na,))
                probs[np.argmax(regrets)] = 1
            else:
                probs = np.ones((Na,)) / Na
        else:
            probs = regrets_p / np.sum(regrets_p)
        if state.curr_player == self.explore_player:
            probs_sample = (1 - self.explore_alpha) * probs + self.explore_alpha * np.ones((Na,)) / Na
        else:
            probs_sample = probs
        return regrets,probs,probs_sample

    def update_prob_info(self,state,probs,probs_sample,a_idx):
        prob_info = state.prob_info
        prob_info_new = {'all':prob_info['all'] * probs_sample[a_idx],
                        'i':prob_info['i'].copy(),
                        '-i':prob_info['-i'].copy()}
        prob_info_new['i'][state.curr_player] *= probs[a_idx]
        for p_ in range(4):
            if p_ != state.curr_player:
                prob_info_new['-i'][p_] *= probs[a_idx]
        state.prob_info = prob_info_new

    def get_curr_regrets(self,state,probs,probs_sample,**kwargs):
        return None,None

    def get_curr_values_baseline(self,state,full_state):
        Na = len(state.action_set)
        return np.zeros((Na,))

    def act(self,state,**kwargs):
        action_set = state.action_set
        prob_info = state.prob_info
        full_state = kwargs.get('full_state')
        sample_weight_r = self.weight_const * prob_info['-i'][state.curr_player] / prob_info['all']
        sample_weight_p = self.weight_const * prob_info['i'][state.curr_player] / prob_info['all']

        Na = len(action_set)
        if Na == 1:
            a_idx = 0
            probs,probs_sample = np.array([1.0]),np.array([1.0])
        else:
            regrets,probs,probs_sample = self.get_policy(state)
            if self.debug_flag:
                ruler = state.ruler
                sort_idxs = np.argsort(regrets)[::-1]
                debug_str = ';'.join(
                    [' {},{:.2f},{:.2f},{:.2f}'.format(ruler.get_codes_repr(action_set[i][0]),regrets[i],probs[i],probs_sample[i]) 
                    for i in sort_idxs])
                print(debug_str)

            if self.infer_flag:
                probs = self.get_predict_values_pnet(state)[-1]
                probs = probs / np.sum(probs)
                probs_sample = probs
                if self.debug_flag:
                    ruler = state.ruler
                    sort_idxs = np.argsort(probs)[::-1]
                    debug_str = ';'.join(
                        [' {},{:.2f}'.format(ruler.get_codes_repr(action_set[i][0]),probs[i]) 
                        for i in sort_idxs])
                    print(debug_str)

            if np.sum(probs_sample) <= 0 or np.any(np.isnan(probs_sample)):
                probs_sample = np.ones((len(probs_sample),))
            a_idx = np.random.choice(len(probs_sample),p=probs_sample/np.sum(probs_sample))
            self.update_prob_info(full_state,probs,probs_sample,a_idx)
            
        if self.sample_collect_flag:
            curr_values,curr_regrets = self.get_curr_regrets(state,probs,probs_sample,**kwargs)
            state_vec,actions_vec,direc = state.get_vecs()
            if curr_regrets is not None:
                self.cfr_samples_cache.append((state_vec,actions_vec,curr_values,curr_regrets,probs,sample_weight_r,sample_weight_p,direc))
            else:
                curr_values_baseline = self.get_curr_values_baseline(state,full_state)
                full_state_vec,_,_ = full_state.get_full_vecs()
                self.cfr_samples_partial_cache.append((state_vec,full_state_vec,actions_vec,a_idx,state.eval_score,curr_values_baseline,probs,probs_sample,sample_weight_r,sample_weight_p,direc))

        return action_set[a_idx]

    def act_probs(self,state,**kwargs):
        action_set = state.action_set

        Na = len(action_set)
        if Na == 1:
            probs = np.array([1.0])
        else:
            if self.infer_flag:
                probs = self.get_predict_values_pnet(state)[-1]
                probs = probs / np.sum(probs)
            else:
                regrets,probs,probs_sample = self.get_policy(state)

        return probs

class CFRRolloutAgent(CFRAgent):
    def __init__(self,regret_models,policy_model,sim_env,explore_alpha=0.1,explore_player=0):
        super().__init__(regret_models,policy_model,explore_alpha,explore_player)
        self.sim_env = sim_env

    def get_curr_regrets(self,state,probs,probs_sample,**kwargs):
        action_set = state.action_set
        full_state = kwargs.get('full_state')
        curr_values = np.zeros((len(action_set),))
        for a_idx in range(len(action_set)):
            full_state_cp = full_state.copy()
            curr_action = action_set[a_idx]
            end_flag = False
            while not end_flag:
                _,end_flag,full_state_cp,state_cp = self.sim_env.step(full_state_cp,curr_action)
                if not end_flag:
                    if len(state_cp.action_set) > 1:
                        _,probs_cp,probs_sample_cp = self.get_policy(state_cp)
                        a_idx = np.random.choice(len(probs_cp),p=probs_cp/np.sum(probs_cp))
                    else:
                        a_idx = 0
                    curr_action = state_cp.action_set[a_idx]
            curr_values[a_idx] = state_cp.eval_score - state.eval_score
        curr_values_direc = curr_values * state.curr_direc
        curr_regrets = curr_values_direc - np.sum(probs * curr_values_direc)
        return curr_values,curr_regrets

class CFRExternalAgent(CFRAgent):
    def __init__(self,regret_models,policy_model,sim_env,explore_alpha=0.1,explore_player=0):
        super().__init__(regret_models,policy_model,explore_alpha,explore_player)
        self.sim_env = sim_env

    def get_value_estimate(self,full_state,state,end_flag):
        if end_flag:
            return state.eval_score
        else:
            action_set = state.action_set
            prob_info = state.prob_info
            Na = len(action_set)
            if prob_info['all'] > 0:
                sample_weight_r = self.weight_const * prob_info['-i'][state.curr_player] / prob_info['all']
                sample_weight_p = self.weight_const * prob_info['i'][state.curr_player] / prob_info['all']
            else:
                sample_weight_p = sample_weight_r = 0.0

            if Na == 1:
                a_idx = 0
                probs,probs_sample = np.array([1.0]),np.array([1.0])
            else:
                _,probs,_ = self.get_policy(state)
                probs_sample = probs
            
            if state.curr_player == self.explore_player:
                values = np.zeros((Na,),dtype=np.float32)
                avg_value = 0
                for a_idx in range(Na):
                    full_state_cp = full_state.copy()
                    curr_action = action_set[a_idx]
                    self.update_prob_info(full_state_cp,probs,probs_sample,a_idx)
                    _,end_flag,full_state_cp,state_cp = self.sim_env.step(full_state_cp,curr_action)
                    value = self.get_value_estimate(full_state_cp,state_cp,end_flag)
                    values[a_idx] = value
                    avg_value += value * probs[a_idx]

                if self.sample_collect_flag and abs(sample_weight_r) + abs(sample_weight_p) > 0:
                    curr_values = values - state.eval_score
                    curr_regrets = (values - avg_value) * state.curr_direc
                    state_vec,actions_vec,direc = state.get_vecs()
                    self.cfr_samples_cache.append((state_vec,actions_vec,curr_values,curr_regrets,probs,sample_weight_r,sample_weight_p,direc))

                return avg_value
            else:
                a_idx = np.random.choice(len(probs_sample),p=probs_sample/np.sum(probs_sample))
                self.update_prob_info(full_state,probs,probs_sample,a_idx)
                curr_action = state.action_set[a_idx]
                _,end_flag,full_state,state = self.sim_env.step(full_state,curr_action)
                return self.get_value_estimate(full_state,state,end_flag)

            return state.eval_score
            
    def get_curr_regrets(self,state,probs,probs_sample,**kwargs):
        action_set = state.action_set
        full_state = kwargs.get('full_state')
        curr_values = np.zeros((len(action_set),))
        for a_idx in range(len(action_set)):
            full_state_cp = full_state.copy()
            curr_action = action_set[a_idx]
            self.update_prob_info(full_state_cp,probs,probs_sample,a_idx)
            _,end_flag,full_state_cp,state_cp = self.sim_env.step(full_state_cp,curr_action)
            sample_value = self.get_value_estimate(full_state_cp,state_cp,end_flag)
            curr_values[a_idx] = sample_value - state.eval_score
        curr_values_direc = curr_values * state.curr_direc
        curr_regrets = curr_values_direc - np.sum(probs * curr_values_direc)
        return curr_values,curr_regrets
    
class CFRBaselineAgent(CFRAgent):
    def __init__(self,regret_models,policy_model,baseline_model,explore_alpha=0.1,explore_player=0):
        super().__init__(regret_models,policy_model,explore_alpha,explore_player)
        self.baseline_model = baseline_model
        self.baseline_samples_cache = []

    def cache_samples(self):
        super().cache_samples()
        for sample in self.baseline_samples_cache:
            self.baseline_model.add_experience(sample)
        self.baseline_samples_cache = []

    def observe(self,observation_id,prev_action,state,start_flag,end_flag,**kwargs):
        if observation_id != self.last_observation_id:
            if self.sample_collect_flag and end_flag:
                valid_samples = [sample for sample in self.cfr_samples_partial_cache if len(sample) == 11]
                for idx in range(len(valid_samples)):
                    state_vec,full_state_vec,actions_vec,action_idx,prev_score,values_baseline,probs,probs_sample,weight_r,weight_p,direc = valid_samples[idx]
                    if idx < len(valid_samples) - 1:
                        terminal_flag = False
                        state_vec_next,full_state_vec_next,actions_vec_next,action_idx_next,prev_score_next,values_baseline_next,probs_next,probs_sample_next,weight_r_next,weight_p_next,direc_next = valid_samples[idx+1]
                        reward = prev_score_next - prev_score
                    else:
                        terminal_flag = True
                        full_state_vec_next,actions_vec_next,probs_next,direc_next = None,None,None,None
                        reward = state.eval_score - prev_score
                    self.baseline_samples_cache.append((full_state_vec,actions_vec[:,action_idx],direc,reward,full_state_vec_next,actions_vec_next,probs_next,direc_next,terminal_flag))

        super().observe(observation_id,prev_action,state,start_flag,end_flag)

    def get_curr_values_baseline(self,state,full_state):
        state_vec,actions_vec,direc = full_state.get_full_vecs()
        if (actions_vec.shape[1]) > 0:
            if self.rotation_flag:
                r = np.random.choice(4)
                rotate_direc = (1 if r % 2 == 0 else -1)
                direc_tmp = direc * rotate_direc
                state_vec_tmp = rotate_state(state_vec,r)
                regrets,values = self.baseline_model.predict(state_vec_tmp,actions_vec,direc_tmp)
                regrets = regrets * rotate_direc
            else:
                regrets,values = self.baseline_model.predict(state_vec,actions_vec,direc)
            return values
        else:
            Na = len(state.action_set)
            return np.zeros((Na,))

class FPAgent(Agent):
    def __init__(self,policy_model,agg_policy_model):
        self.policy_model = policy_model
        self.agg_policy_model = agg_policy_model
        self.q_model = None
        self.weight_const = 1
        
        self.status_flag = 'infer' ## 'BR', 'avg', 'infer'
        self.rotation_flag = False
        self.debug_flag = False
        self.infer_flag = False
        self.sample_collect_flag = False
        self.last_observation_id = 0
        self.value_bootstrap_partial_sample = None
        self.value_bootstrap_samples_cache = []
        self.policy_samples_cache = []

    def cache_samples(self):
        if self.q_model is not None:
            for sample in self.value_bootstrap_samples_cache:
                self.q_model.add_experience(sample)
        self.value_bootstrap_samples_cache = []

        for sample in self.policy_samples_cache:
            self.policy_model.add_experience(sample)
            self.agg_policy_model.add_experience(sample)
        self.policy_samples_cache = []

    def observe(self,observation_id,prev_action,state,start_flag,end_flag,**kwargs):
        if observation_id != self.last_observation_id:
            self.last_observation_id = observation_id
            if self.sample_collect_flag:
                state_vec,actions_vec,direc = state.get_vecs()
                if start_flag:
                    self.value_bootstrap_partial_sample = None

                if end_flag:
                    ## end of game 
                    if self.value_bootstrap_partial_sample is not None and len(self.value_bootstrap_partial_sample) == 3:
                        prev_state_vec,prev_action_vec,prev_score = self.value_bootstrap_partial_sample
                        self.value_bootstrap_samples_cache.append((prev_state_vec,prev_action_vec,state.eval_score-prev_score,True,direc,state_vec,actions_vec))
                        self.value_bootstrap_partial_sample = None

                    self.cache_samples()

    def update_train_meta(self):
        self.weight_const += 1

    def act(self,state,**kwargs):
        state_vec,actions_vec,direc = state.get_vecs()
        action_set = state.action_set

        Na = len(action_set)
        if Na == 1:
            a_idx = 0
        else:
            if self.rotation_flag:
                r = np.random.choice(4)
                rotate_direc = (1 if r % 2 == 0 else -1)
                direc_tmp = direc * rotate_direc
                state_vec_tmp = rotate_state(state_vec,r)

            if self.status_flag in ('BR','avg'):
                if self.rotation_flag:
                    q_values,_,action_samples = self.q_model.predict(state_vec_tmp,actions_vec,direc_tmp)
                    q_values = q_values * rotate_direc
                else:
                    q_values,_,action_samples = self.q_model.predict(state_vec,actions_vec,direc)
                if self.debug_flag:
                    ruler = state.ruler
                    sort_idxs = np.argsort(q_values)[::-1]
                    debug_str = ';'.join(
                        [' {},{:.2f}'.format(ruler.get_codes_repr(action_set[i][0]),q_values[i]) 
                        for i in sort_idxs])
                    print(debug_str)
                    
                if self.infer_flag or random.random() < 0.9:
                    a_idx = action_samples
                else:
                    a_idx = np.random.choice(Na)
            else:
                if self.rotation_flag:
                    _,_,probs = self.policy_model.predict(state_vec_tmp,actions_vec,direc_tmp)
                else:
                    _,_,probs = self.policy_model.predict(state_vec,actions_vec,direc)

                if self.debug_flag:
                    ruler = state.ruler
                    sort_idxs = np.argsort(probs)[::-1]
                    debug_str = ';'.join(
                        [' {},{:.2f}'.format(ruler.get_codes_repr(action_set[i][0]),probs[i]) 
                        for i in sort_idxs])
                    print(debug_str)

                if np.sum(probs) <= 0 or np.any(np.isnan(probs)):
                    probs_sample = np.ones((len(probs),))
                a_idx = np.random.choice(len(probs),p=probs/np.sum(probs))
            
        if self.sample_collect_flag:
            if self.status_flag == 'BR':
                if self.value_bootstrap_partial_sample is not None and len(self.value_bootstrap_partial_sample) == 3:
                    prev_state_vec,prev_action_vec,prev_score = self.value_bootstrap_partial_sample
                    self.value_bootstrap_samples_cache.append((prev_state_vec,prev_action_vec,state.eval_score-prev_score,False,direc,state_vec,actions_vec))
                self.value_bootstrap_partial_sample = (state_vec,actions_vec[:,a_idx],state.eval_score)
            elif self.status_flag == 'avg':
                curr_values = curr_regrets = probs = np.zeros((Na,))
                probs[a_idx] = 1
                self.policy_samples_cache.append((state_vec,actions_vec,curr_values,curr_regrets,probs,self.weight_const,self.weight_const,direc))

        return action_set[a_idx]

    def act_probs(self,state,**kwargs):
        action_set = state.action_set

        Na = len(action_set)
        if Na == 1:
            probs = np.array([1.0])
        else:
            if self.infer_flag:
                probs = self.get_predict_values_pnet(state)[-1]
                probs = probs / np.sum(probs)
            else:
                regrets,probs,probs_sample = self.get_policy(state)

        return probs
