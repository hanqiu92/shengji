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

    def observe(self,observation_id,prev_action,state,start_flag,end_flag):
        pass

    def act(self,state):
        action_set = state.action_set
        a_idx = np.random.choice(len(action_set))
        return action_set[a_idx]

class RandomAgent(Agent):
    def __init__(self,seed):
        self.last_observation_id = 0
        self.seed = seed

    def act(self,state):
        np.random.seed(self.seed)
        action_set = state.action_set
        a_idx = np.random.choice(len(action_set))
        return action_set[a_idx]

class IpyAgent(Agent):
    def act(self,state):
        ## first, print out relevant infos
        print('-' * 50)
        print('Interactive player:')
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

            headers.append(target_header)
            curr_plays.append(target_curr_play)
            remain_plays.append(target_remain_play)

        header_max = np.max([len(header) for header in headers])
        curr_play_max = np.max([len(curr_play) for curr_play in curr_plays])
        for header,curr_play,remain_play in zip(headers,curr_plays,remain_plays):
            print('{} | this round: {} | remain cards: {}.'.format(
                    header.ljust(header_max),curr_play.ljust(curr_play_max),remain_play))
        print('-' * 25)

        if len(state.best_codes) > 0:
            round_best_play = state.ruler.get_codes_repr(state.best_codes)
        else:
            round_best_play = '-' * 3
        print('Game major {}. Round best play {}, round total score {}. Current game score {}. Stack {}.'.format(
            state.ruler.get_code_repr(51),
            round_best_play,
            sum([state.ruler.get_codes_score(codes) for _,codes in state.round_plays],0),
            state.game_score,
            state.ruler.get_codes_repr(state.stack)))
        print('-' * 25)

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

    def observe(self,observation_id,prev_action,state,start_flag,end_flag):
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
        
    def act(self,state):
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

class MCTSAgent(Agent):
    def __init__(self,model,sim_env,N_search=1000,c_puct=400,temp=1,explore_alpha=0.25,batch_size=8):
        self.action_model = model
        self.sim_env = sim_env
        self.N_search = N_search
        self.c_puct = c_puct
        self.temp = temp
        self.explore_alpha = explore_alpha
        self.batch_size = batch_size
        self.root_node = None
        
        self.debug_flag = False
        self.infer_flag = False
        self.sample_collect_flag = False
        self.last_observation_id = 0
        self.samples_cache = []
        self.samples_partial_cache = []

    def cache_samples(self):
        for sample in self.samples_cache:
            self.action_model.add_experience(sample,exp_pool_id=0)
        self.samples_cache = []

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
                for sample in self.samples_partial_cache:
                    if len(sample) == 6:
                        if len(sample[4]) == 1:
                            self.samples_cache.append((sample[0],sample[1],sample[2],state.eval_score-sample[3],sample[4],[state.eval_score-sample[3]]))
                        else:
                            self.samples_cache.append((sample[0],sample[1],sample[2],state.eval_score-sample[3],sample[4],sample[5]))
                self.samples_partial_cache = []

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
        self.root_node = MCTSNode(level=0,father=None,father_direc=0,prev_action=None,prob_sa=1,c_puct=self.c_puct)
        self.init_root_node(self.root_node,state)

    def update_mcts_tree(self,prev_action,state,end_flag):
        if not end_flag:
            if self.root_node is None or prev_action is None:
                # print('restart mcts tree!')
                self.init_mcts_tree(state)
            else:
                prev_codes = tuple(sorted(prev_action[0]))
                if prev_codes in self.root_node.children:
                    self.root_node = self.root_node.children[prev_codes]
                    if not self.root_node.is_expanded:
                        self.init_root_node(self.root_node,state)
                    self.root_node.set_root()
                else:
                    self.init_mcts_tree(state)

    def init_root_node(self,node,state):
        q_values,probs = self.get_predict_values(state)
        node.state = state
        node.predict_value = state.eval_score
        ## add dirichlet noise to the prior
        dirichlet_noise = np.random.dirichlet(np.full((len(probs),),0.3))
        node.child_probs = probs * (1 - self.explore_alpha) + self.explore_alpha * dirichlet_noise
        # node.child_probs = probs
        self.expand_node(node)
        node.set_root()

    def evaluate_node(self,node):
        _,end_flag,state = self.sim_env.step(node.father.state.copy(),node.prev_action,if_display=False)
        if end_flag:
            node.is_leaf = True
            node.predict_value = state.eval_score
            self.update_node(node)
        else:
            node.state = state
            node.is_evaluating = True

    def expand_node(self,node):
        next_level = node.level + 1
        # na = len(node.state.action_set)
        node.children = {}
        for prev_action,prob_sa in (zip(node.state.action_set,node.child_probs)):
            node.children[transform_action_key(prev_action)] = MCTSNode(level=next_level,father=node,father_direc=node.state.curr_direc,prev_action=prev_action,
                prob_sa=prob_sa,
                # prob_sa=1/na,
                c_puct=self.c_puct)

        node.is_expanded = True

    def update_node(self,node):
        # print(node.predict_value)
        value = node.predict_value
        node.update_value(value)
        while (node.father is not None and not node.is_root):
            node = node.father
            node.update_value(value)

    def predict_node_batch(self,nodes_queue):
        if len(nodes_queue) > 1:
            max_mask_len = 0
            s_batch,mask_len_batch,direc_batch = [],[],[]
            for node in nodes_queue:
                state = node.state
                state_vec,actions_vec,direc = state.get_vecs()
                mask_len = len(state.action_set)
                s_batch.append(state_vec)
                mask_len_batch.append(mask_len)
                direc_batch.append(direc)
                if mask_len > max_mask_len:
                    max_mask_len = mask_len

            As = np.zeros((len(nodes_queue),max_mask_len,55),dtype=np.float32)
            for idx,node in enumerate(nodes_queue):
                state = node.state
                mask_len = len(state.action_set)
                As[idx:(idx+1),:mask_len] = state.actions_vec
            v_batch,probs_batch = \
                self.action_model.predict_batch(
                    np.concatenate(s_batch,axis=0),As,
                    np.array(mask_len_batch,dtype=np.int64),
                    np.array(direc_batch,dtype=np.float32))
        else:
            state = nodes_queue[0].state
            state_vec,actions_vec,direc = state.get_vecs()
            v_batch,probs_batch = \
                self.action_model.predict_batch(
                    state_vec,
                    actions_vec,
                    np.array([len(state.action_set)],dtype=np.int64),
                    np.array([direc],dtype=np.float32))

        for node,v,probs in zip(nodes_queue,v_batch,probs_batch):
            na = len(node.state.action_set)
            node.predict_value = node.state.eval_score + v
            node.child_probs = np.maximum(probs[:na],1e-2)
            node.is_evaluating = False
            node.is_evaluated = True
            self.update_node(node)

    def mcts_search(self,state):
        action_set = state.action_set
        Na = len(action_set)
        if Na == 1: ## TODO: consider remove this part for more consistent training samples
            action = action_set[0]
            if self.sample_collect_flag:
                ## value samples: Q(state,action) -> final reward - current reward
                ## only init parts here; append the final reward at the end of the game
                state_vec,actions_vec,direc = state.get_vecs()
                self.samples_partial_cache.append((state_vec,actions_vec,direc,state.eval_score,[1],[state.eval_score]))
        else:
            if self.root_node is None:
                self.init_mcts_tree(state)

            N_total = max(self.N_search,Na)
            # dirichlet_noise = np.random.dirichlet(np.full((Na,),0.3),N_total+1)
            # dirichlet_noise_value = dirichlet_noise * self.explore_alpha * self.c_puct
            
            # ## print fix depth tree
            # ruler = state.ruler
            # curr_node_prints = [([],[],self.root_node)]
            # for level in range(2):
            #     str_ = []
            #     curr_node_prints_next = []
            #     for key_father,key_curr,node in curr_node_prints:
            #         str_.append('({})->({}),{:.0f},{:.0f},{:.0f}'.format(ruler.get_codes_repr(key_father),ruler.get_codes_repr(key_curr),node.predict_value,node.sim_count,node.sim_value))
            #         if node.children is not None:
            #             for key,value in node.children.items():
            #                 curr_node_prints_next.append((key_curr,key,value))
            #     curr_node_prints = curr_node_prints_next
            #     if level > 0:
            #         print(level,'; '.join(str_))

            counts = 0
            nodes_queue = []
            while counts < N_total:
                if len(nodes_queue) >= self.batch_size:
                    self.predict_node_batch(nodes_queue)
                    nodes_queue = []

                # while len(nodes_queue) < self.batch_size and counts < N_total:
                counts += 1
                curr_node = self.root_node
                # print(counts,self.root_node.sim_count)
                while True:
                    if curr_node.is_evaluating:
                        self.predict_node_batch(nodes_queue)
                        nodes_queue = []

                    curr_node.update_count()

                    if curr_node.is_leaf:
                        self.update_node(curr_node)
                        break

                    if not curr_node.is_expanded:
                        if curr_node.is_evaluated:
                            self.expand_node(curr_node)
                        else:
                            self.evaluate_node(curr_node)
                            if curr_node.is_evaluating:
                                nodes_queue.append(curr_node)
                            break

                    _,curr_node = curr_node.select_action_by_value()
                    # if curr_node.is_root:
                    #     _,curr_node = curr_node.select_action_by_value_root(dirichlet_noise_value[counts],self.explore_alpha)
                    # else:
                    #     _,curr_node = curr_node.select_action_by_value()

                # try:
                #     ruler = state.ruler
                #     kvs = list(self.root_node.children.items())
                #     values = [kv[1].sim_count for kv in kvs]
                #     sort_idxs = np.argsort(values)[::-1]
                #     debug_str = []
                #     for idx in sort_idxs:
                #         child = kvs[idx][1]
                #         debug_str.append(' {},{:.0f},{:.0f},{:.0f},{:.0f}'.format(ruler.get_codes_repr(kvs[idx][0]),child.sim_count,child.sim_value,child.predict_value,child.get_select_value(np.sqrt(self.root_node.sim_count))))
                    
                #     debug_str = ';'.join(debug_str)
                #     print(counts,debug_str)
                # except Exception:
                #     pass
                
            if len(nodes_queue) > 0:
                self.predict_node_batch(nodes_queue)
                nodes_queue = []

            if self.infer_flag:
                temp = 0
            else:
                if state.round_ >= 10:
                    temp = 0
                else:
                    temp = self.temp * (1 - state.round_ / 10.0)

            a_idx,action,a_probs,q_values = self.root_node.select_action_by_count_aug(temp=temp)
            # _,_,a_probs,q_values = self.root_node.select_action_by_count_aug(temp=1)

            if self.debug_flag:
                # print(state.round_,self.root_node.child_probs)
                # print(state.round_,a_probs,temp)

                try:
                    ruler = state.ruler
                    kvs = list(self.root_node.children.items())
                    values = [kv[1].sim_count for kv in kvs]
                    sort_idxs = np.argsort(values)[::-1]
                    debug_str = []
                    for idx in sort_idxs[:5]:
                        child = kvs[idx][1]
                        debug_str.append(' {}-->N:{:d},V:{:.0f},{:.0f},p:{:.2f},{:.2f}'.format(ruler.get_codes_repr(kvs[idx][0]),child.sim_count,child.sim_value,child.predict_value,a_probs[idx],child.pc/self.c_puct))
                    
                    debug_str = ';'.join(debug_str)
                    print(debug_str)
                except Exception:
                    pass

            if self.sample_collect_flag:
                state_vec,actions_vec,direc = self.root_node.state.get_vecs()

                self.samples_partial_cache.append((state_vec,actions_vec,direc,state.eval_score,a_probs,q_values))
                # ## policy samples: P(state,actions,direc) -> action_probs
                # self.policy_samples_cache.append((state_vec,actions_vec,direc,a_probs))
                # ## value samples: Q(state,action) -> final reward - current reward
                # ## only init parts here; append the final reward at the end of the game
                # self.value_samples_partial_cache.append((state_vec,actions_vec[:,a_idx],state.eval_score))
            
        return action
