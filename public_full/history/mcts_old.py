import numpy as np
import random
import math
import time
from multiprocessing import Process, Lock, Manager, Value
from agents import MCTSAgent

def transform_action_key(action):
    return tuple(sorted(action[0]))

class OldMCTSNode:
    def __init__(self,level,father,prev_action,father_direc,predict_value,prob_sa,c_puct=20):
        self.level = level
        self.father = father
        self.prev_action = prev_action
        self.predict_value = predict_value
        self.prob_sa = prob_sa
        self.c_puct = c_puct
        self.pc = self.prob_sa * self.c_puct

        # self.father_direc = father_direc
        self.father_direc_bool = father_direc >= 0

        self.is_root = self.is_leaf = self.is_expanded = False
        self.sim_value = predict_value
        self.sim_count = self.total_sim_value = 0
        self.select_value_tuple = self.get_select_value_tuple()

    def __repr__(self):
        return 'Node({},{:.2f},{:.2f})'.format(
            self.sim_count,self.sim_value,self.select_value_tuple[1])

    def set_root(self):
        self.is_root = True
        self.father = None

    def expand(self,state,end_flag,q_values,probs,explore_alpha=0.25,c_puct=20):
        if end_flag:
            self.is_leaf = True
            self.predict_value = state.eval_score
        else:
            self.state = state
            self.children = dict()
            na = len(probs)
            if self.is_root:
                ## add noise
                probs = (1 - explore_alpha) * probs + \
                        explore_alpha * np.random.dirichlet(np.full((len(probs),),0.3))
            for prev_action,predict_value,prob_sa in (zip(state.action_set,q_values,probs)):
                self.children[transform_action_key(prev_action)] = (OldMCTSNode(level=self.level+1,father=self,
                                            prev_action=prev_action,
                                            father_direc=state.curr_direc,
                                            predict_value=predict_value,
                                            # prob_sa=prob_sa,
                                            prob_sa=1/na,
                                            c_puct=c_puct))
        self.is_expanded = True

    def update_value(self,new_value):
        self.total_sim_value += new_value
        self.sim_count += 1
        self.sim_value = self.total_sim_value / self.sim_count
        self.select_value_tuple = self.get_select_value_tuple()

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

    def select_action_by_value(self):
        sim_count_sqrt = math.sqrt(self.sim_count)
        max_action_codes,max_value = (-1,),-1e8
        for action_codes,child in self.children.items():
            tmp_value = child.get_select_value(sim_count_sqrt)
            if tmp_value > max_value:
                max_action_codes,max_value = action_codes,tmp_value
        return max_action_codes,self.children[max_action_codes]

    def select_action_by_count(self,temp=1):
        assert len(self.children) > 0
        values = []
        for action in self.state.action_set:
            action_codes = tuple(sorted(action[0]))
            values.append(self.children[action_codes].sim_count ** temp)
        values = np.array(values)
        # values = np.array([child.sim_count ** temp for child in self.children])
        probs = values/np.sum(values)
        a_idx = np.random.choice(len(values),p=probs)
        action = self.state.action_set[a_idx]
        return a_idx,action,probs

class NodeMeta:
    '''
    creation during father expansion
    '''
    def __init__(self,level,father_state,father_action,father_direc,prob_sa,c_puct=20):
        self.level = level
        self.father_state = father_state
        self.father_action = father_action
        self.father_direc = father_direc
        self.prob_sa = prob_sa
        self.c_puct = c_puct
        self.pc = prob_sa * c_puct

class NodeStatus:
    '''
    update in expansion
    '''
    def __init__(self,father_idx,node_value):
        self.is_root = self.is_leaf = self.is_expanded = False
        self.children_idx = dict()
        self.father_idx = father_idx
        self.node_value = node_value

    def set_root(self):
        self.is_root = True
        self.father = None

class NodeStats:
    '''
    update in real time
    '''
    def __init__(self,node_value,node_status,node_meta):
        self.sim_value = node_value
        self.sim_count = self.total_sim_value = 0
        self.select_value_tuple = self.get_select_value_tuple(node_status,node_meta)

    def update_value(self,new_value,node_status,node_meta):
        self.total_sim_value += new_value
        self.sim_count += 1
        self.sim_value = self.total_sim_value / self.sim_count
        self.select_value_tuple = self.get_select_value_tuple(node_status,node_meta)

    def get_select_value_tuple(self,node_status,node_meta):
        if node_status.is_leaf:
            return (node_status.node_value * node_meta.father_direc,0)
        else:
            return (self.sim_value * node_meta.father_direc,node_meta.pc / (self.sim_count + 1))

    def get_select_value(self,father_sim_count):
        return self.select_value_tuple[0] + self.select_value_tuple[1] * math.sqrt(father_sim_count)

class ParallelMCTSAgent(MCTSAgent):
    def __init__(self,model,sim_env,N_search=1000,max_depth=10,c_puct=20,temp=1,explore_alpha=0.25):
        super().__init__(model,sim_env,N_search,max_depth,c_puct,temp,explore_alpha)
        
        self.tree_size = 0
        self.root_node_idx = -1
        self.manager = Manager()
        self.tree_meta = self.manager.dict({})
        self.tree_status = self.manager.dict({})
        self.tree_stats = self.manager.dict({})
        self.expand_lock = Lock()

    def init_node(self,node_idx,level,father_idx,father_state,father_action,father_direc,edge_prob,node_value,
                 tree_meta,tree_status,tree_stats):
        tree_meta[node_idx] = NodeMeta(level,father_state,father_action,father_direc,edge_prob,
                                       c_puct=self.c_puct)
        tree_status[node_idx] = NodeStatus(father_idx,node_value) 
        tree_stats[node_idx] = NodeStats(node_value,tree_status[node_idx],tree_meta[node_idx])

    def expand_node(self,node_idx,state,end_flag,probs,q_values,
                    tree_size,tree_meta,tree_status,tree_stats):
        node_meta = tree_meta[node_idx]
        node_status = tree_status[node_idx]
        if end_flag:
            node_status.is_leaf = True
            node_status.node_value = state.eval_score
        else:
            if node_status.is_root:
                ## add noise
                probs = (1 - self.explore_alpha) * probs + \
                        self.explore_alpha * np.random.dirichlet(np.full((len(probs),),0.3))
            probs *= len(probs)
            probs = np.ones((len(probs),))

            for father_action,node_value,edge_prob in (zip(state.action_set,q_values,probs)):
                new_node_idx = tree_size
                self.init_node(new_node_idx,node_meta.level+1,node_idx,state,father_action,state.curr_direc,
                               edge_prob,node_value,
                              tree_meta,tree_status,tree_stats)
                tree_size += 1
                action_key = transform_action_key(father_action)
                node_status.children_idx[action_key] = new_node_idx
        node_status.is_expanded = True
        tree_status[node_idx] = node_status
        return tree_size
        
    def init_mcts_tree(self,state):
        self.tree_meta = self.manager.dict({})
        self.tree_status = self.manager.dict({})
        self.tree_stats = self.manager.dict({})
        self.init_node(node_idx=0,level=0,father_idx=None,father_state=None,father_action=None,
                       father_direc=0,edge_prob=1,node_value=state.eval_score,
                      tree_meta=self.tree_meta,tree_status=self.tree_status,tree_stats=self.tree_stats)
        self.tree_size = 1
        self.root_node_idx = 0
        self.tree_status[self.root_node_idx].set_root()
        q_values,probs = self.get_predict_values(state)
        self.tree_size = self.expand_node(self.root_node_idx,state=state,end_flag=False,
                                        probs=probs,q_values=q_values,
                                        tree_size=self.tree_size,tree_meta=self.tree_meta,
                                        tree_status=self.tree_status,tree_stats=self.tree_stats)

    def update_mcts_tree(self,prev_action,state,end_flag):
        if not end_flag:
            if self.root_node_idx < 0 or prev_action is None:
                self.init_mcts_tree(state)
            else:
                action_key = transform_action_key(prev_action)
                if action_key in self.tree_status[self.root_node_idx].children_idx:
                    self.root_node_idx = self.tree_status[self.root_node_idx].children_idx[action_key]
                    node_status = self.tree_status[self.root_node_idx]
                    node_status.set_root()
                    if not node_status.is_expanded:
                        q_values,probs = self.get_predict_values(state)
                        self.tree_size = self.expand_node(self.root_node_idx,state=state,end_flag=False,
                                                         probs=probs,q_values=q_values,
                                                        tree_size=self.tree_size,tree_meta=self.tree_meta,
                                                        tree_status=self.tree_status,tree_stats=self.tree_stats)
                else:
                    self.init_mcts_tree(state)

    def select_action_by_value(self,node_idx,tree_stats,tree_status):
        sim_count = tree_stats[node_idx].sim_count
        children_idx = tree_status[node_idx].children_idx
        max_action_key,max_node_idx,max_value = (-1,),-1,-1e8
        for action_key,child_idx in children_idx.items():
            tmp_value = tree_stats[child_idx].get_select_value(sim_count)
            if tmp_value > max_value:
                max_action_key,max_node_idx,max_value = action_key,child_idx,tmp_value
        return max_node_idx

    def select_action_by_count(self,node_idx):
        node_status = self.tree_status[node_idx]
        assert len(node_status.children_idx) > 0
        root_state = self.tree_meta[list(node_status.children_idx.values())[0]].father_state
        values = []
        for action in root_state.action_set:
            action_key = transform_action_key(action)
            child_idx = node_status.children_idx[action_key]
            values.append(self.tree_stats[child_idx].sim_count ** self.temp)
        values = np.array(values)
        probs = values/np.sum(values)
        a_idx = np.random.choice(len(values),p=probs)
        action = root_state.action_set[a_idx]
        return a_idx,action,probs,root_state

    def act(self,state):
        action_set = state.action_set
        Na = len(action_set)
        if Na == 1:
            action = action_set[0]
            if self.sample_collect_flag:
                ## value samples: Q(state,action) -> final reward - current reward
                ## only init parts here; append the final reward at the end of the game
                self.value_samples_partial_cache.append((state.state_vec.copy(),state.actions_vec[0],state.eval_score))
        else:
            if self.root_node_idx < 0:
                self.init_mcts_tree(state)
                
            max_search_level = self.tree_meta[self.root_node_idx].level + self.max_depth
            N_iter = max(self.N_search,Na)
            
            tt = time.time()
            
            N_process = 4
            tree_size_proxy = Value('i', self.tree_size)
            
            ps = []
            for process_idx in range(N_process):
                p = Process(target=self.mcts_search,args=(N_iter//N_process,max_search_level,
                                      tree_size_proxy,self.tree_meta,self.tree_status,
                                      self.tree_stats,self.expand_lock))
                p.start()
                ps.append(p)
                
            print(1,time.time()-tt)
                
            for p in ps:
                p.join()
                
            print(2,time.time()-tt)
                
            self.tree_size = tree_size_proxy.value
                
            ## need to modify later!!!
            if self.debug_flag:
                ruler = state.ruler
                values = [child.get_value() for child in self.root_node.children.values()]
                sort_idxs = np.argsort(values)[::-1]
                debug_str = ';'.join(
                    [' {},{:.2f}'.format(ruler.get_codes_repr(action_set[i][0]),values[i]) 
                    for i in sort_idxs])
                return debug_str

            a_idx,action,a_probs,root_state = self.select_action_by_count(self.root_node_idx)
                
            if self.sample_collect_flag:
                direc,state_vec,actions_vec = root_state.get_vecs()
                ## policy samples: P(state,actions,direc) -> action_probs
                self.policy_samples_cache.append((state_vec,actions_vec,direc,a_probs))
                ## value samples: Q(state,action) -> final reward - current reward
                ## only init parts here; append the final reward at the end of the game
                self.value_samples_partial_cache.append((state_vec,actions_vec[a_idx],state.eval_score))
            
        return action
    
    def get_predict_values(self,state):
        if len(state.actions_vec) > 0:
            # print(state.state_vec.shape,state.actions_vec.shape,state.curr_direc)
            # try:
            #     q_values,probs,_ = self.action_model.predict(state.state_vec,state.actions_vec,state.curr_direc)
            #     print(q_values.shape,probs.shape)
            # except Exception as e:
            #     print(e)
            q_values,probs = np.zeros((len(state.actions_vec),)),np.ones((len(state.actions_vec),))
            return q_values,probs
        else:
            return None,None
    
    def mcts_search(self,N_iter,max_search_level,tree_size,tree_meta,tree_status,tree_stats,expand_lock):
        for iter_ in range(N_iter):
            ## search
            curr_node_idx = self.root_node_idx
            curr_node_meta,curr_node_status = tree_meta[curr_node_idx],tree_status[curr_node_idx]
            while curr_node_status.is_expanded and not curr_node_status.is_leaf and curr_node_meta.level < max_search_level:
                curr_node_idx = self.select_action_by_value(curr_node_idx,tree_stats,tree_status)
                curr_node_meta,curr_node_status = tree_meta[curr_node_idx],tree_status[curr_node_idx]

            ## tree expansion
            if not curr_node_status.is_expanded:
                _,end_flag,state_tmp = self.sim_env.step(curr_node_meta.father_state.copy(),
                                                         curr_node_meta.father_action,if_display=False)
                try:
                    if not end_flag:
                        q_values,probs = self.get_predict_values(state_tmp)
                    else:
                        q_values,probs = None,None
                except Exception as e:
                    print(e)
                expand_lock.acquire()
                tree_size.value = self.expand_node(curr_node_idx,state=state_tmp,end_flag=end_flag,
                                            probs=probs,q_values=q_values,
                                            tree_size=tree_size.value,tree_meta=tree_meta,
                                            tree_status=tree_status,tree_stats=tree_stats)
                expand_lock.release()
            
            ## update value
            new_value = curr_node_status.node_value
            curr_node_stats = tree_stats[curr_node_idx]
            curr_node_stats.update_value(new_value,curr_node_status,tree_meta[curr_node_idx])
            tree_stats[curr_node_idx] = curr_node_stats
            while (curr_node_status.father_idx is not None and not curr_node_status.is_root):
                curr_node_idx = curr_node_status.father_idx
                curr_node_status = tree_status[curr_node_idx]
                curr_node_stats = tree_stats[curr_node_idx]
                curr_node_stats.update_value(new_value,curr_node_status,tree_meta[curr_node_idx])
                tree_stats[curr_node_idx] = curr_node_stats