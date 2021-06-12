import numpy as np
import random

class State:
    def __init__(self,meta_info):
        number,house,suit,ruler,stack,stack_score = meta_info
        self.number = number
        self.house = house
        self.suit = suit
        self.ruler = ruler
        self.stack = stack
        if stack_score is None:
            self.stack_score = np.sum(ruler.codes_score * ruler.get_codes_encodings(stack))
        else:
            self.stack_score = stack_score

        self.suit_size = self.ruler.suit_size
        self.deck_size = self.ruler.deck_size

    def get_meta_info(self):
        return (self.number,self.house,self.suit,self.ruler,self.stack,self.stack_score)

    def copy(self):
        state_copy = State(self.get_meta_info())
        state_copy.set_round_info(self.get_round_info())
        state_copy.set_game_info(self.get_game_info())
        state_copy.set_action_set(self.get_action_set())
        return state_copy

    def init_game_info(self,game_info):
        codes_by_player,structs_by_player = game_info

        self.round_ = 0
        self.game_score = 0
        self.eval_score = 0
        self.past_best_player = []
        self.past_round_score = []

        # self.codes_by_player = codes_by_player
        self.structs_by_player = structs_by_player

        self.init_round_info(self.house)

    def update_game_info(self,round_score,round_eval_score):
        self.round_ += 1
        self.game_score += round_score
        self.eval_score += round_eval_score
        self.past_best_player.append(self.best_player)
        self.past_round_score.append(round_score)

        self.init_round_info(self.best_player)

    def set_game_info(self,game_info):
        round_,game_score,eval_score,past_best_player,past_round_score,structs_by_player = game_info
        self.round_ = round_
        self.game_score = game_score
        self.eval_score = eval_score

        self.past_best_player = past_best_player[:]
        self.past_round_score = past_round_score[:]

        self.structs_by_player = [[(0,[]) for _ in range(4)] for _ in range(4)]
        for idx_player in range(4):
            for idx_suit in range(4):
                size,struct = structs_by_player[idx_player][idx_suit]
                self.structs_by_player[idx_player][idx_suit] = (size,struct[:])

        # ## only deep copy the curr player
        # self.structs_by_player = structs_by_player[:]
        # self.structs_by_player[self.curr_player] = [(item[0],item[1][:]) for item in structs_by_player[self.curr_player]]        

    def get_game_info(self):
        return (self.round_,self.game_score,self.eval_score,self.past_best_player,self.past_round_score,self.structs_by_player)

    def init_round_info(self,curr_player):
        self.curr_player = curr_player
        self.curr_direc = 1 if self.curr_player % 2 == 0 else -1
        self.is_first = True
        self.round_num = 0
        self.round_suit_enc = self.best_suit_enc = -1
        self.round_struct = self.best_struct = []
        self.best_player = self.curr_player
        self.best_codes = []
        self.round_plays = []

    def update_round_info(self,round_info):
        round_num,round_suit_enc,round_struct,best_suit_enc,best_struct,best_player,best_codes,round_plays = round_info
        self.curr_player = (self.curr_player + 1) % 4
        self.curr_direc = 1 if self.curr_player % 2 == 0 else -1
        self.is_first = False
        self.round_num = round_num
        self.round_suit_enc = round_suit_enc
        self.round_struct = round_struct
        self.best_suit_enc = best_suit_enc
        self.best_struct = best_struct
        self.best_player = best_player
        self.best_codes = best_codes
        self.round_plays = round_plays

    def set_round_info(self,round_info):
        curr_player,curr_direc,is_first,round_num,round_suit_enc,round_struct,best_suit_enc,best_struct,best_player,best_codes,round_plays = round_info
        self.curr_player = curr_player
        self.curr_direc = curr_direc
        self.is_first = is_first
        self.round_num = round_num
        self.round_suit_enc = round_suit_enc
        self.best_suit_enc = best_suit_enc
        self.best_player = best_player

        self.round_struct = round_struct[:]
        self.best_struct = best_struct[:]
        self.best_codes = best_codes[:]
        self.round_plays = round_plays[:]

    def get_round_info(self):
        return (self.curr_player,self.curr_direc,self.is_first,self.round_num,self.round_suit_enc,self.round_struct,self.best_suit_enc,self.best_struct,self.best_player,self.best_codes,self.round_plays)

    def init_action_set(self):
        self.action_set = []

    def update_action_set(self,action_set):
        self.action_set = action_set

    def set_action_set(self,action_set):
        self.action_set = action_set[:]

    def get_action_set(self):
        return self.action_set

    def get_info_set(self):
        return self.copy()

class StateExtend(State):
    def __init__(self,meta_info):
        super().__init__(meta_info)
        self.state_int_idx = 3 * self.deck_size
        self.prev_state_vecs_map = dict([(player,list()) for player in range(4)])
        self.prev_full_state_vecs_map = list()
        self.default_state_vec = self.get_default_state_vec()

    def copy(self):
        state_copy = StateExtend(self.get_meta_info())
        state_copy.set_round_info(self.get_round_info())
        state_copy.set_game_info(self.get_game_info())
        state_copy.set_action_set(self.get_action_set())

        state_copy.state_vec = self.state_vec.copy()
        state_copy.actions_vec = self.actions_vec
        state_copy.score_encoding = self.score_encoding

        state_copy.prev_state_vecs_map = dict([(player,[item.copy() for item in self.prev_state_vecs_map[player]]) for player in range(4)])
        state_copy.prev_full_state_vecs_map = [item.copy() for item in self.prev_full_state_vecs_map]
        state_copy.default_state_vec = self.default_state_vec.copy()
        
        return state_copy

    def get_default_state_vec(self):
        default_state_vec = np.zeros((1,3*self.deck_size+4*3+(1+4*3)+3+1,),dtype=int)
        default_state_vec[0,:(2*self.deck_size)] = -2
        default_state_vec[0,(2*self.deck_size):(3*self.deck_size)] = self.ruler.codes_score
        return default_state_vec
    
    def init_game_info(self,game_info):
        super().init_game_info(game_info)
        self.score_encoding = self.ruler.codes_score
        
        codes_by_player,_ = game_info
        # state = [card_ownership&status,card_score,
        #         player_status,history_score,curr_score,future_score,
        #         turn_status*3,round_]
        self.state_vec = self.get_default_state_vec()
        for code in self.stack:
            if self.state_vec[0,code] != -2:
                self.state_vec[0,code+self.deck_size] = -1
            else:
                self.state_vec[0,code] = -1
        self.state_vec[0,(self.state_int_idx+12)] = self.stack_score
        ## update status of hands
        for player in range(4):
            player_default_status = player + 4 * 2
            score_tmp = 0
            for code in codes_by_player[player]:
                if self.state_vec[0,code] != -2:
                    self.state_vec[0,code+self.deck_size] = player_default_status
                else:
                    self.state_vec[0,code] = player_default_status
                score_tmp += self.score_encoding[code]
            self.state_vec[0,self.state_int_idx+12+1+8+player] = score_tmp

        self.actions_vec = np.zeros((0,self.deck_size+1),dtype=int)

        for player in [self.curr_player]:
            self.prev_state_vecs_map[player].append(self.get_infoset_vec(player=player))
        self.prev_full_state_vecs_map.append(self.state_vec.copy())
        
    def update_game_info(self,round_score,round_eval_score):
        best_player = self.best_player
        int_idx = self.state_int_idx
        
        super().update_game_info(round_score,round_eval_score)

        for player in range(4):
            bools = (self.state_vec[0,:(2*self.deck_size)] == player + 4 * 1)
            self.state_vec[0,:(2*self.deck_size)][bools] = player
        
        self.state_vec[0,(int_idx+12+1):(int_idx+12+5)] += self.state_vec[0,(int_idx+12+5):(int_idx+12+9)]
        self.state_vec[0,(int_idx+12+5):(int_idx+12+9)] = 0
        
        self.state_vec[0,(int_idx):(int_idx+4)] = 1
        self.state_vec[0,(int_idx+4):(int_idx+12)] = 0
        self.state_vec[0,(int_idx+best_player)] = 0
        self.state_vec[0,(int_idx+4+best_player)] = 1
        direc = -1
        if best_player % 2 == 0:
            direc = 1
        self.state_vec[0,-4] = direc ## round start
        self.state_vec[0,-3] = direc ## round best
        self.state_vec[0,-2] = direc ## round curr
        self.state_vec[0,-1] = self.round_

        for player in [self.curr_player]:
            self.prev_state_vecs_map[player].append(self.get_infoset_vec(player=player))
        self.prev_full_state_vecs_map.append(self.state_vec.copy())
        
    def update_round_info(self,round_info):
        super().update_round_info(round_info)
        round_plays = self.round_plays
        if len(round_plays) > 0:
            int_idx = self.state_int_idx
            last_player,last_codes = round_plays[-1]
            if self.best_player % 2 == 0:
                self.state_vec[0,-3] = 1
            else:
                self.state_vec[0,-3] = -1
            new_player = self.curr_player
            if new_player % 2 == 0:
                self.state_vec[0,-2] = 1
            else:
                self.state_vec[0,-2] = -1
            self.state_vec[0,(int_idx+last_player)] = 0
            self.state_vec[0,(int_idx+4+last_player)] = 0
            self.state_vec[0,(int_idx+8+last_player)] = 1
            self.state_vec[0,(int_idx+new_player)] = 0
            self.state_vec[0,(int_idx+4+new_player)] = 1
            self.state_vec[0,(int_idx+8+new_player)] = 0
            score = 0
            last_prev_status = last_player + 4 * 2
            last_status = last_player + 4 * 1
            for code in last_codes:
                score += self.score_encoding[code]
                if self.state_vec[0,code] == -2 or self.state_vec[0,code] == last_prev_status:
                    self.state_vec[0,code] = last_status
                else:
                    self.state_vec[0,code+self.deck_size] = last_status
            self.state_vec[0,int_idx+12+5+last_player] += score
            self.state_vec[0,int_idx+12+5+4+last_player] -= score

            for player in [self.curr_player]:
                self.prev_state_vecs_map[player].append(self.get_infoset_vec(player=player))
            self.prev_full_state_vecs_map.append(self.state_vec.copy())

    def update_action_set(self,action_set):
        super().update_action_set(action_set)
        if action_set is not None and len(action_set) > 0:
            actions_vec = np.zeros((1,len(action_set),self.deck_size+1),dtype=int)
            for i,a in enumerate(action_set):
                score = 0
                for code in a[0]:
                    actions_vec[0,i,code] += 1
                    score += self.score_encoding[code]
                actions_vec[0,i,-1] = score
        else:
            actions_vec = np.zeros((1,0,self.deck_size+1),dtype=int)
        self.actions_vec = actions_vec.astype(np.float32)

    def get_vecs(self):
        history_size = 4
        state_vecs = self.prev_state_vecs_map[self.curr_player][-history_size:]
        if len(state_vecs) < history_size:
            state_vecs = [state_vecs[0] for _ in range(history_size - len(state_vecs))] + state_vecs
        state_vecs = np.stack(state_vecs,axis=1).astype(np.float32)
        state_vecs[0,:,(2*self.deck_size):(3*self.deck_size)] /= 100.0
        state_vecs[0,:,(self.state_int_idx+12):(self.state_int_idx+12+13)] /= 100.0
        return state_vecs,self.actions_vec,self.curr_direc

    def get_full_vecs(self):
        history_size = 4
        state_vecs = self.prev_full_state_vecs_map[-history_size:]
        if len(state_vecs) < history_size:
            state_vecs = [state_vecs[0] for _ in range(history_size - len(state_vecs))] + state_vecs
        state_vecs = np.stack(state_vecs,axis=1).astype(np.float32)
        state_vecs[0,:,(2*self.deck_size):(3*self.deck_size)] /= 100.0
        state_vecs[0,:,(self.state_int_idx+12):(self.state_int_idx+12+13)] /= 100.0
        return state_vecs,self.actions_vec,self.curr_direc

    def get_infoset_vec(self,player=None):
        if player is None:
            player = self.curr_player

        state_vec = self.state_vec.copy()

        ## mask out score
        int_idx = self.state_int_idx
        state_vec[0,(int_idx+12+9):(int_idx+12+13)] = 0
        ## mask out state vec
        card_status = state_vec[0,:(2*self.deck_size)]
        mask_bools = (card_status >= 4 * 2) & (card_status != 4 * 2 + player)
        if self.house != player:
            ## hide stack info
            mask_bools = mask_bools | (card_status == -1)
        state_vec[0,:(2*self.deck_size)][mask_bools] = -2
        return state_vec

    def get_info_set(self):
        info_set = self.copy()
        info_set.state_vec = self.get_infoset_vec()
        return info_set

class StateExtendV2(StateExtend):
    def __init__(self,meta_info):
        super().__init__(meta_info)
        self.prob_info = {'all':1,'i':np.ones((4,)),'-i':np.ones((4,))}

    def copy(self):
        state_copy = StateExtendV2(self.get_meta_info())
        state_copy.set_round_info(self.get_round_info())
        state_copy.set_game_info(self.get_game_info())
        state_copy.set_action_set(self.get_action_set())

        state_copy.state_vec = self.state_vec.copy()
        state_copy.actions_vec = self.actions_vec
        state_copy.score_encoding = self.score_encoding

        state_copy.prev_state_vecs_map = dict([(player,[item.copy() for item in self.prev_state_vecs_map[player]]) for player in range(4)])
        state_copy.prev_full_state_vecs_map = [item.copy() for item in self.prev_full_state_vecs_map]
        state_copy.default_state_vec = self.default_state_vec.copy()

        state_copy.prob_info = {'all':self.prob_info['all'],
                                'i':self.prob_info['i'].copy(),
                                '-i':self.prob_info['-i'].copy()}

        return state_copy

def rotate_state(state_vec,rotation=0):
    int_idx = state_vec.shape[-1] - (4*3 + (1+4*3) + 3 + 1)
    deck_size = int_idx // 3

    new_state_vec = state_vec.copy()
    if rotation % 2 != 0:
        new_state_vec[0,:,-4:-1] = -new_state_vec[0,:,-4:-1]
    old_card_status = state_vec[0,:,:(2*deck_size)]
    idxs_seen = old_card_status >= 0
    new_state_vec[0,:,:(2*deck_size)][idxs_seen] = (old_card_status - old_card_status % 4 + (old_card_status + rotation) % 4)[idxs_seen]

    for idx in range(4):
        idx_new = (idx + rotation) % 4
        ## status
        new_state_vec[0,:,(int_idx+idx_new)] = state_vec[0,:,(int_idx+idx)]
        new_state_vec[0,:,(int_idx+4+idx_new)] = state_vec[0,:,(int_idx+4+idx)]
        new_state_vec[0,:,(int_idx+8+idx_new)] = state_vec[0,:,(int_idx+8+idx)]
        ## scores
        new_state_vec[0,:,(int_idx+12+1+idx_new)] = state_vec[0,:,(int_idx+12+1+idx)]
        new_state_vec[0,:,(int_idx+12+1+4+idx_new)] = state_vec[0,:,(int_idx+12+1+4+idx)]
        new_state_vec[0,:,(int_idx+12+1+8+idx_new)] = state_vec[0,:,(int_idx+12+1+8+idx)]
    return new_state_vec
