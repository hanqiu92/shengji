import numpy as np
import random

def rotate_state(state_vec,rotation=0):
    new_state_vec = state_vec.copy()
    if rotation % 2 != 0:
        new_state_vec[0,-4:-1] = -new_state_vec[0,-4:-1]
    for idx in range(4):
        idx_new = (idx + rotation) % 4
        ## cards
        new_state_vec[0,((2+idx_new)*54):((2+idx_new+1)*54)] = state_vec[0,((2+idx)*54):((2+idx+1)*54)]
        new_state_vec[0,((6+idx_new)*54):((6+idx_new+1)*54)] = state_vec[0,((6+idx)*54):((6+idx+1)*54)]
        new_state_vec[0,((10+idx_new)*54):((10+idx_new+1)*54)] = state_vec[0,((10+idx)*54):((10+idx+1)*54)]
        ## status
        new_state_vec[0,(14*54+idx_new)] = state_vec[0,(14*54+idx)]
        new_state_vec[0,(14*54+4+idx_new)] = state_vec[0,(14*54+4+idx)]
        new_state_vec[0,(14*54+8+idx_new)] = state_vec[0,(14*54+8+idx)]
        ## scores
        new_state_vec[0,(14*54+12+1+idx_new)] = state_vec[0,(14*54+12+1+idx)]
        new_state_vec[0,(14*54+12+1+4+idx_new)] = state_vec[0,(14*54+12+1+4+idx)]
        new_state_vec[0,(14*54+12+1+8+idx_new)] = state_vec[0,(14*54+12+1+8+idx)]
    return new_state_vec

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

    def get_meta_info(self):
        return (self.number,self.house,self.suit,self.ruler,self.stack,self.stack_score)

    def copy(self):
        state_copy = State(self.get_meta_info())
        state_copy.set_round_info(self.get_round_info())
        state_copy.set_game_info(self.get_game_info())
        state_copy.set_action_set(self.get_action_set())
        return state_copy

    def init_game_info(self,structs_by_player):
        # codes_by_player,structs_by_player = game_info

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

        self.structs_by_player = structs_by_player[:]
        self.structs_by_player[self.curr_player] = [(item[0],item[1][:]) for item in structs_by_player[self.curr_player]]        

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

class StateExtend(State):
    def copy(self):
        state_copy = StateExtend(self.get_meta_info())
        state_copy.set_round_info(self.get_round_info())
        state_copy.set_game_info(self.get_game_info())
        state_copy.set_action_set(self.get_action_set())

        state_copy.state_vec = self.state_vec.copy()
        state_copy.actions_vec = self.actions_vec
        state_copy.score_encoding = self.score_encoding
        
        return state_copy
    
    def init_game_info(self,game_info):
        super().init_game_info(game_info)
        self.score_encoding = self.ruler.codes_score
        
        # state = [s_score,s_stack,
        #         s_history,s_curr,s_future,
        #         player_status,
        #         history_score,curr_score,future_score,
        #         turn_status*3,round_]
        self.state_vec = np.zeros((1,54 + 54*(1+4+4+4) + 4*3 + (1+4+4+4) + 2 + 1 + 1,),dtype=int)
        self.state_vec[0,0:54] = self.score_encoding
        self.state_vec[0,(1*54):(2*54)] = self.ruler.get_codes_encodings(self.stack) * 100
        self.state_vec[0,(14*54+12)] = self.stack_score

        self.actions_vec = np.zeros((0,55),dtype=int)
        
    def update_game_info(self,round_score,round_eval_score):
        best_player = self.best_player
        
        super().update_game_info(round_score,round_eval_score)
        
        self.state_vec[0,(2*54):(6*54)] += self.state_vec[0,(6*54):(10*54)]
        self.state_vec[0,(14*54+12+1):(14*54+12+5)] += self.state_vec[0,(14*54+12+5):(14*54+12+9)]
        self.state_vec[0,(6*54):(10*54)] = 0
        self.state_vec[0,(14*54+12+5):(14*54+12+9)] = 0
        
        self.state_vec[0,(14*54):(14*54+4)] = 100
        self.state_vec[0,(14*54+4):(14*54+12)] = 0
        self.state_vec[0,(14*54+best_player)] = 0
        self.state_vec[0,(14*54+4+best_player)] = 100
        direc = -100
        if best_player % 2 == 0:
            direc = 100
        self.state_vec[0,-4] = direc ## round start
        self.state_vec[0,-3] = direc ## round best
        self.state_vec[0,-2] = direc ## round curr
        self.state_vec[0,-1] = self.round_
        
    def update_round_info(self,round_info):
        super().update_round_info(round_info)
        round_plays = self.round_plays
        if len(round_plays) > 0:
            last_player,last_codes = round_plays[-1]
            if self.best_player % 2 == 0:
                self.state_vec[0,-3] = 100
            else:
                self.state_vec[0,-3] = -100
            new_player = self.curr_player
            if new_player % 2 == 0:
                self.state_vec[0,-2] = 100
            else:
                self.state_vec[0,-2] = -100
            self.state_vec[0,(14*54+last_player)] = 0
            self.state_vec[0,(14*54+4+last_player)] = 0
            self.state_vec[0,(14*54+8+last_player)] = 100
            self.state_vec[0,(14*54+new_player)] = 0
            self.state_vec[0,(14*54+4+new_player)] = 100
            self.state_vec[0,(14*54+8+new_player)] = 0
            score = 0
            for code in last_codes:
                score += self.score_encoding[code]
                self.state_vec[0,(6+last_player)*54+code] += 100
                self.state_vec[0,(10+last_player)*54+code] -= 100
            self.state_vec[0,14*54+12+5+last_player] += score
            self.state_vec[0,14*54+12+5+4+last_player] -= score

    def update_action_set(self,action_set):
        super().update_action_set(action_set)
        if action_set is not None and len(action_set) > 0:
            actions_vec = np.zeros((1,len(action_set),55),dtype=int)
            for i,a in enumerate(action_set):
                score = 0
                for code in a[0]:
                    actions_vec[0,i,code] += 1
                    score += self.score_encoding[code]
                actions_vec[0,i,-1] = score
        else:
            actions_vec = np.zeros((1,0,55),dtype=int)
        self.actions_vec = actions_vec.astype(np.float32)

    def get_vecs(self):
        return self.state_vec.astype(np.float32)/100,self.actions_vec,self.curr_direc