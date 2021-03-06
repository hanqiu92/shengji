import numpy as np
import random
import pulp

class GlobalInfo:
    def __init__(self,game_info):
        number,house,suit,ruler,stack = game_info
        self.number = number
        self.house = house
        self.suit = suit
        self.ruler = ruler
        self.stack_encodings = ruler.get_codes_encodings(stack)
        self.stack_score = np.sum(ruler.codes_score * self.stack_encodings)

        self.past_best_player = []
        self.past_round_score = []

    def update_round_history(self,round_plays,best_player,round_score):
        self.past_best_player.append(best_player)
        self.past_round_score.append(round_score)

    def update_round_info(self,round_info):
        round_num,round_suit_enc,round_struct,best_suit_enc,best_struct,best_player,round_plays = round_info
        self.round_num = round_num
        self.round_suit_enc = round_suit_enc
        self.round_struct = round_struct
        self.best_suit_enc = best_suit_enc
        self.best_struct = best_struct
        self.best_player = best_player
        self.curr_round_plays = round_plays

class Env:
    def __init__(self,global_info_generator=GlobalInfo):
        self.decks = [(suit,number) for suit in range(1,5) for number in range(1,14)] + \
                     [(0,number) for number in range(2)]
        self.new_decks = self.decks.copy() + self.decks.copy()
        self.global_info_generator = global_info_generator
        self.reset()

    def reset(self,house=-1,number=2,level=(2,2),score=(0,0)):
        self.curr_house = house
        self.curr_number = number
        self.curr_level = level
        self.curr_score = score
        self.curr_suit = 1
        self.ruler = Ruler(self.curr_number,self.curr_suit)

    def whole_round(self,players,if_display=True):
        self.assign_round(players,if_display)
        self.play_round(players,if_display)

    def assign_round(self,players,if_display=True):
        '''
        call + stack
        call format: (round,player,suit,multiples)
        '''
        for player in players:
            player.reset_cards()

        if self.curr_house >= 0:
            player_start = self.curr_house
        else:
            player_start = np.random.choice(4)

        this_deck = self.new_decks.copy()
        random.shuffle(this_deck)

        self.call_history = []
        curr_call = None
        for round_ in range(25):
            for curr_player in ((np.arange(4)+player_start) % 4):
                curr_card = this_deck.pop()
                players[curr_player].cards.append(curr_card)
                new_call = players[curr_player].call(self.curr_number,self.curr_house,self.call_history,self.ruler)
                if new_call is not None:
                    curr_suit,curr_multiples = new_call
                    new_call = (round_,curr_player,curr_suit,curr_multiples)
                    if (curr_call is None):
                        curr_call = new_call
                        self.call_history.append(new_call)
                    elif (new_call[-1] > curr_call[-1] and new_call[1] != curr_call[1]):
                        curr_call = new_call
                        self.call_history.append(new_call)
        
        if self.curr_house < 0:
            if curr_call is None:
                self.curr_house = player_start
            else:
                self.curr_house = curr_call[1]
        if curr_call is None:
            self.curr_suit = np.random.choice(np.arange(1,5))
        else:
            self.curr_suit = curr_call[2]

        self.ruler.update(self.curr_number,self.curr_suit)
        players[self.curr_house].cards.extend(this_deck)
        for player in players:
            player.cards2codes(self.ruler)

        self.stack = players[self.curr_house].stack(self.call_history,self.ruler)
        
        if if_display:
            for curr_player in range(4):
                print('init:player:{} -> [{}],{}.'.format(curr_player,self.ruler.get_codes_repr(players[curr_player].codes),len(players[curr_player].codes)))

    def play_round(self,players,if_display=True):
        player_start = self.curr_house
        game_score = 0
        global_info = self.global_info_generator(game_info=[self.curr_number,self.curr_house,self.curr_suit,self.ruler,self.stack])

        play_history = []
        for round_ in range(25):
            if len(players[player_start].codes) <= 0:
                break

            round_plays = []
            round_num = 0
            round_suit_enc = best_suit_enc = -1
            round_struct = best_struct = None
            best_codes = []
            best_player = player_start

            global_info.update_round_info(round_info=[round_num,round_suit_enc,round_struct,best_suit_enc,best_struct,best_player,round_plays])

            for curr_player in ((np.arange(4)+player_start) % 4):
                is_first = curr_player == player_start
                action_selected = players[curr_player].play(global_info,is_first,if_display)
                curr_codes,curr_struct,_ = action_selected
                curr_num = len(curr_codes)

                if curr_struct is None:
                    curr_suit_enc,curr_struct_dict = self.ruler.get_struct(curr_codes)
                    if curr_suit_enc >= 0:
                        curr_struct = curr_struct_dict[curr_suit_enc][1]
                    else:
                        curr_struct = []
                else:
                    curr_suit_enc = -1
                    suit_enc_cnt = 0
                    suit_enc_bools = np.full((4,),False)
                    for component in curr_struct:
                        suit_enc = self.ruler.get_code_suit_enc(component[1])
                        if not suit_enc_bools[suit_enc]:
                            suit_enc_cnt += 1
                            curr_suit_enc = suit_enc
                            suit_enc_bools[suit_enc] = True
                    if suit_enc_cnt > 1:
                        curr_suit_enc = -1

                if is_first:
                    assert curr_suit_enc >= 0
                    round_num,round_suit_enc,round_struct = curr_num,curr_suit_enc,curr_struct
                    bool_best = True
                else:
                    bool_best,curr_struct = self.ruler.compare_struct(round_num,round_suit_enc,round_struct,best_suit_enc,best_struct,curr_suit_enc,curr_struct)

                if bool_best:
                    best_suit_enc,best_struct = curr_suit_enc,curr_struct
                    best_player,best_codes = curr_player,curr_codes

                round_plays.append((curr_player,curr_codes))
                global_info.update_round_info(round_info=[round_num,round_suit_enc,round_struct,best_suit_enc,best_struct,best_player,round_plays])

                if if_display:
                    print('round:{},player:{} -> {}, curr best {}. player remain: [{}],{}.'.format(round_,curr_player,self.ruler.get_codes_repr(curr_codes),self.ruler.get_codes_repr(best_codes),self.ruler.get_codes_repr(players[curr_player].codes),len(players[curr_player].codes)))

            player_start = best_player
            round_score = np.sum([self.ruler.get_codes_score(codes) for player,codes in round_plays])
            if (best_player - self.curr_house) % 2 != 0:
                game_score += round_score
            play_history.append([round_plays,best_player,round_score])
            global_info.update_round_history(round_plays,best_player,round_score)

            if if_display:
                print('round:{},current score:{}.'.format(round_,game_score))
                print('-------------------')

        if (best_player - self.curr_house) % 2 != 0:
            score_ratio = np.max([item[0] for item in round_struct]) * 2
        else:
            score_ratio = 1
        stack_score = global_info.stack_score * score_ratio
        if (best_player - self.curr_house) % 2 != 0:
            game_score += stack_score

        for player in players:
            player.update_dataset(global_info,stack_score=stack_score,is_terminated=True)

        if game_score >= 80:
            self.curr_house = (self.curr_house + 1) % 4
            delta_level = (game_score - 80) // 40
        else:
            self.curr_house = (self.curr_house + 2) % 4
            delta_level = (80 - game_score) // 40 + 1
        delta_level = int(delta_level)

        curr_direc = self.curr_house % 2
        self.curr_score,self.curr_level = list(self.curr_score),list(self.curr_level)
        self.curr_score[curr_direc] += delta_level
        self.curr_level[curr_direc] += delta_level
        while self.curr_level[curr_direc] > 13:
            self.curr_level[curr_direc] -= 13
        self.curr_score,self.curr_level = tuple(self.curr_score),tuple(self.curr_level)
        self.curr_number = self.curr_level[curr_direc]

class Ruler:
    def __init__(self,major_level=2,major_suit=1):
        self.suit_enc_size = [12,12,12,18]
        self.suit_enc_cum_size = np.cumsum([0] + self.suit_enc_size)
        self.codes_suit_enc = np.concatenate([np.full((size,),idx,dtype=int) for idx,size in enumerate(self.suit_enc_size)])
        self.suit_signs = '♦♣♥♠'
        self.update(major_level,major_suit)

    def update(self,major_level=2,major_suit=1):
        self.major_level = major_level
        self.major_suit = major_suit
        self.major_level_order = (self.major_level + 11) % 13

        self.codes_score = np.zeros((54,),dtype=int)
        self.codes_value = np.zeros((54,),dtype=int)
        self.codes_repr = []
        for code in range(54):
            card = self.get_card_decoding(code)
            self.codes_score[code] = self.get_card_score(card)
            self.codes_value[code] = self.get_card_value(card)
            self.codes_repr.append(self.get_card_repr(card))

    def get_card_encoding(self,card):
        suit,num = card
        num_order = (num + 11) % 13
        if suit == 0:
            return 52 + num
        else:
            if suit == self.major_suit:
                suit_enc = 3
            else:
                suit_enc = suit - 1 - (suit > self.major_suit)

            if num == self.major_level:
                return 48 + suit_enc
            else:
                return suit_enc * 12 + num_order - (num_order > self.major_level_order)
        return -1

    def get_card_decoding(self,code):
        if code > 51:
            return (0,code - 52)
        else:
            if code >= 48:
                num = self.major_level
                suit_enc = code - 48
            else:
                num_order = code % 12
                num_order += (num_order >= self.major_level_order)
                num = (num_order + 1) % 13 + 1
                suit_enc = code // 12

            if suit_enc == 3:
                suit = self.major_suit
            else:
                suit = suit_enc + 1 + (suit_enc + 1 >= self.major_suit)
            return (suit,num)

    def get_card_score(self,card):
        suit,num = card
        if num == 5:
            score = 5
        elif num == 10 or num == 13:
            score = 10
        else:
            score = 0
        return score

    def get_card_value(self,card):
        suit,num = card
        if suit == 0:
            value = 100
        elif suit == self.major_suit:
            value = 25
        else:
            value = 0
        if num == self.major_level:
            value += 50
        else:
            value += (num + 11) % 13
        return value

    def get_card_repr(self,card):
        suit,num = card
        if suit == 0:
            if num == 0:
                return '☆'
            else:
                return '☆☆'
        else:
            suit_sign = self.suit_signs[suit - 1]
            num_sign = str(num)
            if num == 1:
                num_sign = 'A'
            elif num == 11:
                num_sign = 'J'
            elif num == 12:
                num_sign = 'Q'
            elif num == 13:
                num_sign = 'K'
        return suit_sign + num_sign

    def get_code_score(self,code):
        return self.codes_score[code]

    def get_code_value(self,code):
        return self.codes_value[code]

    def get_code_repr(self,code):
        return self.codes_repr[code]

    def get_code_suit_enc(self,code):
        return self.codes_suit_enc[code]

    def get_codes_score(self,codes):
        score = 0
        for code in codes:
            score += self.get_code_score(code)
        return score

    def get_codes_repr(self,codes,code_order=True):
        if code_order:
            sort_idxs = np.argsort(codes)
            strings = ','.join([self.get_code_repr(codes[idx]) for idx in sort_idxs])
        else:
            strings = ','.join([self.get_code_repr(code) for code in codes])
        return strings

    def get_codes_encodings(self,codes):
        encodings = np.zeros((54,),dtype=int)
        for code in codes:
            encodings[code] += 1
        return encodings

    def get_component_value(self,component):
        size,code = component[0],component[1]
        value = self.get_code_value(code) + (size - 1) * 500
        return value

    def get_minmax_struct_value(self,struct):
        if len(struct) == 0:
            return -1,0,-1,0

        values = [self.get_component_value(component) for component in struct]
        max_idx,min_idx = np.argmax(values),np.argmin(values)
        max_value,min_value = values[max_idx],values[min_idx]
        return min_idx,min_value,max_idx,max_value

    def get_struct(self,codes):
        encodings = self.get_codes_encodings(codes)
        struct_by_suit = dict()
        curr_suit_enc = -1
        suit_enc_cnt = 0
        for suit_enc in range(4):
            struct_by_suit[suit_enc] = self.get_struct_suit(encodings,suit_enc)
            if struct_by_suit[suit_enc][0] > 0:
                suit_enc_cnt += 1
                curr_suit_enc = suit_enc
        if suit_enc_cnt != 1:
            curr_suit_enc = -1
        return curr_suit_enc,struct_by_suit 

    def get_struct_suit(self,encodings,suit_enc):
        code_start_suit = self.suit_enc_cum_size[suit_enc]
        code_end_suit = self.suit_enc_cum_size[suit_enc+1]
        encodings_suit = encodings[code_start_suit:code_end_suit]
        encodings_suit_size = np.sum(encodings_suit)
        if encodings_suit_size > 0:
            struct_suit = []
            ## singles
            bool_singles = encodings_suit == 1
            for code_in_suit in np.where(bool_singles)[0]:
                code = code_in_suit+code_start_suit
                struct_suit.append((1,code,[code]))
            ## pairs
            bool_pairs = encodings_suit == 2
            total_pairs = np.sum(bool_pairs)
            if total_pairs > 0:
                bool_pairs_extend = np.zeros((len(bool_pairs)+2,),dtype=int)
                bool_pairs_extend[1:-1] = bool_pairs
                if suit_enc == 3:
                    bool_pairs_extend = \
                        np.concatenate([bool_pairs_extend[:13],
                                    [(np.sum(bool_pairs_extend[13:16]) > 0).astype(int)],
                                    bool_pairs_extend[16:]])

                idxs_cont_pairs = np.where(np.abs(np.diff(bool_pairs_extend)) > 0)[0]
                idxs_cont_pairs = idxs_cont_pairs.reshape((-1,2))
                size_cont_pairs = idxs_cont_pairs[:,1] - idxs_cont_pairs[:,0]
                code_cont_pairs = idxs_cont_pairs[:,1] - 1

                if suit_enc == 3:
                    for pair_size,code_in_suit in zip(size_cont_pairs,code_cont_pairs):
                        code = code_in_suit + code_start_suit
                        codes_tmp = []
                        for ii in range(pair_size):
                            code_tmp = code_in_suit - ii
                            if code_tmp > 12:
                                code_tmp += 2
                            elif code_tmp == 12:
                                for code_in_suit_tmp in (12,13,14):
                                    if bool_pairs[code_in_suit_tmp]:
                                        code_tmp = code_in_suit_tmp
                                        bool_pairs[code_in_suit_tmp] = False
                                        break
                            code_tmp += code_start_suit
                            codes_tmp.extend([code_tmp,code_tmp])
                        struct_suit.append((2*pair_size,(codes_tmp[0]),(codes_tmp)))

                    for code_in_suit in (12,13,14):
                        if bool_pairs[code_in_suit]:
                            code = code_in_suit + code_start_suit
                            struct_suit.append((2,code,[code,code]))
                else:
                    for pair_size,code_in_suit in zip(size_cont_pairs,code_cont_pairs):
                        code = code_in_suit + code_start_suit
                        codes_tmp = list(range(code-pair_size+1,code+1)) * 2
                        struct_suit.append((2*pair_size,code,sorted(codes_tmp)))

            ## ranking
            values = [self.get_component_value(component) for component in struct_suit]
            sort_idxs = np.argsort(values)[::-1]
            struct_suit = [struct_suit[idx] for idx in sort_idxs]
            return encodings_suit_size,struct_suit

        return 0,[]


    def compare_struct(self,round_num,round_suit_enc,round_struct,best_suit_enc,best_struct,curr_suit_enc,curr_struct):
        if curr_suit_enc < 0:
            return False,None

        if curr_suit_enc != round_suit_enc and curr_suit_enc != 3:
            return False,None

        if round_num == 1:
            if curr_struct[0][1] > best_struct[0][1]:
                return True,curr_struct

        elif round_num == 2:
            if round_struct[0][0] > 1:
                ## pairs
                if len(curr_struct) == 1 and curr_struct[0][0] == 2:
                    if curr_struct[0][1] > best_struct[0][1]:
                        return True,curr_struct
            else:
                ## single flushes
                if max([component[1] for component in curr_struct]) > \
                    max([component[1] for component in best_struct]):
                    return True,curr_struct

        else:
            bool_feas,curr_struct_matched = self.match_struct(round_struct,curr_struct)
            if not bool_feas:
                return False,None

            if self.get_minmax_struct_value(curr_struct_matched)[3] > self.get_minmax_struct_value(best_struct)[3]:
                return True,curr_struct_matched

        return False,None

    def match_struct(self,struct1,struct2,values1=None,values2=None,direc_flag=1):
        ## direc flag: 1 = max, -1 = min
        if len(struct2) == 0:
            return False,None

        if values1 is None or values2 is None:
            values1 = [item[0] * 10000 + sum([code % 12 if code < 36 else code for code in item[-1]]) for item in struct1]
            values2 = [item[0] * 10000 + sum([code % 12 if code < 36 else code for code in item[-1]]) for item in struct2]

        def sort_struct(struct,values):
            sort_idxs = np.argsort(values)[::-1]
            struct = [struct[idx] for idx in sort_idxs]
            values = [values[idx] for idx in sort_idxs]
            return struct,values

        struct1,values1 = sort_struct(struct1,values1)
        struct2,values2 = sort_struct(struct2,values2)
        component_size1 = np.array([item[0] for item in struct1])
        component_size2 = np.array([item[0] for item in struct2])

        component_cum_size1 = np.cumsum(component_size1)
        component_cum_size2 = np.cumsum(component_size2)
        min_length = min(len(component_cum_size1),len(component_cum_size2))
        if np.any(component_cum_size1[:min_length] > component_cum_size2[:min_length]):
            return False,None

        bool_feas,struct_matched = self.match_struct_greedy(struct1,struct2,component_size1,component_size2,values1,values2,direc_flag)
        if bool_feas:
            return bool_feas,struct_matched

        return self.match_struct_lp(struct1,struct2,component_size1,component_size2,values1,values2,direc_flag)

    def decompose_component(self,source_component,target_components,direc_flag=1):
        ## direc flag: 1 = max, -1 = min
        struct_matched,codes_remain = [],np.sort(source_component[-1])
        ## target component从大往小
        ## 正常来说，调用链上应该已经排过序了，这里只是做valid check
        if len(target_components) > 1:
            sort_idxs = np.argsort([component[0] for component in target_components])[::-1]
        else:
            sort_idxs = [0]
        ## 如果是最小化，codes排序从小往大；如果是最大化，codes排序从大往小
        if direc_flag == 1:
            codes_remain = codes_remain[::-1]
        for idx in sort_idxs:
            size = target_components[idx][0]
            struct_matched.append((size,codes_remain[size-1],list(codes_remain[:size])))
            codes_remain = codes_remain[size:]
        codes_remain = list(codes_remain)
        return struct_matched,codes_remain

    def assign_components(self,struct1,struct2,component_size1,component_size2,assign_mat,direc_flag=1):
        struct_matched = []
        # codes_remain = []
        for j in range(len(component_size2)):
            idxs = [i for i in range(len(component_size1)) if assign_mat[i,j] > 1 - 1e-3]
            if len(idxs) > 0:
                struct_matched_tmp,codes_remain_tmp = self.decompose_component(struct2[j],[struct1[i] for i in idxs],direc_flag)
                struct_matched.extend(struct_matched_tmp)
                # codes_remain.extend(codes_remain_tmp)
        return struct_matched

    def match_struct_greedy(self,struct1,struct2,component_size1,component_size2,values1,values2,direc_flag=1):
        x = np.zeros((len(component_size1),len(component_size2)),dtype=int)
        component_size2_ = component_size2.copy()
        for i,s1 in enumerate(component_size1):
            j = np.argmax(component_size2_)
            if s1 <= component_size2_[j]:
                x[i,j] = 1
                component_size2_[j] -= s1
            else:
                return False,None

        struct_matched = self.assign_components(struct1,struct2,component_size1,component_size2,assign_mat=x,direc_flag=direc_flag)
        return True,struct_matched

    def match_struct_lp(self,struct1,struct2,component_size1,component_size2,values1,values2,direc_flag=1):
        model = pulp.LpProblem("cut_stock",pulp.LpMaximize)
        model.solver = pulp.PULP_CBC_CMD(msg=0)
        x_pulp = {}
        for i in range(len(component_size1)):
            for j in range(len(component_size2)):
                x_pulp[(i,j)] = pulp.LpVariable('{}-{}'.format(i,j),lowBound=0,upBound=1,cat='Binary')
        ## selection cons
        for i in range(len(component_size1)):
            model += pulp.LpConstraint(pulp.LpAffineExpression(
                [(x_pulp[(i,j)],1) for j in range(len(component_size2))]),
                sense=0,rhs=1)
        ## match cons
        for j in range(len(component_size2)):
            model += pulp.LpConstraint(pulp.LpAffineExpression(
                [(x_pulp[(i,j)],component_size1[i]) for i in range(len(component_size1))]),
                sense=-1,rhs=component_size2[j])
        model.objective = pulp.LpAffineExpression(
                [(x_pulp[(i,j)],component_size1[i]*values2[j]) for i in range(len(component_size1)) for j in range(len(component_size2))])
        model.solve()

        bool_feas = model.status == 1
        if not bool_feas:
            return False,None

        x = np.zeros((len(component_size1),len(component_size2)),dtype=int)
        for i in range(len(component_size1)):
            for j in range(len(component_size2)):
                x[i,j] = x_pulp[(i,j)].value()

        struct_matched = self.assign_components(struct1,struct2,component_size1,component_size2,assign_mat=x,direc_flag=direc_flag)
        return True,struct_matched

class Player:
    def __init__(self,idx,model):
        self.idx = idx
        self.reset_cards()

        self.action_model = model
        self.dataset = []

    def reset_cards(self):
        self.cards,self.codes,self.struct = [],[],[]
        self.struct_valid_flag = False

    def cards2codes(self,ruler):
        self.codes = [ruler.get_card_encoding(card) for card in self.cards]
        self.codes = sorted(self.codes)
        self.codes2struct(ruler)

    def codes2struct(self,ruler):
        self.struct = ruler.get_struct(self.codes)[-1]
        self.struct_valid_flag = True

    
    def get_action(self,global_info,action_options):
        return action_options[np.random.choice(len(action_options))]

    def update_dataset(self,global_info,stack_score=0,is_terminated=False):
        pass

    def remove_codes_by_code(self,codes,codes_selected,ruler):
        codes = np.sort(codes)
        codes_selected = np.sort(codes_selected)
        idx1,idx2 = 0,0
        len1,len2 = len(codes),len(codes_selected)
        codes_remain = []
        while (idx2 < len2) and (idx1 < len1):
            if codes_selected[idx2] == codes[idx1]:
                idx2 += 1
                idx1 += 1
            else:
                codes_remain.append(codes[idx1])
                idx1 += 1
        while (idx1 < len1):
            codes_remain.append(codes[idx1])
            idx1 += 1
        return codes_remain

    def remove_codes(self,action_selected,ruler):
        if action_selected[-1] and self.struct_valid_flag:
            struct_selected = action_selected[1]
            struct_dict = self.struct
            for component in struct_selected:
                component_suit_enc = ruler.get_code_suit_enc(component[1])
                suit_enc_num,suit_enc_struct = struct_dict[component_suit_enc]
                suit_enc_num -= component[0]
                suit_enc_struct = [c_ for c_ in suit_enc_struct if c_[0] != component[0] or c_[1] != component[1]]
                struct_dict[component_suit_enc] = (suit_enc_num,suit_enc_struct)
            self.struct = struct_dict
            codes_remain = []
            for suit_enc in range(4):
                for component in struct_dict[suit_enc][1]:
                    codes_remain.extend(component[-1])
        else:
            self.struct_valid_flag = False
            codes_remain = self.remove_codes_by_code(self.codes,action_selected[0],ruler)

        self.codes = codes_remain

    def call(self,curr_number,curr_house,call_history,ruler):
        curr_call_num = 0
        if call_history is not None and len(call_history) > 0:
            curr_call_num = call_history[-1][-1]
        own_num = np.zeros((4,))
        for card in self.cards:
            if card[1] == curr_number and card[0] > 0:
                own_num[card[0]-1] += 1
        suit = np.argmax(own_num)
        num = own_num[suit]
        if num > curr_call_num:
            return (suit+1,num)
        else:
            return None

    def stack(self,call_history,ruler):
        assert len(self.codes) == 33
        struct = sum([self.struct[suit_enc][1] for suit_enc in range(4)],[])
        codes_options = self.generate_fold_options_heuristic(struct,ruler,budget=8,value_flag=1)
        stack = codes_options[np.random.choice(len(codes_options))]
        self.codes = self.remove_codes_by_code(self.codes,stack,ruler)
        assert len(self.codes) == 25
        self.codes2struct(ruler)
        return stack

    def play(self,global_info,is_first,debug_flag=False):
        ruler = global_info.ruler
        if is_first:
            action_options = self.generate_option_first(global_info)
        else:
            action_options = self.generate_option_follow(global_info)
        if debug_flag:
            print('options',[ruler.get_codes_repr(item[0]) for item in action_options])
        action_selected = self.get_action(global_info,action_options)
        self.remove_codes(action_selected,ruler)
        return action_selected

    def _eval_code_func(self,code,ruler,value_flag=1,direc_flag=-1):
        ## value flag: 1 = score max, 0 = score neutral, -1 = score min
        ## direc flag: 1 = max, -1 = min; max means the value is in reversed order
        value = code
        if value < 36:
            value = value % 12

        if ruler.get_code_score(code) > 0:
            if value_flag == 1:
                value += 12
            elif value_flag == -1:
                value = - value - 12

        if direc_flag == 1:
            return -value
        else:
            return value

        return value


    def generate_option_first(self,global_info):
        ## option = (codes,struct,is_clean)
        if not self.struct_valid_flag:
            self.codes2struct(global_info.ruler)
        struct_by_suit = self.struct

        action_options = []
        for suit_enc in range(4):
            for component in struct_by_suit[suit_enc][1]:
                action_options.append((component[-1],[component],True))
        return action_options

    def generate_option_follow(self,global_info):
        ## option = (codes,struct,is_clean)
        ruler = global_info.ruler
        round_num,round_struct,round_suit_enc = \
            global_info.round_num,global_info.round_struct,global_info.round_suit_enc
        best_struct,best_suit_enc = \
            global_info.best_struct,global_info.best_suit_enc

        if not self.struct_valid_flag:
            self.codes2struct(ruler)
        struct_by_suit = self.struct
        round_suit_enc_num = struct_by_suit[round_suit_enc][0]

        action_options = []
        if round_num == 1:
            ## generate components
            if round_suit_enc_num >= 1:
                components = struct_by_suit[round_suit_enc][1]
            else:
                components = sum([struct_by_suit[suit_enc][1] for suit_enc in range(4)],[])

            action_options = []
            for component in components:
                if component[0] == 1:
                    action_options.append((component[-1],[component],True))
                else:
                    for code in component[-1]:
                        action_options.append(([code],[(1,code,[code])],False))
            return action_options

        elif round_num == 2:
            ## generate components
            if round_suit_enc_num >= 2:
                ## check for pairs
                components = struct_by_suit[round_suit_enc][1]
                components1,components2,duplicate_flag = components,components,True
                if round_struct[0][0] >= 2:
                    components_pair = [component for component in components if component[0] >= 2]
                    if len(components_pair) > 0:
                        ## no cross, only within
                        components1,components2 = components_pair,[]
                        duplicate_flag = False
            elif round_suit_enc_num == 1:
                components1 = struct_by_suit[round_suit_enc][1]
                components2 = sum([struct_by_suit[suit_enc][1] for suit_enc in range(4) if suit_enc != round_suit_enc],[])
                duplicate_flag = False
            else:
                components1 = components2 = sum([struct_by_suit[suit_enc][1] for suit_enc in range(4)],[])
                duplicate_flag = True
            
            action_options = []
            for component1 in components1:
                ## within component
                if component1[0] > 1:
                    if component1[0] == 2:
                        action_options.append((component1[-1],[component1],True))
                    else:
                        for code1 in set(component1[-1]):
                            action_options.append(([code1,code1],[(2,code1,[code1,code1])],False))

                ## cross component
                for code1 in set(component1[-1]):
                    for component2 in components2:
                        for code2 in set(component2[-1]):
                            if not (code2 <= code1 and duplicate_flag):
                                if component1[0] == 1 and component2[0] == 1:
                                    action_options.append(([code1,code2],[component1,component2],True))
                                else:
                                    action_options.append(([code1,code2],[(1,code1,[code1]),(1,code2,[code2])],False))
            return action_options

        else:
            action_options = []
            ## first, try to follow with minimum costs
            beat_feas_tag = False ## bool to represent beating chances
            if round_suit_enc_num >= round_num:
                curr_struct = struct_by_suit[round_suit_enc][1]
                num_pairs_target = np.sum([component[0] // 2 for component in round_struct])
                num_pairs_curr = np.sum([component[0] // 2 for component in curr_struct])
                if num_pairs_curr > num_pairs_target:
                    ## has flexibility to choose pairs; use different value tags
                    beat_feas_tag = True
                    codes_pairs = list(set(sum([component[-1] for component in curr_struct if component[0] >= 2],[])))
                    cases = []
                    for value_flag in [0,1,-1]:
                        components_remain = [component for component in curr_struct if component[0] <= 1]
                        sort_idxs = np.argsort([self._eval_code_func(code,ruler,value_flag) for code in codes_pairs])
                        ## the remainings
                        for idx in sort_idxs[num_pairs_target:]:
                            components_remain.append((2,codes_pairs[idx],[codes_pairs[idx],codes_pairs[idx]]))
                        codes_selected = [codes_pairs[idx] for idx in sort_idxs[:num_pairs_target]]
                        codes_selected = sorted(codes_selected + codes_selected)
                        cases.append((codes_selected,components_remain))
                else:
                    ## all pairs should be selected
                    codes_selected = sum([component[-1] for component in curr_struct if component[0] >= 2],[])
                    components_remain = [component for component in curr_struct if component[0] <= 1]
                    cases = [(codes_selected,components_remain)]
            elif round_suit_enc_num == 0 and round_suit_enc != 3:
                codes_selected = []
                components_remain = sum([struct_by_suit[suit_enc][1] for suit_enc in range(4)],[])
                cases = [(codes_selected,components_remain)]
            else:
                codes_selected = sum([item[-1] for item in struct_by_suit[round_suit_enc][1]],[])
                components_remain = sum([struct_by_suit[suit_enc][1] for suit_enc in range(4) if suit_enc != round_suit_enc],[])
                cases = [(codes_selected,components_remain)]

            for codes_selected,components_remain in cases:
                if len(codes_selected) < round_num:
                    for value_flag in [0,1,-1]:
                        codes_options_tmp = self.generate_fold_options_heuristic(components_remain,ruler,round_num-len(codes_selected),value_flag=value_flag)
                        for codes in codes_options_tmp:
                            action_options.append((codes + codes_selected,None,False))
                else:
                    action_options.append((codes_selected,None,False))

            ## second, under certain circumstances, can check for beat options
            if beat_feas_tag and best_suit_enc == round_suit_enc:
                ## check if can beat with this suit
                curr_struct = struct_by_suit[round_suit_enc][1]
                action_options.extend(self.generate_beat_options_heuristic(best_struct,curr_struct,ruler,value_target_flag=-1))
            elif struct_by_suit[3][0] >= round_num:
                ## check if can beat with major suit
                struct_major = struct_by_suit[3][1]
                action_options.extend(self.generate_beat_options_heuristic(best_struct,struct_major,ruler,value_target_flag=1))
            return action_options

        return action_options

    def generate_fold_options_heuristic(self,components,ruler,budget,value_flag=1):
        ## select components at extreme
        values = [(component[0] * 10000 + sum([self._eval_code_func(code,ruler,value_flag) for code in component[-1]])) for component in components]
        sort_idxs = np.argsort(values)

        codes_selected = []
        for idx in sort_idxs:
            component = components[idx]
            curr_size,_,curr_codes = component
            if curr_size <= budget:
                budget -= curr_size
                codes_selected.extend(curr_codes)
                if budget == 0:
                    oversize = 0
                    break
            else:
                codes_selected.extend(curr_codes)
                oversize = curr_size - budget
                break
        codes_selected = sorted(codes_selected)

        if oversize == 0:
            codes_options = [codes_selected]
        elif oversize == 1:
            codes_options = []
            for idx1 in range(len(codes_selected)):
                codes_options.append(codes_selected[:idx1] + codes_selected[(idx1+1):])
            codes_options = set([tuple(codes) for codes in codes_options])
            codes_options = [list(codes) for codes in codes_options]
        elif oversize == 2:
            codes_options = []
            for idx1 in range(len(codes_selected)):
                for idx2 in range(idx1+1,len(codes_selected)):
                    codes_options.append(codes_selected[:idx1] + codes_selected[(idx1+1):idx2] + codes_selected[(idx2+1):])
            codes_options = set([tuple(codes) for codes in codes_options])
            codes_options = [list(codes) for codes in codes_options]
        else:
            values_select = [self._eval_code_func(code,ruler,value_flag) for code in codes_selected]
            sort_idxs = np.argsort(values_select)
            codes_options = [[codes_selected[idx] for idx in sort_idxs[oversize:]],
                             [codes_selected[idx] for idx in sort_idxs[:-oversize]]]
        return codes_options

    def generate_beat_options_heuristic(self,target_struct,curr_struct,ruler,value_target_flag=1):
        ## value target flag: -1 = min value, 1 = max value
        action_options = []
        ## try to beat target component with curr component
        bool_feas,_ = ruler.match_struct(target_struct,curr_struct)
        if bool_feas:
            ## try different beating options
            ## the heuristic is: if can beat, select the beating component, then match remains with the min-valued components

            min_idx_target,min_value_target,max_idx_target,max_value_target = \
                ruler.get_minmax_struct_value(target_struct)
            if value_target_flag == 1:
                idx_target,value_target = max_idx_target,max_value_target
            else:
                idx_target,value_target = min_idx_target,min_value_target
            target_struct_remain = target_struct[:idx_target] + target_struct[(idx_target+1):]

            ## take out the max
            ## TODO: actually no need to use match; just find beating options for the max component!
            max_codes_options = []
            for value_flag in [0,1,-1]:
                target_values = [component[0] * 10000 + sum([self._eval_code_func(code,ruler,value_flag) for code in component[-1]]) for component in target_struct]
                curr_values = [component[0] * 10000 + sum([self._eval_code_func(code,ruler,value_flag) for code in component[-1]]) for component in curr_struct]
                bool_feas,struct_matched = ruler.match_struct(target_struct,curr_struct,target_values,curr_values)
                _,_,max_idx_curr,max_value_curr = \
                    ruler.get_minmax_struct_value(struct_matched)
                if max_value_curr > value_target:
                    max_codes_options.append(tuple(struct_matched[max_idx_curr][-1]))
            max_codes_options = [list(codes) for codes in set(max_codes_options)]

            if len(max_codes_options) > 0:
                ## do the min
                codes_all = sum([component[-1] for component in curr_struct],[])
                for codes_selected_tmp in max_codes_options:
                    codes_remain_tmp = self.remove_codes_by_code(codes_all,codes_selected_tmp,ruler)
                    curr_struct_remain_tmp = ruler.get_struct(codes_remain_tmp)[1]
                    curr_struct_remain_tmp = sum([curr_struct_remain_tmp[suit_enc][1] for suit_enc in range(4)],[])
                    for value_flag in [0,1,-1]:
                        target_values = [- component[0] * 10000 - sum([self._eval_code_func(code,ruler,value_flag) for code in component[-1]]) for component in target_struct_remain]
                        curr_values = [- component[0] * 10000 - sum([self._eval_code_func(code,ruler,value_flag) for code in component[-1]]) for component in curr_struct_remain_tmp]
                        bool_feas,struct_remain_matched = ruler.match_struct(target_struct_remain,curr_struct_remain_tmp,target_values,curr_values)
                        if bool_feas:
                            codes_selected_remain = sum([component[-1] for component in struct_remain_matched],[])
                            action_options.append((codes_selected_tmp+codes_selected_remain,None,False))
                        else:
                            print('follow heuristic match problem!')

        return action_options