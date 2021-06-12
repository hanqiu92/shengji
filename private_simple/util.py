import numpy as np
import random
from env import Ruler,Game,DefaultPlayer,Env
from agents import IpyAgent
import asyncio
from IPython.display import clear_output
import pickle

def print_options(state):
    print('options',[state.ruler.get_codes_repr(item[0]) for item in state.action_set])

######################################
## generate valid/test datasets

def compress(state,codes_by_player):
    deck_size = state.deck_size
    compress_game = np.zeros((deck_size*2+4,),dtype=int)
    compress_game[-4:] = [deck_size,state.number,state.house+1,state.suit]
    for player_idx in range(1,6):
        if player_idx == 5:
            codes = state.stack
        else:
            codes = codes_by_player[player_idx - 1]
        for code in codes:
            if compress_game[code] > 0:
                compress_game[code + deck_size] = player_idx
            else:
                compress_game[code] = player_idx
    return compress_game
    
def decompress(compress_game,state_generator,player):
    deck_size,number,house,suit = compress_game[-4:]
    suit_size = (deck_size - 2) // 4
    house -= 1
    codes_by_player = [[] for _ in range(4)]
    stack = []
    for code in range(deck_size):
        for shift in [0,deck_size]:
            if compress_game[code+shift] == 5:
                stack.append(code)
            else:
                codes_by_player[compress_game[code+shift]-1].append(code)
                
    ruler = Ruler(suit_size=suit_size,major_level=number,major_suit=suit)
    structs_by_player = [ruler.get_struct(codes_by_player[curr_player])[-1] for curr_player in range(4)]
    
    ## initialize the state object
    state = state_generator(meta_info=(number,house,suit,ruler,stack,None))
    state.init_game_info((codes_by_player,structs_by_player))

    ## generate the first action set
    state.update_action_set(player.generate_option(state.structs_by_player[state.curr_player],state))
    info_set = state.get_info_set()
    return state,info_set,codes_by_player

def compare_compression(s1,cs1,s2,cs2):
    meta = np.all([s1.house == s2.house, s1.number == s2.number, s1.suit == s2.suit])
    codes = np.zeros((5,))
    for idx in range(5):
        if idx == 4:
            c1,c2 = s1.stack,s2.stack
        else:
            c1,c2 = cs1[idx],cs2[idx]
        for cc1,cc2 in zip(c1,c2):
            if cc1 != cc2:
                codes[idx] += 1
    codes = np.all(codes == 0)
    return (not meta or not codes)

def generate_random_game(env,if_display_private=False,if_display_public=False):
    house = np.random.choice([-1,0,1,2,3])
    number = np.random.choice(np.arange(1,env.game.suit_size+1,dtype=int))
    return env.init((house,number),if_display_private=if_display_private,if_display_public=if_display_public)

def generate_compress_games(N_iter,state_generator,save_fname='dataset/test_set.p'):
    env = Env(Game(),DefaultPlayer(),state_generator=state_generator)
    compress_games = []
    for _ in range(N_iter):
        state,info_set,codes_by_player = generate_random_game(env)
        compress_game = compress(state,codes_by_player)
        compress_games.append(compress_game)

    with open(save_fname,'wb') as fb:
        pickle.dump(compress_games,fb)

######################################
## training and evaluations

def display_game_meta(state,codes,if_display_private=False,if_display_public=False):
    ruler = state.ruler
    if if_display_private:
        print('major suit:{}, major level:{}, curr house:{}.'.format(ruler.suit_signs[state.suit-1],state.number+1,state.house))
        if if_display_public:
            print('stack: [{}]'.format(ruler.get_codes_repr(state.stack)))
            for curr_player in range(4):
                print('init:player:{} -> [{}],{}.'.format(curr_player,ruler.get_codes_repr(codes[curr_player]),len(codes[curr_player])))
        print('-' * 20)

def run_single_game(agents,state,info_set,env,if_display_private=False,if_display_public=False):
    end_flag = False
    observation_id = random.getrandbits(64)
    for agent in agents:
        agent.observe(observation_id=observation_id,prev_action=None,state=info_set,start_flag=True,end_flag=end_flag,full_state=state)
    while not end_flag:
        action = agents[info_set.curr_player].act(info_set,full_state=state)
        _,end_flag,state,info_set = env.step(state,action,if_display_private=if_display_private,if_display_public=if_display_public)
        observation_id = random.getrandbits(64)
        for agent in agents:
            agent.observe(observation_id=observation_id,prev_action=action,state=info_set,start_flag=False,end_flag=end_flag,full_state=state)
    game_score = (-1 if state.house % 2 == 0 else 1) * state.game_score
    eval_score = state.eval_score
    return game_score,eval_score

async def run_single_game_ipy(agents,state,info_set,env,if_display_private=False,if_display_public=False):
    end_flag = False
    observation_id = random.getrandbits(64)
    for agent in agents:
        agent.observe(observation_id=observation_id,prev_action=None,
                      state=info_set,start_flag=True,end_flag=end_flag,full_state=state)
    while not end_flag:
        if type(agents[info_set.curr_player]) == IpyAgent:
            action = await agents[info_set.curr_player].act(info_set,full_state=state,if_display_private=if_display_private,if_display_public=if_display_public)
            if action is None:
                action_codes = await agents[info_set.curr_player].act_single(info_set)
                action_codes = [code for code in action_codes if code >= 0]
                action = (action_codes,None,False)
            if if_display_public:
                clear_output(wait=True)
        else:
            action = agents[info_set.curr_player].act(info_set,full_state=state)
        _,end_flag,state,info_set = env.step(state,action,if_display_private=if_display_private,if_display_public=if_display_public)
        observation_id = random.getrandbits(64)
        for agent in agents:
            agent.observe(observation_id=observation_id,prev_action=action,
                          state=info_set,start_flag=False,end_flag=end_flag,full_state=state)
    game_score = (-1 if state.house % 2 == 0 else 1) * state.game_score
    eval_score = state.eval_score
    return game_score,eval_score

def run_games(agents,state_generator,N_iter=100,if_random_game=True,fname=None,if_train=False,train_freq=1,models=[],if_display_private=False,if_display_public=False):
    env = Env(Game(),DefaultPlayer(),state_generator=state_generator)
    curr_scores,eval_scores = [],[]
    if fname is not None:
        with open(fname,'rb') as fb:
            compress_games = pickle.load(fb)
        player = DefaultPlayer()
        N_iter = min(N_iter,len(compress_games))
        if_random_game = False
    else:
        if_random_game = True
    for iter_ in range(N_iter):
        if if_random_game:
            state,info_set,cs = generate_random_game(env,if_display_private=if_display_private,if_display_public=if_display_public)
        else:
            state,info_set,cs = decompress(compress_games[iter_],state_generator,player)
            display_game_meta(state,cs,if_display_private=if_display_private,if_display_public=if_display_public)
        curr_score,eval_score = run_single_game(agents,state,info_set,env,if_display_private=if_display_private,if_display_public=if_display_public)
        curr_scores.append(curr_score)
        eval_scores.append(eval_score)

        if if_train and (iter_+1) % train_freq == 0:
            for model in models:
                model.learn()
            for agent in agents:
                agent.update_train_meta()

    return np.array(curr_scores),np.array(eval_scores)

