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
    compress_game = np.zeros((108+3,),dtype=int)
    compress_game[-3:] = [state.number,state.house+1,state.suit]
    for player_idx in range(1,6):
        if player_idx == 5:
            codes = state.stack
        else:
            codes = codes_by_player[player_idx - 1]
        for code in codes:
            if compress_game[code] > 0:
                compress_game[code + 54] = player_idx
            else:
                compress_game[code] = player_idx
    return compress_game
    
def decompress(compress_game,state_generator,player):
    number,house,suit = compress_game[-3:]
    house -= 1
    codes_by_player = [[] for _ in range(4)]
    stack = []
    for code in range(54):
        for shift in [0,54]:
            if compress_game[code+shift] == 5:
                stack.append(code)
            else:
                codes_by_player[compress_game[code+shift]-1].append(code)
                
    ruler = Ruler(number,suit)
    structs_by_player = [ruler.get_struct(codes_by_player[curr_player])[-1] for curr_player in range(4)]
    
    ## initialize the state object
    state = state_generator(meta_info=(number,house,suit,ruler,stack,None))
    state.init_game_info(structs_by_player)

    ## generate the first action set
    state.update_action_set(player.generate_option(state.structs_by_player[state.curr_player],state))
    return state,codes_by_player

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

def generate_random_game(env,if_display=False):
    house = np.random.choice([-1,0,1,2,3])
    number = np.random.choice(np.arange(1,14,dtype=int))
    return env.init((house,number),if_display=if_display)

def generate_compress_games(N_iter,state_generator,save_fname='dataset/test_set.p'):
    env = Env(Game(),DefaultPlayer(),state_generator=state_generator)
    compress_games = []
    for _ in range(N_iter):
        state,codes_by_player = generate_random_game(env)
        compress_game = compress(state,codes_by_player)
        compress_games.append(compress_game)

    with open(save_fname,'wb') as fb:
        pickle.dump(compress_games,fb)

######################################
## training and evaluations

def run_single_game(agents,state,env,if_display=False):
    end_flag = False
    observation_id = random.getrandbits(64)
    for agent in agents:
        agent.observe(observation_id=observation_id,prev_action=None,state=state,start_flag=True,end_flag=end_flag)
    while not end_flag:
        action = agents[state.curr_player].act(state)
        _,end_flag,state = env.step(state,action,if_display=if_display)
        observation_id = random.getrandbits(64)
        for agent in agents:
            agent.observe(observation_id=observation_id,prev_action=action,state=state,start_flag=False,end_flag=end_flag)
    game_score = (-1 if state.house % 2 == 0 else 1) * state.game_score
    eval_score = state.eval_score
    return game_score,eval_score

async def run_single_game_ipy(agents,state,env,if_display=False):
    end_flag = False
    observation_id = random.getrandbits(64)
    for agent in agents:
        agent.observe(observation_id=observation_id,prev_action=None,
                      state=state,start_flag=True,end_flag=end_flag)
    while not end_flag:
        if type(agents[state.curr_player]) == IpyAgent:
            action = await agents[state.curr_player].act(state)
            if action is None:
                action_codes = await agents[state.curr_player].act_single(state)
                action_codes = [code for code in action_codes if code >= 0]
                action = (action_codes,None,False)
            clear_output(wait=True)
        else:
            action = agents[state.curr_player].act(state)
        _,end_flag,state = env.step(state,action,if_display=if_display)
        observation_id = random.getrandbits(64)
        for agent in agents:
            agent.observe(observation_id=observation_id,prev_action=action,
                          state=state,start_flag=False,end_flag=end_flag)
    game_score = (-1 if state.house % 2 == 0 else 1) * state.game_score
    eval_score = state.eval_score
    return game_score,eval_score

def run_games(agents,state_generator,N_iter=100,if_random_game=True,fname=None,if_train=False,models=[],if_display=False):
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
            state,_ = generate_random_game(env,if_display=if_display)
        else:
            state,_ = decompress(compress_games[iter_],state_generator,player)
        curr_score,eval_score = run_single_game(agents,state,env,if_display=if_display)
        curr_scores.append(curr_score)
        eval_scores.append(eval_score)

        if if_train:
            for model in models:
                model.learn()

    return np.array(curr_scores),np.array(eval_scores)

