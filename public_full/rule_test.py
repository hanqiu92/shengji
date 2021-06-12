import numpy as np
import random
from util import Ruler,Env,Player

e = Env()
r = Ruler()
r.update(5,3)

##################################
## check enc and dec behavior
print('##################################')
print('check enc and dec behavior')
for card in e.decks:
    code = r.get_card_encoding(card)
    card_new = r.get_card_decoding(code)
    print(card,code,card_new,r.get_card_value(card))

##################################
## check struct extraction behavior
print('##################################')
print('check struct extraction behavior')
test_cases = [
    [(2,2),(2,2),(2,3),(2,3),(2,4),(2,4),(2,6),(2,6)],
    [(0,0),(0,0),(3,5),(3,5)],
    [(3,1),(3,1),(1,5),(1,5),(2,5),(2,5),(3,5),(3,5)],
    [(3,10),(3,10),(3,11),(3,11),(3,2),(3,3),(3,5),(3,5)],
    [(3,1),(3,1),(1,5),(1,5),(2,5),(2,5)],
    [(3,12),(3,13),(3,13),(3,1),(3,1)]
]
for cards in test_cases:
    curr_suit_enc,struct_dicts = r.get_components([r.get_card_encoding(card) for card in cards])
    print(cards,curr_suit_enc,struct_dicts)

##################################
## check struct extraction behavior 2
print('##################################')
print('check struct extraction behavior 2')
r2 = Ruler()
test_cases2 = [
    [45,45,47,47,48,49,49,51,52],
    [45,45,47,47,48,48,49,49,51,52],
    [45,45,47,47,48,49,49,50,51,51,52]
]
for codes in test_cases2:
    print(r2.get_components_suit(r.get_codes_encodings(codes),3))

##################################
## check struct compare behavior
print('##################################')
print('check struct compare behavior')
test_cases = [
    ([(1,0,[0]),(1,1,[1])], [(2,3,[3,3])]),
    ([(2,0,[0,0])], [(1,0,[0]),(1,1,[1])]),
    ([(1,6,[6]),(1,5,[5]),(1,11,[11]),(2,10,[10,10])], [(4,3,[3,3,3,3]),(1,2,[2])]),
    ([(2,10,[10,10]),(2,6,[6,6]),(2,5,[5,5])], [(4,3,[3,3,3,3]),(1,2,[2]),(1,1,[1])]),
    ([(6,1,[1,1,1,1,1,1]),(4,2,[2,2,2,2]),(4,3,[3,3,3,3])], [(6,4,[4,4,4,4,4,4]),(8,5,[5,5,5,5,5,5,5,5])])
]
for struct1,struct2 in test_cases:
    values1 = [sum([code % 12 if code < 36 else code for code in item[-1]]) for item in struct1]
    values2 = [sum([code % 12 if code < 36 else code for code in item[-1]]) for item in struct2]
    print(struct1,struct2,r.match_structure(struct1,struct2,values1,values2))