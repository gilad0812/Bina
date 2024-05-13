from HalmaMove import halma
import numpy as np
import torch
from State import State
env = halma()
board = env.state.board
t = 7
# for i in range(4,8):            
#     for j in reversed(range(t,8)):
#         env.state.board[i][j] != 1
#     t -= 1

print(board)
       
board[0,3]= 0
board[1,4]= 1
board[5,7]= 0
board[4,6]= -1

print(board)
print("----------------")
board_r = np.flip(board*-1).T

print(board_r)
print("++++++++++++++++++++++++")
print((env.state.reverse()).board)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

