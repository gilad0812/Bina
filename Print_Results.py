import numpy as np
import torch
import matplotlib.pyplot as plt

Directory = 'Data'
# file 9 - vs rnd gamma 0.99 FAILED
# file 10 - vs rnd gamma 0.95 
# file 11 - vs fix gamma 0.95 FAILED
# file 12 - vs rnd gamma 0.95 R black FAILED
# file 13 - vs Fix gamma 0.95, epsiln = 1000 , fix.random = 0.60
# file 14 - vs rnd gamma 0.95 black R FAILED
# file 15 - vs rnd gamma 0.95 black R
# file 16 - vs rnd gamma 0.95 lr = 0.001 C = 25
# file 17 - vs rnd gamma 0.95 black R FAILED
# file 18 - vs rnd gamma 0.95 black FAILED
# file 19 - vs rnd gamma 0.95 black R
# file 20 - vs rnd gamma 0.95 BW 
# file 21 - vs rnd gamma 0.95 black 
Files_num = [22]
results_path = []
random_results_path = []
for num in Files_num:
    file = f'checkpoint{num}.pth'
    results_path.append(file)
    # file = f'random_results_{num}.pth'
    # random_results_path.append(file)

results = []
for path in results_path:
    results.append(torch.load(Directory+'/'+path))

# random_results = []
# for path in random_results_path:
#     random_results.append(torch.load(Directory+'/'+path))

for i in range(len(results)):
    print(results_path[i], max(results[i]['results']), np.argmax(results[i]['results']), len(results[i]['results']))
    results[i]['avglosses'] = list(filter(lambda k:  0< k <100, results[i]['avglosses'] ))

with torch.no_grad():
    for i in range(len(results)):
        fig, ax_list = plt.subplots(2,1)
        fig.suptitle(results_path[i])
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        ax_list[0].plot(results[i]['results'])
        
        ax_list[1].plot(results[i]['avglosses']) 
        plt.tight_layout()

plt.show()