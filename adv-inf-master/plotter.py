import matplotlib.pyplot as plt
import numpy as np
import ast
import torch
# sample_tensor = torch.rand(3)
dir = '/Users/emre/Desktop/Dense_Visual/plots/'
name = 'slurm-5079_out'
# num_epochs = int(name.split('_')[1].strip('epoch'))
type = '.out'
file = '/Users/emre/Desktop/'
filename = file + name + type
with open(filename, 'r') as f:
    data = f.readlines()

iter_line = []
for line in data:
    if 'g_iter' in line:
        iter_line.append(line)

epoch = []
iter = []
iter_loss = []
c = 1
i = 0
for it in iter_line:
    iter_l = it.split(' ')
    g_epoch = ast.literal_eval(iter_l[3].strip(',').strip(')'))
    g_iter = ast.literal_eval(iter_l[1])
    g_iter_loss = float(ast.literal_eval(iter_l[6])[0])
    if g_epoch == 0:
        i += 1
    if g_epoch not in epoch:
        epoch.append(g_epoch)
    iter.append(g_iter)
    iter_loss.append(g_iter_loss)

for a in range(len(iter)):
    iter[a] = iter[a] / i

plt.plot(iter, iter_loss, '-b', label='g_loss')

plt.xlabel(name)
plt.legend(loc='upper left')
plt.title("losses")


# plt.show() #to save the figure this line must be closed.
plt.savefig(dir+name+''+'.png')
