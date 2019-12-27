import matplotlib.pyplot as plt
import numpy as np
import ast
import torch
# sample_tensor = torch.rand(3)
dir = '/Users/emre/Desktop/Dense_Visual/plots/'
name = 'slurm-5093_out'
# num_epochs = int(name.split('_')[1].strip('epoch'))
type = '.out'
file = '/Users/emre/Desktop/'
filename = file + name + type
with open(filename, 'r') as f:
    data = f.readlines()

iter_line = []
v_iter_line = []
for line in data:
    if 'g_iter' in line:
        iter_line.append(line)

    elif 'evaluating validation preformance... 11/' in line:
        v_iter_line.append(line)

epoch = []
iter = []
iter_loss = []
epoch_loss = []
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

g_epoch_loss = []

for j in range(len(epoch)):
    start_index = epoch[j] * i
    end_index = (epoch[j] + 1) * i
    g_epoch_loss.append(sum(iter_loss[start_index:end_index]) / i)

v_count = 0
v_g_epoch_loss = []
for it in v_iter_line:
    print(v_count)
    v_count += 1
    v_iter_l = it.split(' ')
    try:
        v_g_epoch = float(ast.literal_eval(v_iter_l[4].strip('\n').strip('(').strip(')')))
    except :
        break
    v_g_epoch_loss.append(v_g_epoch)



for a in range(len(iter)):
    iter[a] = iter[a] / i

x_for_v = range(len(v_g_epoch_loss))
x_for_t = range(len(g_epoch_loss))
# plt.plot(iter, iter_loss, '-b', label='g_loss')
plt.plot(x_for_v, v_g_epoch_loss, '-r', label='v_g_loss')
plt.plot(x_for_t, g_epoch_loss, '-b', label='v_g_loss')

plt.xlabel(name)
plt.legend(loc='upper left')
plt.title("losses")


# plt.show() #to save the figure this line must be closed.
name = 'closest10_encodesize10'
plt.savefig(dir+name+''+'.png')
