import matplotlib.pyplot as plt
import cPickle as pickle

dir = '/Users/emre/Desktop/Advinf_Visual/'
read_file = 'files/histories_closest10_encodesize10_cont1.pkl'
save_file = 'plots/closest10_encodesize10'
plot_label = 'Closest captions: 10, Encode size for aux: 10'

with open(dir + read_file, 'rb') as f:
    histories = pickle.load(f)

g_epoch = []
g_val_epoch = []
for key in sorted(histories['g_loss_history'].keys()):
    if histories['g_loss_history'][key]['g_epoch'] not in g_epoch:
        g_epoch.append(histories['g_loss_history'][key]['g_epoch'])

g_loss = {}
for e in g_epoch:
    g_loss_ = []
    for key in sorted(histories['g_loss_history'].keys()):
        if histories['g_loss_history'][key]['g_epoch'] == e:
            g_loss_.append(histories['g_loss_history'][key]['g_loss'])
    g_loss.update({e: g_loss_})

g_loss_epoch = {}
for key in g_loss.keys():
    summed = sum(g_loss[key]) / len(g_loss[key])
    g_loss_epoch.update({key: summed})

g_loss_epoch_plot = []
for key in g_loss_epoch.keys():
    g_loss_epoch_plot.append(g_loss_epoch[key])

g_val_epoch_loss = {}
for e in histories['g_val_result_history'].keys():
    g_val_epoch_loss.update({e: histories['g_val_result_history'][e]['g_loss']})

g_val_epoch_loss_plot = []
for key in g_val_epoch_loss.keys():
    g_val_epoch_loss_plot.append(g_val_epoch_loss[key])

x_for_v = range(len(g_val_epoch_loss_plot))
x_for_t = range(len(g_loss_epoch_plot))
# plt.plot(iter, iter_loss, '-b', label='g_loss')
plt.plot(x_for_v, g_val_epoch_loss_plot, '-r', label='g_val_loss')
plt.plot(x_for_t, g_loss_epoch_plot, '-b', label='g_train_loss')

plt.xlabel(plot_label)
plt.legend(loc='upper left')
plt.title("Losses")

# plt.show() #to save the figure this line must be closed.
plt.savefig(dir+save_file+''+'.png')

print('Done')
# dir = '/Users/emre/Desktop/Dense_Visual/plots/'
# name = 'slurm-5101'
# new_name = 'slurm-5101_out'
# type = '.out'
# file = '/Users/emre/Desktop/'
# filename = file + name + type
# out_filename = file + new_name + type
# with open('slurm-5093.out', 'r') as f:
#     out = f.readlines()
#
# new = []
#
# for i in range(20000):
#     new.append('%d\n'%i)
#
# with open('slurm-5093_out.out', 'w') as newfile:
#     for line in out:
#          if line not in new:
#                 newfile.write(line)
# with open(out_filename, 'w') as newfile:
#     for line in out:
#         if 'ri : ' not in line and 'max_index' not in line:
#             newfile.write(line)
#
# print ('done')
# with open(filename, 'r') as f:
#     out = f.readlines()
# with open(out_filename, 'r') as f:
#     data = f.readlines()
#
# iter_line = []
# v_iter_line = []
# for line in data:
#     if 'g_iter' in line:
#         iter_line.append(line)
#
#     elif 'evaluating validation preformance... 11/' in line:
#         v_iter_line.append(line)
#
# epoch = []
# iter = []
# iter_loss = []
# epoch_loss = []
# c = 1
# i = 0
# for it in iter_line:
#     iter_l = it.split(' ')
#     g_epoch = ast.literal_eval(iter_l[3].strip(',').strip(')'))
#     g_iter = ast.literal_eval(iter_l[1])
#     g_iter_loss = float(ast.literal_eval(iter_l[6])[0])
#     if g_epoch == 13:
#         i += 1
#     if g_epoch not in epoch:
#         epoch.append(g_epoch)
#     iter.append(g_iter)
#     iter_loss.append(g_iter_loss)
#
# g_epoch_loss = []
#
# for j in range(len(epoch)):
#     start_index = epoch[j] * i
#     end_index = (epoch[j] + 1) * i
#     g_epoch_loss.append(sum(iter_loss[start_index:end_index]) / i)
#
# v_count = 0
# v_g_epoch_loss = []
# for it in v_iter_line:
#     print(v_count)
#     v_count += 1
#     v_iter_l = it.split(' ')
#     try:
#         v_g_epoch = float(ast.literal_eval(v_iter_l[4].strip('\n').strip('(').strip(')')))
#     except :
#         break
#     v_g_epoch_loss.append(v_g_epoch)
#
#
#
# for a in range(len(iter)):
#     iter[a] = iter[a] / i
#
# x_for_v = range(len(v_g_epoch_loss))
# x_for_t = range(len(g_epoch_loss))
# # plt.plot(iter, iter_loss, '-b', label='g_loss')
# plt.plot(x_for_v, v_g_epoch_loss, '-r', label='v_g_loss')
# plt.plot(x_for_t, g_epoch_loss, '-b', label='v_g_loss')
#
# plt.xlabel(name)
# plt.legend(loc='upper left')
# plt.title("losses")
#
#
# # plt.show() #to save the figure this line must be closed.
# name = 'closest10_encodesize10'
# plt.savefig(dir+name+''+'.png')