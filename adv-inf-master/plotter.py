import matplotlib.pyplot as plt
import cPickle as pickle

dir = '/Users/emre/Desktop/Advinf_Visual/'
read_file = 'files/histories_lang1.pkl'
save_file = 'plots/histories_lang1'
plot_label = 'Closest captions: 10, Encode size for aux: 10 Batch size:32-16 '

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
#
plt.xlabel(plot_label)
plt.legend(loc='upper left')
plt.title("Losses")

# plt.show() #to save the figure this line must be closed.
plt.savefig(dir+save_file+''+'.png')

#
# d_epoch = []
# d_val_epoch = []
# for key in sorted(histories['d_loss_history'].keys()):
#     if histories['d_loss_history'][key]['d_epoch'] not in d_epoch:
#         d_epoch.append(histories['d_loss_history'][key]['d_epoch'])
#
# dis_v_loss = {}
# dis_p_gen_accuracy = {}
# dis_v_mm_accuracy = {}
# dis_l_neg_accuracy = {}
# dis_v_gen_accuracy = {}
# dis_p_neg_accuracy = {}
# dis_l_loss = {}
# dis_l_gen_accuracy = {}
# dis_p_loss = {}
# for e in d_epoch:
#     dis_v_loss_ = []
#     dis_p_gen_accuracy_ = []
#     dis_v_mm_accuracy_ = []
#     dis_l_neg_accuracy_ = []
#     dis_v_gen_accuracy_ = []
#     dis_p_neg_accuracy_ = []
#     dis_l_loss_ = []
#     dis_l_gen_accuracy_ = []
#     dis_p_loss_ = []
#     for key in sorted(histories['d_loss_history'].keys()):
#         if histories['d_loss_history'][key]['d_epoch'] == e:
#             dis_v_loss_.append(histories['d_loss_history'][key]['dis_v_loss'])
#             dis_p_gen_accuracy_.append(histories['d_loss_history'][key]['dis_p_gen_accuracy'])
#             dis_v_mm_accuracy_.append(histories['d_loss_history'][key]['dis_v_mm_accuracy'])
#             dis_l_neg_accuracy_.append(histories['d_loss_history'][key]['dis_l_neg_accuracy'])
#             dis_v_gen_accuracy_.append(histories['d_loss_history'][key]['dis_v_gen_accuracy'])
#             dis_p_neg_accuracy_.append(histories['d_loss_history'][key]['dis_p_neg_accuracy'])
#             dis_l_loss_.append(histories['d_loss_history'][key]['dis_l_loss'])
#             dis_l_gen_accuracy_.append(histories['d_loss_history'][key]['dis_l_gen_accuracy'])
#             dis_p_loss_.append(histories['d_loss_history'][key]['dis_p_loss_'])
#     dis_v_loss.update({e: dis_v_loss_})
#     dis_p_gen_accuracy.update({e: dis_p_gen_accuracy_})
#     dis_v_mm_accuracy.update({e: dis_v_mm_accuracy_})
#     dis_l_neg_accuracy.update({e: dis_l_neg_accuracy_})
#     dis_v_gen_accuracy.update({e: dis_v_gen_accuracy_})
#     dis_p_neg_accuracy.update({e: dis_p_neg_accuracy_})
#     dis_l_loss.update({e: dis_l_loss_})
#     dis_l_gen_accuracy.update({e: dis_l_gen_accuracy_})
#     dis_p_loss.update({e: dis_p_loss_})
#
# dis_v_loss_epoch = {}
# for key in dis_v_loss.keys():
#     summed = sum(dis_v_loss[key]) / len(dis_v_loss[key])
#     dis_v_loss_epoch.update({key: summed})
#
# d_loss_epoch_plot = []
# for key in d_loss_epoch.keys():
#     d_loss_epoch_plot.append(d_loss_epoch[key])
#
# d_val_epoch_loss = {}
# for e in histories['d_val_result_history'].keys():
#     d_val_epoch_loss.update({e: histories['d_val_result_history'][e]['d_loss']})
#
# d_val_epoch_loss_plot = []
# for key in d_val_epoch_loss.keys():
#     d_val_epoch_loss_plot.append(d_val_epoch_loss[key])
#
# d_for_v = range(len(d_val_epoch_loss_plot))
# d_for_t = range(len(d_loss_epoch_plot))
# # plt.plot(iter, iter_loss, '-b', label='g_loss')
# plt.plot(d_for_v, d_val_epoch_loss_plot, '-m', label='d_val_loss')
# plt.plot(d_for_t, d_loss_epoch_plot, '-c', label='d_train_loss')

#
# plt.xlabel(plot_label)
# plt.legend(loc='upper left')
# plt.title("Losses")
#
# # plt.show() #to save the figure this line must be closed.
# plt.savefig(dir+save_file+''+'.png')
print('Done')