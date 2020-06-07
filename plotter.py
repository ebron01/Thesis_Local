import matplotlib.pyplot as plt
import pickle
import os
"use Thesis_Local Environment"
SSD_dir = '/home/luchy/Desktop/results'
HDD_dir = '/media/luchy/HDD/results'
HDD = os.listdir(HDD_dir)
SSD = os.listdir(SSD_dir)

file_dirs = {'HDD': HDD, 'SSD': SSD}
for key in file_dirs:
    for file in file_dirs[key]:
        # if file != 'result_concat_sum_aux_cooccur':
        #     continue
        if file == 'plots' or file == 'plots_3Jun':
            continue
        try:
            if key == 'HDD':
                histories = pickle.load(open(os.path.join(HDD_dir, file, 'histories.pkl'), 'rb'))
            else:
                histories = pickle.load(open(os.path.join(SSD_dir, file, 'histories.pkl'), 'rb'))
        except Exception as e:
            print(e)
            continue
        if key == 'HDD':
            save_file_gen = os.path.join(HDD_dir, 'plots_3Jun',file) + '_gen1'
            save_file_dis = os.path.join(HDD_dir, 'plots_3Jun', file) + '_dis'
        else:
            save_file_gen = os.path.join(SSD_dir, 'plots_3Jun',file) + '_gen1'
            save_file_dis = os.path.join(SSD_dir, 'plots_3Jun', file) + '_dis'
        plot_label_gen = 'Generator losses plot for %s generator'% file
        plot_label_dis = 'Discriminator losses for %s discriminator'% file

        g_epoch = histories['g_val_result_history'].keys()
        # g_epoch = []
        # g_val_epoch = []
        # for key in sorted(histories['g_loss_history'].keys()):
        #     if histories['g_loss_history'][key]['g_epoch'] not in g_epoch:
        #         g_epoch.append(histories['g_loss_history'][key]['g_epoch'])

        g_loss = {}
        for e in g_epoch:
            g_loss_ = []
            for key in sorted(histories['g_loss_history'].keys()):
                if histories['g_loss_history'][key]['g_epoch'] == e:
                    g_loss_.append(histories['g_loss_history'][key]['g_loss'])
            g_loss.update({e: g_loss_})

        g_loss_epoch = {}
        for key in g_loss.keys():
            try:
                summed = sum(g_loss[key]) / len(g_loss[key])
            except:
                summed = 0
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
        # import numpy as np
        # g_loss_epoch_plot1 = np.load(os.path.join(HDD_dir, file, 'gen_losses.npy'))
        # a = {index: v for index, v in np.ndenumerate(g_loss_epoch_plot1)}
        # g_loss_epoch_plot1 = a[()]
        # g_loss_epoch_plot = []
        # for key in g_loss_epoch_plot1.keys():
        #     g_loss_epoch_plot.append(g_loss_epoch_plot1[key])
        x_for_v = range(len(g_val_epoch_loss_plot))
        x_for_t = range(len(g_loss_epoch_plot))
        # plt.plot(iter, iter_loss, '-b', label='g_loss')
        plt.plot(x_for_v, g_val_epoch_loss_plot, '-r', label='g_val_loss')
        plt.plot(x_for_t, g_loss_epoch_plot, '-b', label='g_train_loss')

        plt.xlabel(plot_label_gen)
        plt.legend(loc='upper left')
        plt.title("Losses")
        #
        # plt.show() #to save the figure this line must be closed.
        plt.savefig(save_file_gen + '.jpg')

        plt.close()

        d_epoch = []
        d_val_epoch = []
        try:
            for key in sorted(histories['d_loss_history'].keys()):
                if histories['d_loss_history'][key]['d_epoch'] not in d_epoch:
                    d_epoch.append(histories['d_loss_history'][key]['d_epoch'])
        except:
            continue


        dis_v_loss = {}
        dis_l_loss = {}
        dis_p_loss = {}
        # dis_p_gen_accuracy = {}
        # dis_v_mm_accuracy = {}
        # dis_l_neg_accuracy = {}
        # dis_v_gen_accuracy = {}
        # dis_p_neg_accuracy = {}
        # dis_l_gen_accuracy = {}

        for e in d_epoch:
            dis_v_loss_ = []
            dis_l_loss_ = []
            dis_p_loss_ = []
            # dis_p_gen_accuracy_ = []
            # dis_v_mm_accuracy_ = []
            # dis_l_neg_accuracy_ = []
            # dis_v_gen_accuracy_ = []
            # dis_p_neg_accuracy_ = []
            # dis_l_gen_accuracy_ = []
            for key in sorted(histories['d_loss_history'].keys()):
                if histories['d_loss_history'][key]['d_epoch'] == e:
                    dis_v_loss_.append(histories['d_loss_history'][key]['dis_v_loss'])
                    dis_l_loss_.append(histories['d_loss_history'][key]['dis_l_loss'])
                    dis_p_loss_.append(histories['d_loss_history'][key]['dis_p_loss'])
                    # dis_p_gen_accuracy_.append(histories['d_loss_history'][key]['dis_p_gen_accuracy'])
                    # dis_v_mm_accuracy_.append(histories['d_loss_history'][key]['dis_v_mm_accuracy'])
                    # dis_l_neg_accuracy_.append(histories['d_loss_history'][key]['dis_l_neg_accuracy'])
                    # dis_v_gen_accuracy_.append(histories['d_loss_history'][key]['dis_v_gen_accuracy'])
                    # dis_p_neg_accuracy_.append(histories['d_loss_history'][key]['dis_p_neg_accuracy'])
                    # dis_l_gen_accuracy_.append(histories['d_loss_history'][key]['dis_l_gen_accuracy'])
            dis_v_loss.update({e: dis_v_loss_})
            dis_l_loss.update({e: dis_l_loss_})
            dis_p_loss.update({e: dis_p_loss_})
            # dis_p_gen_accuracy.update({e: dis_p_gen_accuracy_})
            # dis_v_mm_accuracy.update({e: dis_v_mm_accuracy_})
            # dis_l_neg_accuracy.update({e: dis_l_neg_accuracy_})
            # dis_v_gen_accuracy.update({e: dis_v_gen_accuracy_})
            # dis_p_neg_accuracy.update({e: dis_p_neg_accuracy_})
            # dis_l_gen_accuracy.update({e: dis_l_gen_accuracy_})

        dis_v_loss_epoch = {}
        dis_l_loss_epoch = {}
        dis_p_loss_epoch = {}
        for key in dis_v_loss.keys():
            summed_v = sum(dis_v_loss[key]) / len(dis_v_loss[key])
            summed_l = sum(dis_l_loss[key]) / len(dis_l_loss[key])
            summed_p = sum(dis_p_loss[key]) / len(dis_p_loss[key])
            dis_v_loss_epoch.update({key: summed_v})
            dis_l_loss_epoch.update({key: summed_l})
            dis_p_loss_epoch.update({key: summed_p})

        d_v_loss_epoch_plot = []
        for key in dis_v_loss_epoch.keys():
            d_v_loss_epoch_plot.append(dis_v_loss_epoch[key])

        d_l_loss_epoch_plot = []
        for key in dis_v_loss_epoch.keys():
            d_l_loss_epoch_plot.append(dis_l_loss_epoch[key])

        d_p_loss_epoch_plot = []
        for key in dis_v_loss_epoch.keys():
            d_p_loss_epoch_plot.append(dis_p_loss_epoch[key])

        # d_v_gen_acc_epoch = {}
        # for e in histories['d_val_result_history'].keys():
        #     d_v_gen_acc_epoch.update({e: histories['d_val_result_history'][e]['val_results']['v_gen_accuracy']})
        #
        # d_val_epoch_loss_plot = []
        # for key in d_v_gen_acc_epoch.keys():
        #     d_val_epoch_loss_plot.append(d_v_gen_acc_epoch[key])

        # d_for_v = range(len(d_val_epoch_loss_plot))
        d_for_t_v = range(len(d_v_loss_epoch_plot))
        d_for_t_l = range(len(d_l_loss_epoch_plot))
        d_for_t_p = range(len(d_p_loss_epoch_plot))
        # plt.plot(iter, iter_loss, '-b', label='g_loss')
        # plt.plot(d_for_v, d_val_epoch_loss_plot, '-m', label='d_v_gen_acc')
        plt.plot(d_for_t_v, d_v_loss_epoch_plot, '-c', label='dis_t_v_loss')
        plt.plot(d_for_t_l, d_l_loss_epoch_plot, '-g', label='dis_t_l_loss')
        plt.plot(d_for_t_p, d_p_loss_epoch_plot, '-p', label='dis_t_p_loss')

        #
        plt.xlabel(plot_label_dis)
        plt.legend(loc='upper left')
        plt.title("Losses")

        # plt.show() #to save the figure this line must be closed.
        plt.savefig(save_file_dis + '.png')
        plt.close()

#This part is used before 3June
"""
dir = '/home/luchy/Desktop/results/result_concat/'
read_file = dir + 'histories.pkl'
result_dir = '/home/luchy/Desktop/results/'

save_file_gen = 'plots/result_concat_gen_23Mar'
save_file_dis = 'plots/result_concat_gen_dis_23Mar'
plot_label_gen = 'Generator losses plot for CC Concat'
plot_label_dis = 'Discriminator losses for CC Concat'


with open(read_file, 'rb') as f:
    histories = pickle.load(f)

g_epoch = histories['g_val_result_history'].keys()
# g_epoch = []
# g_val_epoch = []
# for key in sorted(histories['g_loss_history'].keys()):
#     if histories['g_loss_history'][key]['g_epoch'] not in g_epoch:
#         g_epoch.append(histories['g_loss_history'][key]['g_epoch'])

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

plt.xlabel(plot_label_gen)
plt.legend(loc='upper left')
plt.title("Losses")
#
# plt.show() #to save the figure this line must be closed.
plt.savefig(result_dir + save_file_gen + '.jpg')

plt.close()

d_epoch = []
d_val_epoch = []
for key in sorted(histories['d_loss_history'].keys()):
    if histories['d_loss_history'][key]['d_epoch'] not in d_epoch:
        d_epoch.append(histories['d_loss_history'][key]['d_epoch'])

dis_v_loss = {}
dis_l_loss = {}
dis_p_loss = {}
# dis_p_gen_accuracy = {}
# dis_v_mm_accuracy = {}
# dis_l_neg_accuracy = {}
# dis_v_gen_accuracy = {}
# dis_p_neg_accuracy = {}
# dis_l_gen_accuracy = {}

for e in d_epoch:
    dis_v_loss_ = []
    dis_l_loss_ = []
    dis_p_loss_ = []
    # dis_p_gen_accuracy_ = []
    # dis_v_mm_accuracy_ = []
    # dis_l_neg_accuracy_ = []
    # dis_v_gen_accuracy_ = []
    # dis_p_neg_accuracy_ = []
    # dis_l_gen_accuracy_ = []
    for key in sorted(histories['d_loss_history'].keys()):
        if histories['d_loss_history'][key]['d_epoch'] == e:
            dis_v_loss_.append(histories['d_loss_history'][key]['dis_v_loss'])
            dis_l_loss_.append(histories['d_loss_history'][key]['dis_l_loss'])
            dis_p_loss_.append(histories['d_loss_history'][key]['dis_p_loss'])
            # dis_p_gen_accuracy_.append(histories['d_loss_history'][key]['dis_p_gen_accuracy'])
            # dis_v_mm_accuracy_.append(histories['d_loss_history'][key]['dis_v_mm_accuracy'])
            # dis_l_neg_accuracy_.append(histories['d_loss_history'][key]['dis_l_neg_accuracy'])
            # dis_v_gen_accuracy_.append(histories['d_loss_history'][key]['dis_v_gen_accuracy'])
            # dis_p_neg_accuracy_.append(histories['d_loss_history'][key]['dis_p_neg_accuracy'])
            # dis_l_gen_accuracy_.append(histories['d_loss_history'][key]['dis_l_gen_accuracy'])
    dis_v_loss.update({e: dis_v_loss_})
    dis_l_loss.update({e: dis_l_loss_})
    dis_p_loss.update({e: dis_p_loss_})
    # dis_p_gen_accuracy.update({e: dis_p_gen_accuracy_})
    # dis_v_mm_accuracy.update({e: dis_v_mm_accuracy_})
    # dis_l_neg_accuracy.update({e: dis_l_neg_accuracy_})
    # dis_v_gen_accuracy.update({e: dis_v_gen_accuracy_})
    # dis_p_neg_accuracy.update({e: dis_p_neg_accuracy_})
    # dis_l_gen_accuracy.update({e: dis_l_gen_accuracy_})

dis_v_loss_epoch = {}
dis_l_loss_epoch = {}
dis_p_loss_epoch = {}
for key in dis_v_loss.keys():
    summed_v = sum(dis_v_loss[key]) / len(dis_v_loss[key])
    summed_l = sum(dis_l_loss[key]) / len(dis_l_loss[key])
    summed_p = sum(dis_p_loss[key]) / len(dis_p_loss[key])
    dis_v_loss_epoch.update({key: summed_v})
    dis_l_loss_epoch.update({key: summed_l})
    dis_p_loss_epoch.update({key: summed_p})

d_v_loss_epoch_plot = []
for key in dis_v_loss_epoch.keys():
    d_v_loss_epoch_plot.append(dis_v_loss_epoch[key])

d_l_loss_epoch_plot = []
for key in dis_v_loss_epoch.keys():
    d_l_loss_epoch_plot.append(dis_l_loss_epoch[key])

d_p_loss_epoch_plot = []
for key in dis_v_loss_epoch.keys():
    d_p_loss_epoch_plot.append(dis_p_loss_epoch[key])

d_v_gen_acc_epoch = {}
for e in histories['d_val_result_history'].keys():
    d_v_gen_acc_epoch.update({e: histories['d_val_result_history'][e]['val_results']['v_gen_accuracy']})

d_val_epoch_loss_plot = []
for key in d_v_gen_acc_epoch.keys():
    d_val_epoch_loss_plot.append(d_v_gen_acc_epoch[key])

# d_for_v = range(len(d_val_epoch_loss_plot))
d_for_t_v = range(len(d_v_loss_epoch_plot))
d_for_t_l = range(len(d_l_loss_epoch_plot))
d_for_t_p = range(len(d_p_loss_epoch_plot))
# plt.plot(iter, iter_loss, '-b', label='g_loss')
# plt.plot(d_for_v, d_val_epoch_loss_plot, '-m', label='d_v_gen_acc')
plt.plot(d_for_t_v, d_v_loss_epoch_plot, '-c', label='dis_t_v_loss')
plt.plot(d_for_t_l, d_l_loss_epoch_plot, '-g', label='dis_t_l_loss')
plt.plot(d_for_t_p, d_p_loss_epoch_plot, '-p', label='dis_t_p_loss')

#
plt.xlabel(plot_label_dis)
plt.legend(loc='upper left')
plt.title("Losses")

# plt.show() #to save the figure this line must be closed.
plt.savefig(result_dir+save_file_dis+''+'.png')

print('Done')
"""
