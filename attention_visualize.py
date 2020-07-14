import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# plt.switch_backend('TkAgg')
plt.switch_backend('agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import json
from copy import deepcopy
import numpy as np

exp_name = 'result_concat_aux_attent_concat_visualized_weights_masked'
weight_path = '/media/luchy/HDD/results/%s/weights'% exp_name
plot_path = '/home/luchy/Desktop/AttentionGraphs/plots_7Tem'
date = '_gen31_dis7_2May'
if not os.path.exists(plot_path):
    os.mkdir(plot_path)

generated_captions_path = '/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_concat/densevid_eval/caption_%s%s.json'%(exp_name,date)
cc_np_vp_path = '/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_gen/parsers/parser_cc.json'

generated_captions = json.load(open(generated_captions_path, 'r'))
cc_np_vp = json.load(open(cc_np_vp_path, 'r'))

generated_captions_results = generated_captions['results']

generated_captions = {}
for key in generated_captions_results.keys():
    captions = []
    for i in range(len(generated_captions_results[key])):
        captions.append(generated_captions_results[key][i]['sentence'])
    generated_captions.update({key: captions})

files = os.listdir(weight_path)
print(len(files))
attention = {}
count = 0
for file in files:
    if count % 100 == 0:
        print count
    count += 1
    attention[file.strip('.npy')] = np.load(os.path.join(weight_path, file))

attention_weights = {}
print(len(attention.keys()))

for key in attention.keys():
    attention_weights.update({key.rsplit('_', 1)[0]: {}})

attention_weights_sent = deepcopy(attention_weights)
vid_key = 'v_fJNauQt9Di0'
count = 0
for key in attention.keys():
    if vid_key not in key:
        continue
    count += 1
    if count % 100 == 0:
        print(count)
    for key_ in attention.keys():
        if vid_key not in key:
            continue
        if key.rsplit('_', 1)[0] == key_.rsplit('_', 1)[0]:
            attention_weights[key.rsplit('_', 1)[0]].update({key_.rsplit('_', 1)[1]: attention[key_]})

count = 0
for key in attention_weights.keys():
    count += 1
    attent = []
    for k in sorted(attention_weights[key].keys()):
        attent.append(attention_weights[key][k])
    attention_weights_sent.update({key: attent})

vid_length = []
for sent in range(len(cc_np_vp[vid_key])):
    length = []
    for v in range(len(cc_np_vp[vid_key][sent])):
        length.append(len(cc_np_vp[vid_key][sent][v].split()))
    vid_length.append(length)

attention_weights_sent_noword = deepcopy(attention_weights_sent)
attention_weights_sent_word = deepcopy(attention_weights_sent)

for g in range(len(generated_captions[vid_key])):
    sent_len = len(generated_captions[vid_key][g].split())
    # total = np.sum(vid_length[g]) #This is true
    total = len(vid_length[g])
    attention_weights_sent_word[vid_key][g] = attention_weights_sent[vid_key][g][:sent_len, :total]
    attention_weights_sent_noword[vid_key][g] = attention_weights_sent[vid_key][g][:sent_len, total:]

attention_weights_sent = deepcopy(attention_weights_sent_word)
# This must be updated
# attention_weights_sent_upd = attention_weights_sent[vid_key]
# for g in range(len(vid_length)):
#     a=0
#     b=0
#     c=0
#     d=1
#     for w in range(len(vid_length[g])):
#         b = a + vid_length[g][w]
#         if b-a > 1:
#             attention_weights_sent_upd[g][w][c:d] = np.sum(attention_weights_sent[vid_key][g][w][a:b])
#         else:
#             attention_weights_sent_upd[g][w][c:d] = attention_weights_sent[vid_key][g][w][a:b]
#         a = b
#         c += 1
#         d += 1
attention_weights_sent_normal = deepcopy(attention_weights_sent)
attention_weights_sent_alternatif = deepcopy(attention_weights_sent)
for g in range(len(attention_weights_sent[vid_key])):
    for w in range(len(attention_weights_sent[vid_key][g])):
        total = np.sum(attention_weights_sent[vid_key][g][w])
        total_noword = np.sum(attention_weights_sent_noword[vid_key][g][w])
        total_alternatif = total + total_noword
        attention_weights_sent_normal[vid_key][g][w] = attention_weights_sent[vid_key][g][w] / total
        attention_weights_sent_alternatif[vid_key][g][w] = attention_weights_sent[vid_key][g][w] / total_alternatif
attention_weights_sent = deepcopy(attention_weights_sent_normal)

def showasfigure():
    for sent_num in range(len(attention_weights_sent[vid_key])):
        attent = attention_weights_sent[vid_key][sent_num]

        #change attention for video to numpy.array
        attention = np.zeros((len(attent), len(attent[0])))
        for i in range(len(attent)):
            for j in range(len(attent[i])):
                attention[i][j] = attent[i][j]

        caption = generated_captions[vid_key][sent_num].split()
        # colors = ['-r', '-b', '-c', '-g', '-m', '-y', '-k', '-slategrey', '-limegreen', '-deeppink', '-darkorange', '-tab:purple', '-lightcoral']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
        for i in range(attention.shape[1]):
            try:
                plt.plot(attention[:, i], colors[i], label=cc_np_vp[vid_key][sent_num][i])
            except:
                print ('Fault')
        x_values = np.arange(0, len(caption), 1)
        plt.xticks(x_values, caption, rotation=45)

        plt.legend(loc='upper right', fontsize='x-small')
        plt.title('Attention Results for ' + str(vid_key) + ' Event ' + str(sent_num))
        plt.savefig(plot_path + '/' + vid_key + '_'+  str(sent_num) +'.png')
        plt.clf()

    for sent_num in range(len(attention_weights_sent[vid_key])):
        attent_alternatif = attention_weights_sent_alternatif[vid_key][sent_num]

        # change attention for video to numpy.array
        attention = np.zeros((len(attent_alternatif), len(attent_alternatif[0])))
        for i in range(len(attent_alternatif)):
            for j in range(len(attent_alternatif[i])):
                attention[i][j] = attent_alternatif[i][j]

        noword = []
        for i in range(len(attention)):
            noword.append(1-attention[i].sum())

        caption = generated_captions[vid_key][sent_num].split()
        # colors = ['-r', '-b', '-c', '-g', '-m', '-y', '-k']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
        for i in range(attention.shape[1]):
            if i > len(colors):
                colors[i] = colors[0]
            plt.plot(attention[:, i], colors[i], label=cc_np_vp[vid_key][sent_num][i])

        plt.plot(noword, '-k', label='noword')

        x_values = np.arange(0, len(caption), 1)
        plt.xticks(x_values, caption, rotation=45)

        plt.legend(loc='upper right', fontsize='x-small')
        plt.title('Attention Alternatif Results for ' + str(vid_key) + ' Event ' + str(sent_num))
        plt.savefig(plot_path + '/' + 'Alternatif_' + vid_key + '_' + str(sent_num) + '.png')
        plt.clf()

def showAttention():
    color_map = plt.cm.get_cmap('magma_r')
    reversed_color_map = color_map.reversed()
    for sent_num in range(len(attention_weights_sent[vid_key])):
        attent_alternatif = attention_weights_sent_alternatif[vid_key][sent_num]

        # change attention for video to numpy.array
        attention = np.zeros((len(attent_alternatif), len(attent_alternatif[0]) + 1))
        for i in range(len(attent_alternatif)):
            for j in range(len(attent_alternatif[i])):
                attention[i][j] = attent_alternatif[i][j]
            attention[i][len(attent_alternatif[0])] = 1 - (attention[i].sum())

        # noword = []
        # for i in range(len(attention)):
        #     noword.append(1-attention[i].sum())

        caption = generated_captions[vid_key][sent_num].split()
        sent_cc_np_vp = cc_np_vp[vid_key][sent_num]
        sent_cc_np_vp.append('no bias')

        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #plt.title('Masked Attention Results for ' + str(vid_key) + ' Event ' + str(sent_num))

        #cax = ax.matshow(attention.T, cmap='magma_r', vmin=0, vmax=1) #cmap='bone'
        cax = ax.matshow(attention.T, cmap=reversed_color_map, vmin=0, vmax=1)  # cmap='bone'
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size=0.11, pad=0.3) #"4%"
        fig.colorbar(cax, cax=cax1)

        # Set up axes
        # ax.set_xticklabels([''] + input_sentence.split(' ') +
        #                    ['<EOS>'], rotation=90)
        ax.set_xticklabels([''] + caption, ha="right", rotation=45, rotation_mode="anchor")
        ax.set_yticklabels([''] + sent_cc_np_vp)

        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_tick_params(labelsize=8)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # plt.tight_layout()
        # plt.show()
        plt.savefig(plot_path + '/' + vid_key + '_' + str(sent_num) + '.png', bbox_inches = "tight")
        plt.clf()

# showasfigure()
showAttention()

print('Done')
