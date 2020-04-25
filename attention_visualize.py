import matplotlib.pyplot as plt
import os
import numpy as np
import json

weight_path = '/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_concat/weights'
generated_captions_path = '/home/luchy/PycharmProjects/Thesis_Local/adv_inf_master_v2_concat/densevid_eval/caption_result_concat_aux_attent_concat_visualized_23Nis_gen_40_dis2.json'
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

attention = {}
for file in files:
    attention[file.strip('.npy')] = np.load(os.path.join(weight_path, file))

attention_weights = {}
print(len(attention.keys()))

for key in attention.keys():
    attention_weights.update({key.rsplit('_', 1)[0]: {}})

attention_weights_sent = attention_weights

count = 0
for key in attention.keys():
    count += 1
    print(count)
    for key_ in attention.keys():
        if key.rsplit('_',1)[0] == key_.rsplit('_',1)[0]:
            attention_weights[key.rsplit('_',1)[0]].update({key_.rsplit('_', 1)[1]: attention[key_]})

count = 0
for key in attention_weights.keys():
    count += 1
    attent = []
    for k in sorted(attention_weights[key].keys()):
        attent.append(attention_weights[key][k])
    attention_weights_sent.update({key: attent})

vid_key = 'v_fJNauQt9Di0'
vid_length = []
for sent in range(len(cc_np_vp[vid_key])):
    length = []
    for v in range(len(cc_np_vp[vid_key][sent])):
        length.append(len(cc_np_vp[vid_key][sent][v].split()))
    vid_length.append(length)


for g in range(len(generated_captions[vid_key])):
    sent_len = len(generated_captions[vid_key][g].split())
    # total = np.sum(vid_length[g]) #This is true
    total = len(vid_length[g])
    attention_weights_sent[vid_key][g] = attention_weights_sent[vid_key][g][:sent_len, :total]

#This must be updated
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

for g in range(len(attention_weights_sent[vid_key])):
    for w in range(len(attention_weights_sent[vid_key][g])):
        total = np.sum(attention_weights_sent[vid_key][g][w])
        attention_weights_sent[vid_key][g][w] = attention_weights_sent[vid_key][g][w] / total


sent_num = 1
attent = attention_weights_sent[vid_key][sent_num]
attention = np.zeros((len(attent), len(attent[0])))
for i in range(len(attent)):
    for j in range(len(attent[i])):
        attention[i][j] = attent[i][j]

caption = generated_captions[vid_key][sent_num].split()
plt.plot(attention[:,0], '-r', label=cc_np_vp[vid_key][0][0])
plt.plot(attention[:,1], '-g', label=cc_np_vp[vid_key][0][1])
plt.plot(attention[:,2], '-p', label=cc_np_vp[vid_key][0][2])
plt.plot(attention[:,3], '-c', label=cc_np_vp[vid_key][0][3])

x_values = np.arange(0, len(caption), 1)
plt.xticks(x_values, caption, rotation=45)

plt.legend(loc='upper right', fontsize='x-small')
plt.title("Attention Results")
plt.show()

print('Done')
