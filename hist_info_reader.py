import pickle

path = '/home/luchy/Desktop/results/result_concat_aux_attent_concat_15_lr2/'

histories = pickle.load(open(path + 'histories.pkl', 'r'))
infos = pickle.load(open(path + 'infos.pkl', 'r'))

g_val_results = histories['g_val_result_history']
d_val_results = histories['d_val_result_history']
g_loss_history = histories['g_loss_history']

#part for generator METEOR scores
max_M = 0
max_B = 0
max_C = 0

for key in g_val_results.keys():
    print (key, g_val_results[key]['lang_stats']['METEOR'])
    if g_val_results[key]['lang_stats']['METEOR'] > max_M:
        max_M = g_val_results[key]['lang_stats']['METEOR']
        max_M_key = key
    if g_val_results[key]['lang_stats']['Bleu_4'] > max_B:
        max_B = g_val_results[key]['lang_stats']['Bleu_4']
        max_B_key = key
    if g_val_results[key]['lang_stats']['CIDEr'] > max_C:
        max_C = g_val_results[key]['lang_stats']['CIDEr']
        max_C_key = key
print ('max METEOR epoch', max_M_key)
print ('max METEOR', max_M)
print ('Bleu_4 in max METEOR epoch', g_val_results[max_M_key]['lang_stats']['Bleu_4'])
print ('CIDEr in max METEOR epoch', g_val_results[max_M_key]['lang_stats']['CIDEr'])
print ('max Bleu_4 epoch', max_B_key)
print ('max Bleu_4', max_B)
print ('max CIDEr epoch', max_C_key)
print ('max CIDEr', max_C)


v_weight = 0.8
l_weight = 0.2
p_weight = 1.0

score = {}
max_e_score = 0
e_score = 0
max_key = 0

M_score = {}
max_M_score = 0
m_score = 0
max_val_M_key = 0

for key in d_val_results.keys():
    e_score = v_weight * (d_val_results[key]['val_results']['v_gen_accuracy'] + d_val_results[key]['val_results']['v_mm_accuracy']) + \
    l_weight * (d_val_results[key]['val_results']['l_gen_accuracy'] + d_val_results[key]['val_results']['l_neg_accuracy']) + \
    p_weight * (d_val_results[key]['val_results']['p_gen_accuracy'] + d_val_results[key]['val_results']['p_neg_accuracy'])
    score[key] = e_score
    m_score = d_val_results[key]['lang_scores']
    M_score[key] = m_score

    if e_score > max_e_score:
        max_score = e_score
        max_key = key
    if m_score > max_M_score:
        max_val_M_key = key
        max_M_score = m_score

print('max val score', max_score)
print('max val epoch', max_key)
print('max val M score', max_M_score)
print('max val epoch', max_val_M_key)
print('Done')