import json

filename = 'sorted_10closest_updated_query_mid.json'
filename_vp_np = 'sorted_10closest_parsed_np_vp.json'
with open(filename, 'r') as f:
    query = json.load(f)

with open(filename_vp_np, 'r') as f:
    npvp = json.load(f)


all = {}
for key in query.keys():
    if 'concap' in key:
        continue
    # all.update({key: sorted({key: query[key], key + '_concap': query[key + '_concap'], key + '_np_vp' : npvp[key + '_concap']})})
    # all.update({key: {key: query[key], key + '_concap': query[key + '_concap'], key + '_np_vp': npvp[key + '_concap']}})
    part = {}
    for k in npvp[key + '_concap'].keys():
        part.update({k: {'caption': query[key + '_concap'][k]['caption'], 'np': npvp[key + '_concap'][k]['np'], 'vp': npvp[key + '_concap'][k]['vp'], 'order': npvp[key + '_concap'][k]['order'] }})
    all.update({key: {'caption': query[key][key], '_concap': part}})

with open('caption_np_vp_pairs.json', 'w') as f:
    json.dump(all, f)
print('Done')
