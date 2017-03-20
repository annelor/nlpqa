import numpy as np
import pandas as pd

# ########## IMPORT DATA #############

df = pd.read_csv('data/train.csv')
unique_qids = np.unique(np.hstack((df['qid1'], df['qid2'])))

print('Total number of questions = %d' % unique_qids.shape[0])

# ########### EXTRACT GROUPS ########

qid1 = df['qid1'].values[df['is_duplicate'].values == 1]
qid2 = df['qid2'].values[df['is_duplicate'].values == 1]

groups = [[]]
for idx in range(qid1.shape[0]):
    if idx % 100 == 0:
        print('%d / %d' % (idx, qid1.shape[0]))
    is_duplicate_pair = False
    for g in groups:
        if qid1[idx] in g and qid2[idx] not in g:
            g.append(qid2[idx])
            is_duplicate_pair = True
            break
        elif qid2[idx] in g and qid1[idx] not in g:
            g.append(qid1[idx])
            is_duplicate_pair = True
            break
    if not is_duplicate_pair:
        groups.append([qid1[idx], qid2[idx]])

groups.remove([])

lens = list()
for g in groups:
    lens.append(len(g))
idx = np.argsort(lens)[::-1]

for i in idx[:20]:
    print(groups[i])

# ########### WRITE GROUPS ########

lines = list()
for i in idx:
    lines.append(', '.join(str(e) for e in groups[i]))

with open('data/groups.txt', 'w') as f:
    f.write('\n'.join(lines))
