import numpy as np
import pandas as pd

# ########## IMPORT DATA #############

df = pd.read_csv('data/train.csv')
unique_qids = np.unique(np.hstack((df['qid1'], df['qid2'])))

print('Total number of questions = %d' % unique_qids.shape[0])

# ########### EXTRACT GROUPS ########

groups = [[]]
for idx in range(df.shape[0]):
    print(idx)
    if df['is_duplicate'][idx] == 1:
        is_duplicate_pair = False
        for g in groups:
            if df['qid1'][idx] in g and df['qid2'][idx] not in g:
                g.append(df['qid2'][idx])
                is_duplicate_pair = True
                break
            elif df['qid2'][idx] in g and df['qid1'][idx] not in g:
                g.append(df['qid1'][idx])
                is_duplicate_pair = True
                break
        if not is_duplicate_pair:
            groups.append([df['qid1'][idx], df['qid2'][idx]])

for g in groups:
    if len(g) > 2:
        print(g)
