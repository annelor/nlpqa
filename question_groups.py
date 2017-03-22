import pandas as pd

df = pd.read_csv('data/train.csv')

questions = df[['qid1', 'question1']].set_index('qid1').to_dict()['question1']
questions.update(df[['qid2', 'question2']].set_index('qid2').to_dict()
                 ['question2'])

with open('data/groups.txt', 'r') as f:
    group_ids = f.readlines()

lines = list()
for idx, g_ids in enumerate(group_ids):
    qids = g_ids.strip('\n').split(', ')
    lines.append('\n%d' % idx)
    lines.append('\n'.join(questions[int(qid)] for qid in qids))

with open('data/question_groups.txt', 'w') as f:
    f.write('\n'.join(lines))
