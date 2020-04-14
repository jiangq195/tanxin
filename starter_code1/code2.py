import json
from collections import Counter

def read_corpus(filepath):
    """
    读取给定的语料库，并把问题列表和答案列表分别写入到 qlist, alist 里面。 在此过程中，不用对字符换做任何的处理（这部分需要在 Part 2.3里处理）
    qlist = ["问题1"， “问题2”， “问题3” ....]
    alist = ["答案1", "答案2", "答案3" ....]
    务必要让每一个问题和答案对应起来（下标位置一致）
    """

    with open(filepath) as f:
        data = json.load(f)

    qlist = []
    alist = []

    for item in data['data']:
        for para in item['paragraphs']:
            for qa in para['qas']:
                qlist.append(qa['question'])

                try:
                    alist.append(qa['answers'][0]['text'])

                except IndexError:
                    qlist.pop()

    assert len(qlist) == len(alist)

    return qlist, alist


qlist, alist = read_corpus('./data/train-v2.0.json')

q_all_list = []

for q in qlist:
    q_list = q.strip(' .!?').split()
    q_all_list.append(q_list)

print(len(q_all_list))
print(len(Counter(q_all_list).keys()))
