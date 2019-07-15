import json,re
import jieba
from gensim.models.word2vec import Word2Vec

# data preprocessing

# 讀入資料
with open("test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 只留評論字串
comments = [i['content'] for i in data['data']]

# 有翻譯的只留翻譯(中文)
for index,value in enumerate(comments):
    match = re.search(r'\(由 Google 翻譯\)(.*)\(原文\)',value,flags=re.S)
    if match:
        comments[index] = match.group(1)

# 過濾非中文評論
def contains_chinese(s):
    match = re.search(r'[\u4e00-\u9fff]+',s)
    if match:
        return True
    return False

comments = list(filter(contains_chinese,comments))

# 過濾標點符號
def delete_punctuation(text):
    text = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff]+', '', text)
    return text

comments = list(map(delete_punctuation,comments))

# 結巴分詞
comments = [list(jieba.cut(i,cut_all=False)) for i in comments]

# 移除停用字詞
stopWords=[]

with open('stopWords.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopWords.append(data)

def delete_stopWords(arr):
    words = list(filter(lambda a: a not in stopWords and a != '\n', arr))
    return words

comments = list(map(delete_stopWords,comments))

print(comments)

# 訓練
model = Word2Vec(comments,sg=0,size=150,window=5)
model.save('word2vec.model')
