import pandas as pd
from gensim.models.word2vec import Word2Vec

model = Word2Vec.load('word2vec.model')

print(pd.DataFrame(model.wv.most_similar(positive=[u'吃', u'食物', u'食材', u'口味', u'味道',u'種類'], negative=[u'態度',u'環境',u'服務',u'價格'], topn=20), columns=['word', 'cos']))
