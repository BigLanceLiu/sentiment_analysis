import yaml
import sys
import importlib,sys
importlib.reload(sys)
from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(133)  # For Reproducibility
import jieba
import pandas as pd
import sys
sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 100
maxlen = 100
n_iterations = 10  # ideally more..
n_exposures = 3
window_size = 5
batch_size = 32
n_epoch = 3
input_length = 100
cpu_count = multiprocessing.cpu_count()


#加载训练文件
def loadfile():
    neg=pd.read_excel('data_2/neg.xls',header=None,index=None)
    pos=pd.read_excel('data_2/pos.xls',header=None,index=None)

    combined=np.concatenate((pos[0], neg[0]))
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))

    return combined,y

#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''

    text = [jieba.lcut(str(document).replace(',', '').replace('。','').replace('\n','')) for document in text]
    return text



#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined, epochs=model.iter, total_examples=model.corpus_count)
    model.save('lstm_data/Word2vec_model.pkl')

    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items():#从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)

    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):


    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    #model.add(LSTM(output_dim=50, activation='sigmoid', inner_activation='hard_sigmoid'))
    model.add(Bidirectional(LSTM(32, activation='relu',inner_activation='sigmoid')))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    #model.summary()
    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print("Train...")
    model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=n_epoch,verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('lstm_data/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('lstm_data/lstm.h5')
    print('Test score:', score)

#训练模型，并保存
def train():

    combined,y=loadfile()

    combined = tokenizer(combined)

    index_dict, word_vectors,combined=word2vec_train(combined)

    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)

    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)




def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('lstm_data/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

def lstm_predict(string):
    #print('loading model......')
    with open('lstm_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    #print('loading weights......')
    model.load_weights('lstm_data/lstm.h5')
    #model.compile(loss='binary_crossentropy',
    #              optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    #print(data)
    #result=model.predict_classes(data)
    predict = model.predict(data)
    if predict < 0.35:
        print(string,"特别差评")
    elif predict > 0.8:
    	print(string,"特别好评")
    elif predict >0.36 and predict <0.6:
    	print(string,"差评")
    else:
    	print(string,"好评")
    #print (predict)
    ''' 
    if result[0][0]==1:
        print(string,' positive')
    else:
        print(string,' negative')
    '''
if __name__=='__main__':
    #train()
    list = ['如果你不喜欢这部电影，说明他不是为你准备的，故事的终章是为读过故事的人准备的','我是一个90后，我曾经很羡慕“上一代人”：40年前的观众，他们的影院里有星战正传三部曲的落幕；20年前的观众，能在影院里看到指环王系列的终章。影迷们的悲欢并不相通，但现在我们也有了共同的记忆：漫威电影宇宙。谢谢你，《复仇者联盟》。','献给我人生中最美好的十一年。谢谢你，漫威，谢谢你让我的青春有了一个最完美的结局','“I AM IRONMAN! ” 既是开始也是结束。谢谢钢铁侠，谢谢漫威给我们带来的欢笑、泪水、感动以及爱你的3000个日日夜夜。','五星全给黑寡妇，她是最伟大的复仇者！！！','用三个小时的时间，与大家告别。是集结，是重聚，是告别，是牺牲。作为系列的最终篇，确实已经努力做到最好了，有笑点、有泪点、也有燃点。这个世代结束了，下个世代再会！','我爱 小罗伯特·唐尼 / 克里斯·埃文斯 / 克里斯·海姆斯沃斯 / 杰瑞米·雷纳 / 马克·鲁弗洛 / 斯嘉丽·约翰逊 以及漫威所有的人！十年青春，感谢有你，此生不悔！！！','三个小时剧情太拖沓，成了七龙珠寻宝了，堆砌拼凑的剧情，真是不知道浪费睡觉时间来看首映。','抱歉，这谢幕我实在吹不出口，剧情逻辑bug我都不care了，人物ooc和潦草处理真的不能接受。','将近300元买的杜比首映场，是真的不值。除了花里胡哨的特效外，一无是处，没有插科打诨和反高潮冷笑话，漫威就不会拍电影啦？花了十年塑造的人物形象性格在本作随意全部推翻，每个人所做的决策都不计后果，意气用事，幼稚且儿戏。巨作难逃烂尾，或许真的是个定律。']
    for string in list:
        lstm_predict(string)
    
