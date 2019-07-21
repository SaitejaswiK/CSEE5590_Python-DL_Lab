import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Activation, Dense, Embedding, SimpleRNN, LSTM
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
import time
from keras import metrics
print('import done')

DATA_FILE = 'SPAM_text.csv'
df = pd.read_csv(DATA_FILE,encoding='latin-1')
print(df.head())

tags = df.Category
texts = df.Message

num_max = 1000
# preprocess
le = LabelEncoder()
tags = le.fit_transform(tags)
tok = Tokenizer(num_words=num_max)
tok.fit_on_texts(texts)
mat_texts = tok.texts_to_matrix(texts,mode='count')
print(tags[:5])
print(mat_texts[:5])
print(tags.shape,mat_texts.shape)


def check_model(model,x,y):
    model.fit(x,y,batch_size=32,epochs=2,verbose=1,validation_split=0.2)


max_len = 100
cnn_texts_seq = tok.texts_to_sequences(texts)
print(cnn_texts_seq[0])
cnn_texts_mat = sequence.pad_sequences(cnn_texts_seq,maxlen=max_len)
print(cnn_texts_mat[0])
print(cnn_texts_mat.shape)


def get_rnn_model_v1():
    rnn_model = Sequential()
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    # 1000 is num_max
    rnn_model.add(Embedding(1000,
                            20,
                            input_length=max_len))

    rnn_model.add(SimpleRNN(32))
    rnn_model.add(Dense(1))
    rnn_model.add(Activation('sigmoid'))
    rnn_model.summary()

    rnn_model.compile(optimizer="adam",
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

    return rnn_model

m = get_rnn_model_v1()
check_model(m,cnn_texts_mat,tags)