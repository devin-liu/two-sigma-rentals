
# coding: utf-8

# In[44]:

import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle


# In[45]:

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import TruncatedNormal
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, Reshape, Merge, BatchNormalization, TimeDistributed, Lambda, Activation, concatenate, LSTM, Flatten, Convolution1D, GRU, MaxPooling1D
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping


# In[4]:

train_pd = pd.read_json('train.json')
test_pd = pd.read_json('test.json')


# In[5]:

train_pd.head()


# In[6]:

y_train = train_pd.interest_level


# In[7]:

y_train.head()


# In[24]:

tokenizer = Tokenizer()
listing_features = train_pd.features.str.join(' ')
tokenizer.fit_on_texts(train_pd.description + listing_features)
train_description_word_sequences = tokenizer.texts_to_sequences(train_pd.description)
train_features_word_sequences = tokenizer.texts_to_sequences(listing_features)


# In[28]:

max_word_len = 36


# In[29]:

train_feat = pad_sequences(train_description_word_sequences, 
                              maxlen = max_word_len)
train_desc = pad_sequences(train_features_word_sequences, 
                              maxlen = max_word_len)


# In[30]:

word_index = tokenizer.word_index
embedding_dim = 300
nb_words = len(word_index)


# In[31]:

units = 128
dropout = 0.25
nb_filter = 32
filter_length = 3


# In[32]:

weights = TruncatedNormal(mean=0.0, stddev=0.05, seed=2)
bias = bias_initializer='zeros'


# In[37]:

model1 = Sequential()
model1.add(Embedding(nb_words + 1,
                    embedding_dim,
                    input_length=max_word_len,
                    trainable=False))
model1.add(Convolution1D(filters=nb_filter,
                        kernel_size=filter_length,
                        padding='same'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(dropout))
model1.add(Convolution1D(filters=nb_filter,
                        kernel_size=filter_length,
                        padding='same'))
model1.add(BatchNormalization())
model1.add(Activation('relu'))
model1.add(Dropout(dropout))
model1.add(Flatten())


# In[38]:

model2 = Sequential()
model2.add(Embedding(nb_words + 1,
                    embedding_dim,
                    input_length=max_word_len,
                    trainable=False))
model2.add(Convolution1D(filters=nb_filter,
                        kernel_size=filter_length,
                        padding='same'))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(dropout))
model2.add(Convolution1D(filters=nb_filter,
                        kernel_size=filter_length,
                        padding='same'))
model2.add(BatchNormalization())
model2.add(Activation('relu'))
model2.add(Dropout(dropout))
model2.add(Flatten())


# In[39]:

model3 = Sequential()
model3.add(Embedding(nb_words + 1,
                    embedding_dim,
                    input_length=max_word_len,
                    trainable=False))
model3.add(TimeDistributed(Dense(embedding_dim)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Dropout(dropout))
model3.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim,)))


# In[40]:

model4 = Sequential()
model4.add(Embedding(nb_words + 1,
                    embedding_dim,
                    input_length=max_word_len,
                    trainable=False))
model4.add(TimeDistributed(Dense(embedding_dim)))
model4.add(BatchNormalization())
model4.add(Activation('relu'))
model4.add(Dropout(dropout))
model4.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(embedding_dim,)))


# In[41]:

modela = Sequential()
modela.add(Merge([model1,model2],mode='concat'))
modela.add(Dense(units*2,kernel_initializer=weights,bias_initializer=bias))
modela.add(BatchNormalization())
modela.add(Activation('relu'))
modela.add(Dropout(dropout))

modela.add(Dense(units,kernel_initializer=weights,bias_initializer=bias))
modela.add(BatchNormalization())
modela.add(Activation('relu'))
modela.add(Dropout(dropout))


# In[42]:

modelb = Sequential()
modelb.add(Merge([model3,model4], mode='concat'))
modelb.add(Dense(units*2,kernel_initializer=weights,bias_initializer=bias))
modelb.add(BatchNormalization())
modelb.add(Activation('relu'))
modelb.add(Dropout(dropout))

modelb.add(Dense(units, kernel_initializer=weights,bias_initializer=bias))
modelb.add(BatchNormalization())
modelb.add(Activation('relu'))
modelb.add(Dropout(dropout))


# In[43]:

model = Sequential()
model.add(Merge([modela, modelb], mode='concat'))
model.add(Dense(units*2,kernel_initializer=weights,bias_initializer=bias))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(units, kernel_initializer=weights,bias_initializer=bias))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(units, kernel_initializer=weights,bias_initializer=bias))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))

model.add(Dense(1,kernel_initializer=weights,bias_initializer=bias))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[47]:

import time
# save the best weights for predicting the test question pairs
save_best_weights = 'question_pairs_weights.h5'

t0 = time.time()
callbacks = [ModelCheckpoint(save_best_weights, monitor='val_loss', save_best_only=True),
             EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')]
history = model.fit([train_feat, train_desc, train_feat, train_desc],
                    y_train,
                    batch_size=256,
                    epochs=2, #Use 100, I reduce it for Kaggle,
                    validation_split=0.15,
                    verbose=True,
                    shuffle=True,
                    callbacks=callbacks)
t1 = time.time()
print("Minutes elapsed: %f" % ((t1 - t0) / 60.))


# In[ ]:

model.save('rentals_weights_2.h5')
predictions = model.predict([train_q1[:10], train_q1[:10], train_q1[:10], train_q1[:10]], verbose = True)

