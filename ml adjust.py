import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Flatten
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.layers import Dropout

# load data
df = pd.read_csv('all_gene_with_pse.csv')

# split data into training and test sets
train_x, test_x, train_y, test_y = train_test_split(df['sequence'], df['label'], test_size=0.2, random_state=42)

# Tokenize sequences
max_len = 500
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['sequence'])

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

# Pad sequences
train_x = pad_sequences(train_x, maxlen=max_len, padding='post')
test_x = pad_sequences(test_x, maxlen=max_len, padding='post')

# convert labels into categorical data
encoder = LabelEncoder()
encoder.fit(pd.concat([train_y,test_y]))
train_y = encoder.transform(train_y)
train_y = to_categorical(train_y)
test_y = encoder.fit_transform(test_y)
test_y = to_categorical(test_y)

# create model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len))

'''model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len))
model.add(SpatialDropout1D(0.5))
model.add(Conv1D(128, kernel_size=3))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))'''
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 创建自定义的Adam优化器，并设置学习率
custom_adam = Adam(learning_rate=0.001)  # 您可以根据需要更改这个值

# compile model
model.compile(loss='categorical_crossentropy', optimizer=custom_adam, metrics=['acc'])


# train model
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0,mode='auto')
history= model.fit(train_x, train_y, epochs=30, batch_size=5, validation_data=(test_x, test_y))
'''history= model.fit(train_x, train_y, epochs=100, batch_size=5, validation_data=(test_x, test_y), callbacks=[earlystop])
# evaluate model
score, acc = model.evaluate(test_x, test_y, verbose=2)
print('test accuracy', acc)'''

#save model
model.save('sequence_model.h5')

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()