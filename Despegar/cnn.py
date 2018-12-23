# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

# machine learning
import seaborn as sns
import matplotlib.pyplot as plt

train_df = pd.read_csv('train.csv')

train_df = train_df.drop(['many_names_for_document','many_holders_for_card','case_minutes_distance'
                         ,'count_different_installments','count_different_cards','eulerFriendsIds',
                         'domain_proc','eulerSocialNetwork','TimeOnPage','billingCountryCode' ], axis=1)

train_df['cardCountryCode'] = train_df.cardCountryCode.astype("category").cat.codes
train_df['caseDate'] = train_df.caseDate.astype("category").cat.codes
train_df['channel'] = train_df.channel.astype("category").cat.codes

train_df['countryCode'] = train_df.countryCode.astype("category").cat.codes
train_df['countryFrom'] = train_df.countryFrom.astype("category").cat.codes
train_df['countryTo'] = train_df.countryTo.astype("category").cat.codes

train_df['eaFirstVerificationDate'] = train_df.eaFirstVerificationDate.astype("category").cat.codes
train_df['eulerBadge'] = train_df.eulerBadge.astype("category").cat.codes
train_df['eulerBuyPaxDist'] = train_df.eulerBuyPaxDist.astype("category").cat.codes

train_df['eulerBuyTripType'] = train_df.eulerBuyTripType.astype("category").cat.codes
train_df['eulerSearchUrgency'] = train_df.eulerSearchUrgency.astype("category").cat.codes
train_df['iataFrom'] = train_df.iataFrom.astype("category").cat.codes

train_df['iataTo'] = train_df.iataTo.astype("category").cat.codes
train_df['ip_city'] = train_df.ip_city.astype("category").cat.codes
train_df['julixBrowserLanguage'] = train_df.julixBrowserLanguage.astype("category").cat.codes

train_df['julixOs'] = train_df.julixOs.astype("category").cat.codes
train_df['julixReasonCode'] = train_df.julixReasonCode.astype("category").cat.codes
train_df['julixTrueIpCity'] = train_df.julixTrueIpCity.astype("category").cat.codes

train_df['julixTrueIpRegion'] = train_df.julixTrueIpRegion.astype("category").cat.codes
train_df['paymentsCardType'] = train_df.paymentsCardType.astype("category").cat.codes
train_df['same_field_features'] = train_df.same_field_features.astype("category").cat.codes

df_filtered = train_df
df_filtered.update(df_filtered[df_filtered.columns].fillna(0))


from IPython.display import HTML
import reg_helper as RHelper
import numpy as np
import draw_nn
from matplotlib import pyplot as plt 
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import initializers
from fnn_helper import PlotLosses

X = df_filtered.drop("fraud", axis=1)
y = df_filtered.filter(items=["fraud"])
X = np.expand_dims(X, axis=2)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
X_train, X_test, y_train, y_test= train_test_split(X, dummy_y, test_size=0.2, random_state=42)

from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D,Conv2D,Conv1D
p = 0.25
default_initializer = 'RandomUniform'
model3=Sequential()

model3.add(Conv1D(filters=5, kernel_size=2, padding="valid", name='Conv1',input_shape=(82,1)))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Dropout(p))
model3.add(Conv1D(filters=5, kernel_size=3, padding="same", name='Conv2'))
model3.add(BatchNormalization())
model3.add(Activation('relu'))
model3.add(Dropout(p))

model3.add(Flatten())

model3.add(Dense(20, kernel_initializer=default_initializer,bias_initializer=default_initializer))

model3.add(Activation('relu'))

model3.add(Dense(2, activation='softmax'))


epochs = 10
#lr = 0.00000001
lr = 0.01
#optim = optimizers.sgd(lr=lr)
optim = optimizers.adam(lr=lr, decay=0.0002)
#optim = optimizers.rmsprop(lr=lr, decay=0.01)
model3.compile(loss = 'categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

batch_size = 256

checkpointer = ModelCheckpoint(filepath='cnn1.hdf5', verbose=1, save_best_only=True)
plot_losses = PlotLosses(plot_interval=1, evaluate_interval=None, x_val=X_test, y_val_categorical=y_test)

model3.fit(X_train, 
          y_train,
          epochs=epochs, batch_size=batch_size, 
          #verbose=1, 
          validation_data=(X_test, y_test), 
          callbacks=[plot_losses, checkpointer],
         )
