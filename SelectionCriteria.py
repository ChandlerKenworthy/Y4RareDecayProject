# Useful imports, including all the necessary machine learning librares
import pandas as pd
import numpy as np
from sklearn import metrics
import uproot as up
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Define some constants used throughout the script
fName="/disk/moose/lhcb/djdt/Lb2L1520mueTuples/MC/2016MD/100FilesCheck/job185-CombDVntuple-15314000-MC2016MD_100F-pKmue-MC.root"

# Gather the required features and start trimming and organising the data
# We gather here all the Lambda B (LB) necessary data and particle ID
# Remember to drop the PID before we train the ML algorithm
with up.open(fName + ":DTT1520me/DecayTree") as f:
    features = ['Lb_PX', 'Lb_PY', 'Lb_PZ', 'Lb_TRUEID', 'Lb_ID', 'Lb_TAU', 'Lb_ENDVERTEX_X', 'Lb_ENDVERTEX_Y', 'Lb_ENDVERTEX_Z', 'Lb_DTF_PV_JPs_M012']
    # We can add or remove features and feature engineer as is required
    df = f.arrays(features, library="pd")

# Sort between what is and what is not signal by adding a category column
# A category of 1 is a signal and 0 is a background or non-signal event
def category_finder(row):
    if row['Lb_TRUEID'] == row['Lb_ID']:
        val = 1
    else:
        val = 0
    return val

df['category'] = df.apply(category_finder, axis=1)

# Start setting up our model which will detect signal
df.drop(['Lb_TRUEID', 'Lb_ID'], axis=1, inplace=True)
df_train = df.sample(frac=0.7, random_state=0)
df_valid = df.drop(df_train.index)

max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)

df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)
df_train.dropna(axis=1, inplace=True)
df_valid.dropna(axis=1, inplace=True)

X_train = df_train.drop('category', axis=1)
X_valid = df_valid.drop('category', axis=1)
y_train = df_train['category']
y_valid = df_valid['category']

print(X_train.head())

model = keras.Sequential([
    layers.Dense(4, activation='relu', input_shape=[7]),
    layers.Dense(4, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
print("...Compiling the model...")
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['binary_accuracy']
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=20,
    min_delta=0.001,
    restore_best_weights=True,
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=512,
    epochs=100,
    callbacks=[early_stopping]
)

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[5:, ['loss', 'val_loss']].plot()
history_df.loc[5:, ['binary_accuracy', 'val_binary_accuracy']].plot()

print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_binary_accuracy'].max()))