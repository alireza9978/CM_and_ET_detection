import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from AddAttacksToUser import attack1, attack2, attack3, attack4, attack5, attack6
from models.Preprocessing import load_data_frame
from models.fill_nan import FillNanMode
from models.filters import select_random_user
from models_two.Visualization import plot

path = "/mnt/79e06c5d-876b-45fd-a066-c9aac1a1c932/Dataset/Power Distribution/irish.csv"
df = load_data_frame(path, False, False, FillNanMode.linear_auto_fill, True)
df, user_id = select_random_user(df)
df.set_index("date").groupby("id").apply(plot)
df = df.drop(columns=["id"]).set_index("date")

print(df.shape)

training_mean = df.mean()
training_std = df.std()
my_scaler = MinMaxScaler()
my_scaler.fit(df)
df.usage = my_scaler.transform(df)

TIME_STEPS = 24 * 7


# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps + 1):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(df.usage)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
print("Training input shape: ", x_train.shape)

# x_train, x_test = train_test_split(x_train, test_size=0.3)
division_point = int(x_train.shape[0] * 0.5)
x_test = x_train[division_point:]
x_train = x_train[:division_point]

y_test = []
attacks = [attack1, attack2, attack3, attack4, attack5, attack6]
for i in range(x_test.shape[0]):
    if random.random() > 0.7:
        x_test[i] = attacks[random.randint(0, 5)](x_test[i])
        y_test.append(1)
    else:
        y_test.append(0)

model = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()

history = model.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")],
)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# Get train MAE loss.
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No of samples")
plt.show()

# Get reconstruction loss threshold.
threshold = np.max(train_mae_loss)
print("Reconstruction error threshold: ", threshold)

x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)

plt.hist(test_mae_loss, bins=50)
plt.xlabel("Test MAE loss")
plt.ylabel("No of samples")
plt.show()

y_pred = test_mae_loss > threshold
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
