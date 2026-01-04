import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =====================
# LOAD & SORT LABEL
# =====================
DATA_DIR = "data"
labels = sorted(os.listdir(DATA_DIR))   # ⬅️ WAJIB SORT
label_map = {label: idx for idx, label in enumerate(labels)}

print("Label map:", label_map)

# =====================
# LOAD DATA
# =====================
X, y = [], []

for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        data = np.load(path)

        # SAFETY CHECK
        if data.shape != (30, 126):
            continue

        X.append(data.astype(np.float32))
        y.append(label_map[label])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("Dataset shape:", X.shape, y.shape)

# =====================
# MODEL (LEBIH STABIL)
# =====================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, 126)),
    Dropout(0.3),
    LSTM(32),
    Dense(32, activation="relu"),
    Dense(len(labels), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================
# TRAIN
# =====================
early_stop = EarlyStopping(
    monitor="loss",
    patience=5,
    restore_best_weights=True
)

model.fit(
    X, y,
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    shuffle=True
)

model.save("gesture_lstm.h5")
print("✅ Model berhasil dilatih & disimpan")
