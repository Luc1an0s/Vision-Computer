import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

GESTOS_DIR = "gestos_dinamicos"
MODELOS_DIR = "modelos"
SEQ_LEN = 20

os.makedirs(MODELOS_DIR, exist_ok=True)

X, y = [], []


for arquivo in os.listdir(GESTOS_DIR):
    if arquivo.endswith(".csv"):
        caminho = os.path.join(GESTOS_DIR, arquivo)
        nome_gesto = os.path.splitext(arquivo)[0]
        try:
            df = pd.read_csv(caminho, header=None)
            dados = df.values  # (frames, 63)
            
            for i in range(0, len(dados) - SEQ_LEN + 1, SEQ_LEN):
                bloco = dados[i:i+SEQ_LEN]
                if bloco.shape == (SEQ_LEN, 63):
                    X.append(bloco)
                    y.append(nome_gesto)
        except Exception as e:
            print(f"Erro ao ler {arquivo}: {e}")

if not X:
    print("Nenhuma sequencia valida encontrada em 'gestos_dinamicos/'.")
    raise SystemExit

X = np.array(X) 
y = np.array(y)


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)


encoder_path = os.path.join(MODELOS_DIR, "label_encoder.pkl")
with open(encoder_path, "wb") as f:
    pickle.dump(encoder, f)


model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("Treinando LSTM em sequencias:", X.shape)
model.fit(X, y_cat, epochs=30, batch_size=16, verbose=1)


modelo_path = os.path.join(MODELOS_DIR, "modelo_gestos_dinamicos.h5")
model.save(modelo_path)

print("✅ Modelo dinamico (LSTM) treinado com sucesso!")
print("Gestos:", list(encoder.classes_))
print(f"Modelo salvo em: {modelo_path}")
print(f"Encoder salvo em: {encoder_path}")
