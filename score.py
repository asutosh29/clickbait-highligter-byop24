import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import re

# Define the model
def create_model():
    model = models.Sequential([
        layers.Input(shape=(384,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(32, activation='sigmoid'),
        layers.BatchNormalization(),
        layers.Dense(1)
    ])

    # initial_learning_rate = 0.0005
    initial_learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

    model.compile(optimizer=optimizer,
                 loss='mse',
                 metrics=['mae'])
    return model

def get_embeddings(texts, tokenizer, model, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256, # Fine because Data set had very less exmples beyond this
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use attention-weighted mean pooling
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )

            # Get CLS token embedding and mean pooling
            cls_embedding = token_embeddings[:, 0, :]
            mean_embedding = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            # Combine CLS and mean pooling
            final_embedding = (cls_embedding + mean_embedding) / 2
            embeddings.extend(final_embedding.cpu().numpy())

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return np.array(embeddings)


def predict_single_example(model, input_vector):
    # Ensure input is in the correct shape (1, 384)

    if input_vector.ndim == 1:
        input_vector = input_vector.reshape(1, -1)

    # Make prediction
    prediction = model.predict(input_vector, verbose=0)[0][0]

    print(f"Predicted value: {prediction:.4f}")
    return prediction

def get_score(transformer_model,tokenizer,NN_model,text):
    X_sample = get_embeddings([text], tokenizer, transformer_model)
    sample_input = X_sample[0]  # or any other input vector of shape (384,)
    predicted_value = predict_single_example(NN_model, sample_input)
    return predicted_value

# Example usage:
# Take a single example from your test set

tokenizer = AutoTokenizer.from_pretrained("Clickbait1",is_split_into_words=True)
model = AutoModel.from_pretrained(
    "Clickbait1", output_hidden_states=True,output_attentions=True
)

# NN_model = tf.keras.models.load_model('D:\\Tech\\MachineLearning\\ClickBaitHighlighter\\models\\NN_truth_mean.keras')
import keras
NN_model = latest_model = keras.models.load_model('models/NN_model_title_moredata.keras')
# NN_model = create_model()
# NN_model.load_weights('D:\\Tech\\MachineLearning\\ClickBaitHighlighter\\models\\checkpoints.weights.h5')


# k = 2
# X_sample = get_embeddings(["A Click bait text"], tokenizer, model)
# sample_input = X_sample[0]  # or any other input vector of shape (384,)
# predicted_value = predict_single_example(NN_model, sample_input)

text = "President announce free PS4 for all kids who don't want to go school but become the greatest gamers of all time"
get_score(model, tokenizer,NN_model, text)

def get_value(text):
    return get_score(model,tokenizer,NN_model,text)

import pickle

    
def get_score_regressor(transformer_model,tokenizer,text):
    X_sample = get_embeddings([text], tokenizer, transformer_model)

    sample_input = X_sample  # or any other input vector of shape (384,)
    
    with open('models/new_scaler.pkl','rb') as f:
      print("Loading scaler")
      sc = pickle.load(f)
      print("Scaler loaded")
    sample_input=sc.transform(sample_input)

    with open('models/sgd_regressor.pkl', 'rb') as f:
      new_regressor = pickle.load(f)
    predicted_value = new_regressor.predict(sample_input)
    return predicted_value[0]

def get_value_regress(text):
    return get_score_regressor(model,tokenizer,text)

