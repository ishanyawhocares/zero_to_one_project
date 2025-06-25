import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def create_sentiment_model(vocab_size, max_length, embedding_dim, lstm_units, dense_units, dropout_rate_lstm, dropout_rate_dense, learning_rate):
    """
    Creates a Keras Model for sentiment analysis with tunable hyperparameters.
    """
    inputs = Input(shape=(max_length,), name='text_input')
    x = Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)
    x = Bidirectional(LSTM(lstm_units, dropout=dropout_rate_lstm, recurrent_dropout=dropout_rate_lstm))(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate_dense)(x)
    outputs = Dense(3, activation='softmax', name='sentiment_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='sentiment_analyzer')

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
