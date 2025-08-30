# %% 
# 1. IMPORT LIBRARIES
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense

# 2. DATA PREPARATION
# Sample English sentences
input_texts = ['hello', 'how are you', 'good morning']
# Corresponding French translations
target_texts = ['\tbonjour\n', '\tcomment ça va\n', '\tbondjour\n']  # \t=start, \n=end

# Create a set of unique characters for input and output
input_characters = sorted(list(set(''.join(input_texts))))
target_characters = sorted(list(set(''.join(target_texts))))

# Lengths
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)

# Create char-to-index dictionaries
input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

# 3. VECTORIZE DATA (one-hot encoding)
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[char]] = 1.0
        if t > 0:
            # Decoder target data is ahead by one timestep
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

# 4. BUILD SEQ2SEQ MODEL
# Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Complete model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 5. COMPILE AND TRAIN
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=1,
          epochs=200,
          validation_split=0.2)

# 6. INFERENCE MODE FOR TESTING
# Encoder model for prediction
encoder_model = Model(encoder_inputs, encoder_states)

# Decoder model for prediction
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

# 7. FUNCTION TO TRANSLATE ENGLISH TO FRENCH
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['\t']] = 1.0
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_characters[sampled_token_index]
        decoded_sentence += sampled_char
        if sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]
    return decoded_sentence

# Test translation
for seq_index in range(len(input_texts)):
    input_seq = encoder_input_data[seq_index:seq_index+1]
    print(f"English: {input_texts[seq_index]}")
    print(f"Predicted French: {decode_sequence(input_seq)}")

# %%
