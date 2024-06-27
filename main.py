import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import mido

database_path = "C:/Users/Jakub/PycharmProjects/MLPython/mozart"


def get_notes(database_path):
    notes = []
    for file in glob.glob(database_path + "/*.mid"):
        print("Loading file:", file)
        midi = mido.MidiFile(file)
        for msg in midi:
            if msg.type == 'note_on' and msg.velocity != 0:
                print("Loaded note:", msg.note)
                notes.append(str(msg.note))
    print("Number of loaded notes:", len(notes))
    return notes


def prepare_sequences(notes, n_vocab):
    sequence_length = 100
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    network_input = []
    network_output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = tf.keras.utils.to_categorical(network_output)
    return network_input, network_output


def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(Dense(256))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return model


def train_network(database_path):
    notes = get_notes(database_path)
    if len(notes) == 0:
        print("No notes found in MIDI files. Try using different MIDI files.")
        return

    n_vocab = len(set(notes))
    print("Number of unique notes: ", n_vocab)
    network_input, network_output = prepare_sequences(notes, n_vocab)

    if len(network_output) == 0:
        print("No data in network_output. Try using different MIDI files.")
        return

    print("Length of network_input: ", len(network_input))
    print("Length of network_output: ", len(network_output))

    model = create_network(network_input, n_vocab)
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.keras"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    model.fit(network_input, network_output, epochs=100, batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network(database_path)
