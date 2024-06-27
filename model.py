import numpy as np
import tensorflow as tf
from mido import MidiFile, MidiTrack, Message
import pretty_midi
import matplotlib.pyplot as plt

def analyze_input_file(midi_file):
    mid = MidiFile(midi_file)

    note_counts = {}

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'note_on' and msg.note != 0:
                note = msg.note
                if note not in note_counts:
                    note_counts[note] = 1
                else:
                    note_counts[note] += 1

    # Tworzenie histogramu
    notes = list(note_counts.keys())
    counts = list(note_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(notes, counts, color='skyblue')
    plt.xlabel('Nut')
    plt.ylabel('Liczba wystąpień')
    plt.title('Histogram wybranych nut')
    plt.grid(True)
    plt.savefig('note_histogram.png')
    plt.show()

def visualize_input_file(midi_file):
    mid = MidiFile(midi_file)

    note_times = []

    for track in mid.tracks:
        time = 0
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.note != 0:
                note_times.append((msg.note, time))

    # Przygotowanie danych do wizualizacji
    notes, times = zip(*note_times)

    plt.figure(figsize=(10, 6))
    plt.scatter(times, notes, color='skyblue', marker='o', s=10)
    plt.xlabel('Czas (tiki)')
    plt.ylabel('Nuty')
    plt.title('Wizualizacja sekwencji muzycznej')
    plt.grid(True)
    plt.savefig('music_sequence_visualization.png')  # Zapisywanie wizualizacji do pliku
    plt.show()
def generate_midi(model, pitchnames, n_vocab, output_length=500):
    pattern = []
    output_notes = []

    for _ in range(output_length):
        if not pattern:
            start = np.random.randint(0, len(pitchnames) - 1)
            pattern.append(pitchnames[start])
        else:
            prediction_input = [pitch_to_int.get(note, 0) for note in pattern]
            prediction_input = np.reshape(prediction_input, (1, len(pattern), 1))
            prediction_input = prediction_input / float(n_vocab)
            prediction = model.predict(prediction_input, verbose=0)
            index = np.argmax(prediction)
            result = pitchnames[index]

            # Prosta heurystyka: Jeśli model generuje tę samą nutę, spróbuj wylosować inną nutę
            if result == pattern[-1]:
                result = np.random.choice(pitchnames)

            pattern.append(result)
            pattern = pattern[1:len(pattern)]
            output_notes.append(result)

    return output_notes


def create_midi_file(notes, output_file):
    midi_data = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    empty_note_count = 0  # Licznik pustych nazw nut

    for idx, note_name in enumerate(notes):
        try:
            if note_name.strip():  # Sprawdź, czy nazwa nuty nie jest pusta ani złożona tylko z białych znaków
                note_number = int(note_name) + 7  # Poprawiamy nutę o 7 półtonów, aby uzyskać poprawny numer MIDI
                note_start_time = idx * 0.3  # Przykładowy czas trwania nuty, możesz dostosować według potrzeb
                note_end_time = note_start_time + 0.3  # Przykładowy czas trwania nuty, możesz dostosować według potrzeb
                note_obj = pretty_midi.Note(
                    velocity=100, pitch=note_number, start=note_start_time, end=note_end_time
                )
                piano.notes.append(note_obj)
            else:
                empty_note_count += 1  # Zwiększ licznik pustych nazw nut
                print(f"Uwaga: Pusta nazwa nuty na pozycji {idx}, pominięto.")
        except Exception as e:
            print(f"Błąd podczas przetwarzania nuty na pozycji {idx}: {e}")
            print(f"Nuta na pozycji {idx}: {note_name}")

    midi_data.instruments.append(piano)
    midi_data.write(output_file)
    print("Plik MIDI został pomyślnie utworzony.")
    print(f"Liczba pustych nazw nut: {empty_note_count}")
if __name__ == '__main__':
    model_path = "C:/Users/Jakub/PycharmProjects/MLPython/loss_approximated_0_6.keras"
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model został wczytany pomyślnie.")
    except Exception as e:
        print("Wystąpił błąd podczas wczytywania modelu:", str(e))
    midi_file_path = "generated_music.mid"

    # Generowanie nazw nut za pomocą pretty_midi
    pitchnames = [str(p) for p in range(21, 109)]  # Od nuty A0 (numer MIDI 21) do nuty C8 (numer MIDI 108)

    n_vocab = len(pitchnames)
    pitch_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    generated_notes = generate_midi(model, pitchnames, n_vocab)

    if generated_notes:
        print("Wygenerowane nuty:", generated_notes)
        create_midi_file(generated_notes, "generated_music.mid")
    else:
        print("Brak wygenerowanych nut. Sprawdź działanie funkcji generate_midi().")

    analyze_input_file(midi_file_path)
    visualize_input_file(midi_file_path)