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
    plt.savefig('note_histogram_mozart.png')
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
    plt.savefig('music_sequence_visualization_mozart.png')  # Zapisywanie wizualizacji do pliku
    plt.show()

midi_file_path = "mozart/mz_311_2.mid"

analyze_input_file(midi_file_path)
visualize_input_file(midi_file_path)