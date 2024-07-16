import os
import pandas as pd

def create_emotion_mapping():
    # Define the emotion to numerical value mapping
    return {
        'a': 0,
        'b': 1,
        'c': 2,
        'd': 3,
        'e': 4,
        'f': 5,
        'g': 6
    }

class AudioTextDataset:
    def __init__(self, file, audio_dir):
        self.data = pd.read_csv(file)
        self.audio_dir = audio_dir
        emotion_mapping = create_emotion_mapping()
        self.data['Label'] = self.data['file'].apply(lambda x: emotion_mapping[x.split('_')[1][0]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file = self.data.loc[idx, 'file'] + '.wav'
        audio_path = os.path.join(self.audio_dir, audio_file)
        text = self.data.loc[idx, 'text']
        emotion = self.data.loc[idx, 'Label']

        return audio_path, text, emotion

def load_dataset(file_path, audio_directory):
    return AudioTextDataset(file_path, audio_directory)

'''
# Example usage:
excel_file_path = 'Origin_Text.csv'
audio_directory = '/media/neuroai/5E1227AF12278B5B/Seminor_Emotion_Data_Preprocessing_Voice'  # Update with the actual path to the directory containing the .wav files

dataset = load_dataset(excel_file_path, audio_directory)

for idx in range(len(dataset)):
    audio, text, emotion = dataset[idx]
    print(f"Audio Path: {audio}, Text: {text}, Emotion: {emotion}")
'''