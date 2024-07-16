import os
import opensmile
import torch
import pandas as pd
import numpy as np
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
from dataloader.dataloader import AudioTextDataset, load_dataset
from tqdm import tqdm
import speech_recognition as sr
import subprocess
import shutil

class DataProcessor:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals
        )
        self.audio_features = []
        self.text_features = []

    def extract_audio_features(self):
        for idx in tqdm(range(len(self.dataset)), desc="Extracting Audio Features"):
            audio_path, _, emotion = self.dataset[idx]
            filename = os.path.basename(audio_path).split('.')[0]
            try:
                if os.path.exists(audio_path):
                    y = self.smile.process_file(audio_path)
                    audio_features = y.to_numpy().flatten()
                    self.audio_features.append([filename, emotion] + audio_features.tolist())
                else:
                    print(f"File not found: {audio_path}")
                    self.audio_features.append([filename, emotion] + [0.0] * 88)  # Assuming 88 features for eGeMAPSv02
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                self.audio_features.append([filename, emotion] + [0.0] * 88)

    def convert_to_pcm_wav(self, input_file, output_file):
        try:
            # 출력 파일의 디렉토리 경로 추출
            output_dir = os.path.dirname(output_file)

            # 디렉토리가 존재하지 않으면 생성
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # ffmpeg 명령어 실행
            result = subprocess.run(
                ['ffmpeg', '-i', input_file, '-acodec', 'pcm_s16le', output_file],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            # 오류 메시지 출력
            print(f'오류 발생: {e.stderr.decode()}')

    def STT(self):
        r = sr.Recognizer()
        # recognize_google() : Google Web Speech API
        # recognize_google_cloud() : Google Cloud Speech API
        # recognize_bing() : Microsoft Bing Speech API
        # recognize_houndify() : SoundHound Houndify API
        # recognize_ibm() : IBM Speech to Text API
        # recognize_wit() : Wit.ai API
        # recognize_sphinx() : CMU Sphinx (오프라인에서 동작 가능)
        output_text = pd.DataFrame(columns=['file', 'text'])
        
        shutil.rmtree('dataset/converted')
        # STT를 하기 위해 음성 파일을 PCM WAV 파일로 변환하고 동작해야함
        # 임시 PCM WAV 파일이 저장될 디렉토리를 비움

        for idx in tqdm(range(len(self.dataset)), desc="STT"):
            audio_path, _, emotion = self.dataset[idx]
            filename = os.path.basename(self.dataset[idx][0]).split('.')[0]
            converted_audio_path = os.path.join('dataset/converted', os.path.basename(self.dataset[idx][0]))
            self.convert_to_pcm_wav(audio_path, converted_audio_path)
            
            korean_audio = sr.AudioFile(converted_audio_path)
            try:
                with korean_audio as source:
                    audio = r.record(source)    
                stt_result = r.recognize_google(audio_data=audio, language='ko-KR')
            except:
                stt_result = ' '
            
            df = pd.DataFrame({'file': [filename], 'text': [stt_result]}, index=[idx])
            output_text = pd.concat([output_text, df])
        return output_text


    def extract_text_embeddings(self):
        tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        model = BertModel.from_pretrained('skt/kobert-base-v1')
        if self.args.text == 'stt':
            stt_text = self.STT()
        for idx in tqdm(range(len(self.dataset)), desc="Extracting Text Features"):
            _, text, emotion = self.dataset[idx]

            if self.args.text == 'stt':
                text = stt_text[idx]

            filename = os.path.basename(self.dataset[idx][0]).split('.')[0]

            if isinstance(text, float) and pd.isna(text):
                text = ''

            inputs = tokenizer.batch_encode_plus([text], padding=True, truncation=True, return_tensors='pt')
            outputs = model(**inputs)
            outputs = outputs[1].detach().squeeze(0).numpy().astype("float32")
            self.text_features.append([filename, emotion] + outputs.tolist())

    def combine_features(self):
        audio_df = pd.DataFrame(self.audio_features, columns=["filename", "emotion"] + [f"audio_feature{i}" for i in range(1, 89)])
        text_df = pd.DataFrame(self.text_features, columns=["filename", "emotion"] + [f"text_feature{i}" for i in range(1, 769)])
        combined_df = audio_df if text_df.empty else text_df if audio_df.empty else pd.merge(audio_df, text_df, on=['filename', 'emotion'])
        return combined_df

    def process(self, output_csv='combined_features.csv'):
        if 'a' in self.args.modality:
            self.extract_audio_features()
        if 't' in self.args.modality:
            self.extract_text_embeddings()
        # if self.args.modality == 'at':
        combined_df = self.combine_features()

        # combined_df.to_csv(output_csv, index=False)
        return combined_df

'''
# Example usage:
def main():
    file_path = 'Origin_Text.csv'
    audio_dir = '/media/neuroai/5E1227AF12278B5B/Seminor_Emotion_Data_Preprocessing_Voice'

    dataset = load_dataset(file_path, audio_dir)
    processor = DataProcessor(dataset)
    combined_features = processor.process()

    print(combined_features.head())


if __name__ == "__main__":
    main()
'''