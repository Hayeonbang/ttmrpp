import os
import numpy as np
import librosa
from tqdm import tqdm

"""
16000으로 샘플링된 npy 파일들을 22050으로 샘플링하여 저장하는 스크립트
"""


# 원본 데이터가 있는 폴더 경로와 변환된 데이터를 저장할 폴더 경로 설정
input_folder = '/home/habang8/muscall/data/datasets/youtube/y_audio'  # 원본 npy 파일들이 있는 경로
output_folder = '/home/habang8/music-text-representation-pp/mtrpp/Audio/Youtube'   # 변환된 npy 파일들을 저장할 경로

# 출력 폴더가 존재하지 않으면 생성
os.makedirs(output_folder, exist_ok=True)

# 샘플링 레이트 설정
target_sr = 22050
original_sr = 16000

# 변환 과정에서 길이가 변한 파일들을 저장할 리스트
changed_length_files = []

# npy 파일들을 변환 및 저장하는 함수
def convert_and_save_files(input_folder, output_folder):
    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]
    for filename in tqdm(files, desc="Processing files"):
        # npy 파일 로드
        file_path = os.path.join(input_folder, filename)
        data = np.load(file_path)
        
        # 기존 데이터 길이
        original_length = len(data)
        # 변환 후 목표 길이
        target_length = int(original_length * (target_sr / original_sr))
        
        # 샘플링 레이트 변환
        data_resampled = librosa.resample(data, orig_sr=original_sr, target_sr=target_sr)
        
        # 길이 맞추기
        if len(data_resampled) < target_length:
            # 패딩
            data_processed = np.pad(data_resampled, (0, target_length - len(data_resampled)), 'constant')
            changed_length_files.append(filename)
        elif len(data_resampled) > target_length:
            # 자르기
            data_processed = data_resampled[:target_length]
            changed_length_files.append(filename)
        else:
            data_processed = data_resampled
        
        # 변환된 데이터 저장
        output_file_path = os.path.join(output_folder, filename)
        np.save(output_file_path, data_processed)

    # 길이가 변한 파일들 출력
    if changed_length_files:
        print("\nFiles with changed length after resampling:")
        for file in changed_length_files:
            print(file)

# 파일 변환 및 저장 실행
convert_and_save_files(input_folder, output_folder)