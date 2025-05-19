import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import parselmouth
from pathlib import Path

# Function to extract vocal features from a single WAV file
def extract_vocal_features(audio_path, sr=22050):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Extract features
        features = {}
        
        # 1. Pitch (F0) and related features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_values = pitch_values[pitch_values > 0]  # Remove non-voiced frames
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # 2. Jitter (pitch perturbation)
        f0, voiced_flag, _ = librosa.pyin(y, fmin=75, fmax=600, sr=sr)
        f0 = f0[voiced_flag]  # Only voiced frames
        if len(f0) > 1:
            jitter = np.mean(np.abs(np.diff(f0)) / f0[:-1]) * 100  # Jitter percentage
            features['jitter'] = jitter
        else:
            features['jitter'] = 0
        
        # 3. Shimmer (amplitude perturbation)
        amplitudes = np.abs(y)
        if len(amplitudes) > 1:
            shimmer = np.mean(np.abs(np.diff(amplitudes)) / amplitudes[:-1]) * 100  # Shimmer percentage
            features['shimmer'] = shimmer
        else:
            features['shimmer'] = 0
        
        # 4. Formant frequencies (F1, F2, F3) using parselmouth
        try:
            snd = parselmouth.Sound(y, sampling_frequency=sr)
            formant = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)
            times = formant.xs()
            f1_values, f2_values, f3_values = [], [], []
            
            for t in times:
                f1 = formant.get_value_at_time(1, t) if formant.get_value_at_time(1, t) else np.nan
                f2 = formant.get_value_at_time(2, t) if formant.get_value_at_time(2, t) else np.nan
                f3 = formant.get_value_at_time(3, t) if formant.get_value_at_time(3, t) else np.nan
                if not np.isnan(f1) and 90 < f1 < 5000:
                    f1_values.append(f1)
                if not np.isnan(f2) and 90 < f2 < 5000:
                    f2_values.append(f2)
                if not np.isnan(f3) and 90 < f3 < 5000:
                    f3_values.append(f3)
            
            features['formant_f1'] = np.mean(f1_values) if f1_values else 0
            features['formant_f2'] = np.mean(f2_values) if f2_values else 0
            features['formant_f3'] = np.mean(f3_values) if f3_values else 0
        except:
            features['formant_f1'] = 0
            features['formant_f2'] = 0
            features['formant_f3'] = 0
        
        # 5. Temporal features (for rhythm tasks)
        # Speech rate (syllables per second) for rhythm tasks
        if 'rhythm' in audio_path.lower():
            envelope = np.abs(y)
            peaks, _ = find_peaks(envelope, height=np.mean(envelope), distance=int(sr * 0.05))  # Min 50ms between syllables
            duration = len(y) / sr
            if duration > 0:
                features['speech_rate'] = len(peaks) / duration  # Syllables per second
            else:
                features['speech_rate'] = 0
        else:
            features['speech_rate'] = 0
        
        # 6. Harmonics-to-Noise Ratio (HNR)
        try:
            hnr = librosa.effects.harmonic(y)
            noise = y - hnr
            if np.std(noise) > 0:
                features['hnr'] = 10 * np.log10(np.std(hnr) ** 2 / np.std(noise) ** 2)
            else:
                features['hnr'] = 0
        except:
            features['hnr'] = 0
        
        # 7. Duration of audio (after trimming)
        features['duration'] = len(y) / sr
        
        # 8. MFCC features (first 13 coefficients, mean and std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=int(sr * 0.01), win_length=int(sr * 0.025))
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfcc[i])
        
        return features
    
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Main function to process dataset
def process_dataset(dataset_path, output_csv):
    # Initialize feature list
    all_features = []
    file_names = []
    
    # Define task folders
    task_folders = [
        'A_sound', 'E_sound', 'I_sound', 'O_sound', 'U_sound',
        'KA_sound', 'PA_sound', 'TA_sound'
    ]
    
    # Process each task folder
    for task in task_folders:
        task_path = os.path.join(dataset_path, task)
        if not os.path.exists(task_path):
            print(f"Folder {task_path} does not exist, skipping...")
            continue
        
        print(f"Processing folder: {task}")
        for file_name in os.listdir(task_path):
            if file_name.endswith('.wav'):
                file_path = os.path.join(task_path, file_name)
                features = extract_vocal_features(file_path)
                
                if features is not None:
                    features['task'] = task
                    features['file_name'] = file_name
                    all_features.append(features)
                    file_names.append(file_name)
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Reorder columns for clarity
    columns = [
        'file_name', 'task', 'pitch_mean', 'pitch_std', 'pitch_range', 'jitter',
        'shimmer', 'formant_f1', 'formant_f2', 'formant_f3', 'speech_rate',
        'hnr', 'duration'
    ] + [f'mfcc_{i+1}_mean' for i in range(13)] + [f'mfcc_{i+1}_std' for i in range(13)]
    df = df[columns]
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")
    
    return df

# Example usage
if __name__ == "__main__":
    # Replace with your dataset path
    DATASET_PATH = "C:/Users/anshi/Desktop/speech/speech_dataset"
    OUTPUT_CSV = "vocal_features.csv"
    
    # Process the dataset
    features_df = process_dataset(DATASET_PATH, OUTPUT_CSV)
    print(features_df.head())