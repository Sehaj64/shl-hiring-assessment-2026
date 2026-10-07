import os
import time
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import opensmile
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

def init_worker():
    global smile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

def extract_single_audio(args):
    filepath, filename, label = args
    global smile
    try:
        # 1. openSMILE eGeMAPSv02
        egemaps = smile.process_file(filepath).reset_index(drop=True).to_dict(orient='records')[0]
        
        # 2. Audio reading
        y, sr = sf.read(filepath)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        
        dur = len(y) / sr if sr > 0 else 0.0
        
        # RMS & Energy
        frame_len, hop_len = 2048, 512
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        rms_max = float(np.max(rms)) if len(rms) > 0 else 0.0
        p10 = float(np.percentile(rms, 10)) if len(rms) > 0 else 0.0
        p90 = float(np.percentile(rms, 90)) if len(rms) > 0 else 0.0
        
        # Voice activity / silence
        silence_thresh = 0.05 * rms_max if rms_max > 0 else 0.0
        speech_frames = rms > silence_thresh
        speech_ratio = float(np.mean(speech_frames)) if len(speech_frames) > 0 else 0.0
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_len, hop_length=hop_len)[0]
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))
        zcr_max = float(np.max(zcr)) if len(zcr) > 0 else 0.0
        
        # Spectral
        cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_len)[0]
        flat = librosa.feature.spectral_flatness(y=y, hop_length=hop_len)[0]
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_len)[0]
        rolloff85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85, hop_length=hop_len)[0]
        rolloff95 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95, hop_length=hop_len)[0]
        
        # Onsets & Rhythm
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_len)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_len)
        onset_rate = len(onsets) / dur if dur > 0 else 0.0
        
        if len(onsets) > 1:
            onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop_len)
            ioi = np.diff(onset_times)
            ioi_mean = float(np.mean(ioi))
            ioi_std = float(np.std(ioi))
        else:
            ioi_mean, ioi_std = 0.0, 0.0
            
        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_len)
        contrast_means = np.mean(contrast, axis=1)
        contrast_stds = np.std(contrast, axis=1)
        
        # MFCCs 1-20
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop_len)
        mfcc_means = np.mean(mfcc, axis=1)
        mfcc_stds = np.std(mfcc, axis=1)
        
        # Delta MFCCs
        mfcc_delta = librosa.feature.delta(mfcc)
        delta_means = np.mean(mfcc_delta, axis=1)
        delta_stds = np.std(mfcc_delta, axis=1)
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_len)
        chroma_means = np.mean(chroma, axis=1)
        
        row_dict = {
            'filename': filename,
            'label': label,
            'duration': dur,
            'rms_mean': rms_mean,
            'rms_std': rms_std,
            'rms_max': rms_max,
            'rms_p90_p10': p90 - p10,
            'speech_ratio': speech_ratio,
            'zcr_mean': zcr_mean,
            'zcr_std': zcr_std,
            'zcr_max': zcr_max,
            'cent_mean': float(np.mean(cent)),
            'cent_std': float(np.std(cent)),
            'flat_mean': float(np.mean(flat)),
            'flat_std': float(np.std(flat)),
            'bw_mean': float(np.mean(bw)),
            'bw_std': float(np.std(bw)),
            'rolloff85_mean': float(np.mean(rolloff85)),
            'rolloff95_mean': float(np.mean(rolloff95)),
            'onset_rate': onset_rate,
            'ioi_mean': ioi_mean,
            'ioi_std': ioi_std,
            'onset_env_mean': float(np.mean(onset_env)),
            'onset_env_std': float(np.std(onset_env)),
        }
        
        for i in range(len(contrast_means)):
            row_dict[f'contrast_band_{i}_mean'] = float(contrast_means[i])
            row_dict[f'contrast_band_{i}_std'] = float(contrast_stds[i])
            
        for i in range(12):
            row_dict[f'chroma_{i}_mean'] = float(chroma_means[i])
            
        for i in range(20):
            row_dict[f'mfcc_{i+1}_mean'] = float(mfcc_means[i])
            row_dict[f'mfcc_{i+1}_std'] = float(mfcc_stds[i])
            row_dict[f'mfcc_delta_{i+1}_mean'] = float(delta_means[i])
            row_dict[f'mfcc_delta_{i+1}_std'] = float(delta_stds[i])
            
        row_dict.update(egemaps)
        return row_dict
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None

def process_dataset(csv_path, audio_dir, out_path, num_workers=6):
    df = pd.read_csv(csv_path)
    tasks = []
    for _, row in df.iterrows():
        fname = row['filename']
        lbl = row['label']
        fpath = os.path.join(audio_dir, fname)
        tasks.append((fpath, fname, lbl))
    
    print(f"Extracting features for {len(tasks)} files from {audio_dir} using {num_workers} workers...")
    t0 = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
        futures = {executor.submit(extract_single_audio, task): task[1] for task in tasks}
        completed = 0
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                results.append(res)
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                print(f"[{completed}/{len(tasks)}] elapsed={elapsed:.1f}s ({(elapsed/completed):.2f}s/file)")
                
    feat_df = pd.DataFrame(results)
    # Ensure correct ordering matching input csv
    feat_df = df[['filename']].merge(feat_df, on='filename', how='left')
    feat_df.to_csv(out_path, index=False)
    print(f"Saved {len(feat_df)} rows and {feat_df.shape[1]} columns to {out_path} in {time.time()-t0:.1f}s")
    return feat_df

if __name__ == '__main__':
    print("=== Processing Train Set ===")
    process_dataset('Dataset_Final/train.csv', 'Dataset_Final/train', 'features_train.csv', num_workers=6)
    
    print("\n=== Processing Test Set ===")
    process_dataset('Dataset_Final/test.csv', 'Dataset_Final/test', 'features_test.csv', num_workers=6)
    print("\nAll feature extraction finished successfully!")
