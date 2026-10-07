import os
import time
import pandas as pd
from faster_whisper import WhisperModel
from concurrent.futures import ProcessPoolExecutor, as_completed

def init_worker():
    global model
    model = WhisperModel('tiny.en', device='cpu', compute_type='int8')

def transcribe_single(args):
    filepath, filename, label = args
    global model
    try:
        segments, info = model.transcribe(filepath, beam_size=1)
        text = ' '.join([s.text.strip() for s in segments])
        words = text.split()
        return {
            'filename': filename,
            'label': label,
            'transcript': text,
            'word_count': len(words),
            'language': info.language,
            'language_probability': info.language_probability
        }
    except Exception as e:
        print(f"Error transcribing {filename}: {e}")
        return {
            'filename': filename,
            'label': label,
            'transcript': '',
            'word_count': 0,
            'language': 'en',
            'language_probability': 0.0
        }

def process_transcription(csv_path, audio_dir, out_path, num_workers=4):
    df = pd.read_csv(csv_path)
    
    # Check if partially done
    existing = {}
    if os.path.exists(out_path):
        try:
            ex_df = pd.read_csv(out_path)
            existing = {row['filename']: row.to_dict() for _, row in ex_df.iterrows()}
            print(f"Found existing {len(existing)} transcribed files in {out_path}")
        except Exception:
            pass
            
    tasks = []
    for _, row in df.iterrows():
        fname = row['filename']
        lbl = row['label']
        if fname in existing:
            continue
        fpath = os.path.join(audio_dir, fname)
        tasks.append((fpath, fname, lbl))
        
    print(f"Transcribing {len(tasks)} files from {audio_dir} using {num_workers} workers...")
    t0 = time.time()
    results = list(existing.values())
    
    if len(tasks) > 0:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=init_worker) as executor:
            futures = {executor.submit(transcribe_single, task): task[1] for task in tasks}
            completed = 0
            for f in as_completed(futures):
                res = f.result()
                if res is not None:
                    results.append(res)
                completed += 1
                if completed % 25 == 0 or completed == len(tasks):
                    elapsed = time.time() - t0
                    print(f"[{completed}/{len(tasks)}] elapsed={elapsed:.1f}s ({(elapsed/completed):.2f}s/file)")
                    # Save incremental checkpoint
                    temp_df = pd.DataFrame(results)
                    temp_df = df[['filename']].merge(temp_df, on='filename', how='left')
                    temp_df.to_csv(out_path, index=False)
                    
    final_df = pd.DataFrame(results)
    final_df = df[['filename']].merge(final_df, on='filename', how='left')
    final_df.to_csv(out_path, index=False)
    print(f"Saved {len(final_df)} transcribed rows to {out_path} in {time.time()-t0:.1f}s")
    return final_df

if __name__ == '__main__':
    print("=== Transcribing Train Set ===")
    process_transcription('Dataset_Final/train.csv', 'Dataset_Final/train', 'transcripts_train.csv', num_workers=4)
    
    print("\n=== Transcribing Test Set ===")
    process_transcription('Dataset_Final/test.csv', 'Dataset_Final/test', 'transcripts_test.csv', num_workers=4)
    print("\nAll transcriptions completed successfully!")
