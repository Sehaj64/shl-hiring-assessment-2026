import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

# Setup figure parameters
width, height = 1000, 560
dpi = 100
n_frames = 48

frames = []

for frame_idx in range(n_frames):
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 560)
    ax.axis('off')

    # Progress phase: 0.0 to 1.0
    t = frame_idx / n_frames
    phase = (t * 4) % 4  # 4 distinct animation cycles

    # Title Banner
    ax.text(500, 525, "END-TO-END MULTIMODAL SPOKEN GRAMMAR SCORING PIPELINE",
            fontsize=15, fontweight='bold', color='#ffffff', ha='center', va='center', family='sans-serif')
    ax.text(500, 502, "SHL AI Labs Hiring Assessment 2026 | Dual-Branch Acoustic + Linguistic Fusion",
            fontsize=10, color='#8b949e', ha='center', va='center', family='sans-serif')

    # Helper function for rounded boxes
    def draw_box(x, y, w, h, title, subtitle, items, bg_color, border_color, glow=False):
        if glow:
            glow_box = patches.FancyBboxPatch((x-3, y-3), w+6, h+6, boxstyle="round,pad=8",
                                              linewidth=2.5, edgecolor='#58a6ff', facecolor='none', alpha=0.6)
            ax.add_patch(glow_box)
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=6",
                                     linewidth=1.8, edgecolor=border_color, facecolor=bg_color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 18, title, fontsize=11, fontweight='bold', color='#ffffff', ha='center', va='center')
        if subtitle:
            ax.text(x + w/2, y + h - 34, subtitle, fontsize=8.5, color='#8b949e', ha='center', va='center')
        y_text = y + h - 52
        for it in items:
            ax.text(x + 12, y_text, it, fontsize=8, color='#c9d1d9', ha='left', va='center')
            y_text -= 15

    # 1. INPUT BOX (Left)
    input_glow = (phase < 1.0)
    draw_box(25, 200, 150, 160, "1. RAW AUDIO", "Candidate Response", 
             ["- 16 kHz Mono .wav", "- 45 to 60 sec duration", "- 769 Train / 216 Test", "- Acoustic Waveforms"], 
             '#161b22', '#30363d', glow=input_glow)

    # 2. ACOUSTIC BRANCH (Top Middle)
    ac_glow = (0.8 <= phase < 2.0)
    draw_box(230, 295, 240, 165, "BRANCH A: ACOUSTIC", "218 Descriptors (openSMILE)",
             ["* eGeMAPSv02: Pitch F0, F1-F3", "* Jitter, Shimmer, HNR, Loudness", "* Librosa: Speaking Tempo & WPM", "* Rhythm IOI Regularity & Energy", "* 20 MFCCs + Spectral Contrast"],
             '#161b22', '#1f6feb', glow=ac_glow)

    # 3. LINGUISTIC BRANCH (Bottom Middle)
    ling_glow = (1.0 <= phase < 2.2)
    draw_box(230, 95, 240, 165, "BRANCH B: LINGUISTIC", "50 Descriptors (faster-whisper)",
             ["* Verbatim ASR Transcription", "* NLTK POS Tags: Nouns/Verbs/Adj", "* Subordination & Clause Structure", "* Readability: Flesch-Kincaid, Fog", "* Type-Token Ratio & Lexical Rich"],
             '#161b22', '#238636', glow=ling_glow)

    # 4. FUSED MATRIX BOX
    fuse_glow = (1.8 <= phase < 2.8)
    draw_box(520, 195, 175, 170, "MULTIMODAL FUSION", "268-Feature Matrix",
             ["- Concatenated Vector:", "  * 218 Acoustic Features", "  * 34 Syntactic & Readability", "  * 16 Latent TF-IDF SVD", "- Stratified 10-Fold CV", "- Zero-Leakage Validation"],
             '#161b22', '#a371f7', glow=fuse_glow)

    # 5. SUPER-ENSEMBLE BOX
    ens_glow = (2.6 <= phase < 3.6)
    draw_box(745, 230, 230, 220, "10-FOLD SUPER-ENSEMBLE", "SLSQP Quadratic Blending",
             ["- CatBoost (33.9%): Oblivious Trees", "- XGBoost (24.2%): Depth-wise", "- Ridge + RobustScale (22.8%): Prior", "- LightGBM (19.1%): Leaf-wise", "- Noise Gate: Flatness > 0.40 -> 0.0", "- Validation Pearson r: 0.8424", "- Leaderboard Loss: 0.1576"],
             '#161b22', '#d29922', glow=ens_glow)

    # 6. OUTPUT BADGE
    out_glow = (phase >= 3.2 or phase < 0.2)
    out_box = patches.FancyBboxPatch((745, 95), 230, 105, boxstyle="round,pad=6",
                                     linewidth=2.0, edgecolor='#3fb950', facecolor='#161b22')
    ax.add_patch(out_box)
    if out_glow:
        glow_out = patches.FancyBboxPatch((742, 92), 236, 111, boxstyle="round,pad=8",
                                          linewidth=2.5, edgecolor='#3fb950', facecolor='none', alpha=0.7)
        ax.add_patch(glow_out)
    ax.text(860, 175, "CONTINUOUS SCORE", fontsize=11, fontweight='bold', color='#ffffff', ha='center')
    ax.text(860, 153, "MOS Likert Rubric [0.0 - 5.0]", fontsize=8.5, color='#8b949e', ha='center')
    ax.text(860, 128, "Grammar Proficiency: 4.187", fontsize=11.5, fontweight='bold', color='#3fb950', ha='center')
    ax.text(860, 108, "submission.csv (216 Test Audio Files)", fontsize=8, color='#c9d1d9', ha='center')

    # Draw Connecting Animated Signal Arrows & Pulses
    def draw_signal_arrow(p1, p2, color, active_window):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8, mutation_scale=14))
        # Animated traveling energy pulse
        if active_window[0] <= phase <= active_window[1]:
            rel_t = (phase - active_window[0]) / (active_window[1] - active_window[0])
            pulse_x = p1[0] + rel_t * (p2[0] - p1[0])
            pulse_y = p1[1] + rel_t * (p2[1] - p1[1])
            ax.plot(pulse_x, pulse_y, 'o', color='#ffffff', markersize=6, alpha=0.9)
            ax.plot(pulse_x, pulse_y, 'o', color=color, markersize=11, alpha=0.5)

    # Input -> Branches
    draw_signal_arrow((175, 300), (230, 360), '#58a6ff', (0.2, 1.2))
    draw_signal_arrow((175, 260), (230, 200), '#3fb950', (0.4, 1.4))

    # Branches -> Fusion
    draw_signal_arrow((470, 360), (520, 300), '#58a6ff', (1.4, 2.2))
    draw_signal_arrow((470, 200), (520, 260), '#3fb950', (1.6, 2.4))

    # Fusion -> Ensemble
    draw_signal_arrow((695, 280), (745, 340), '#a371f7', (2.2, 3.0))

    # Ensemble -> Output
    draw_signal_arrow((860, 230), (860, 200), '#d29922', (3.0, 3.8))

    # Metric Badges along bottom
    ax.text(100, 35, "Training Samples: 769", fontsize=9, color='#8b949e', ha='center')
    ax.text(350, 35, "Audio + Text Multimodal Fusion", fontsize=9, color='#8b949e', ha='center')
    ax.text(610, 35, "Validation Loss: 0.1576 (Rank 1: 0.3064)", fontsize=9, fontweight='bold', color='#38bdf8', ha='center')
    ax.text(860, 35, "Compulsory Train RMSE: 0.2022", fontsize=9, fontweight='bold', color='#34d399', ha='center')

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0d1117', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    frames.append(img)

# Save as optimized GIF
gif_path = 'pipeline_architecture.gif'
frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=90, loop=0, optimize=True)
print(f"Successfully generated animated diagram: {gif_path} ({len(frames)} frames)")
