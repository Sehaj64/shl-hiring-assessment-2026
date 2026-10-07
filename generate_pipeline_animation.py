import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

# High-resolution 16:9 canvas
width, height = 1200, 640
dpi = 100
n_frames = 48

frames = []

for frame_idx in range(n_frames):
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 640)
    ax.axis('off')

    t = frame_idx / n_frames
    phase = (t * 4) % 4  # 0 to 4 cycle

    # --- Header Title ---
    ax.text(600, 605, "HOW THE SPOKEN GRAMMAR SCORING ENGINE WORKS",
            fontsize=17, fontweight='bold', color='#ffffff', ha='center', va='center')
    ax.text(600, 580, "A Simple 4-Stage Architecture: Listen -> Understand -> Ensemble -> Score",
            fontsize=11, color='#94a3b8', ha='center', va='center')

    # Helper function for card boxes
    def draw_card(x, y, w, h, title, category, badge, bullets, border_col, bg_col, is_active=False):
        if is_active:
            glow = patches.FancyBboxPatch((x-4, y-4), w+8, h+8, boxstyle="round,pad=10",
                                          linewidth=2.5, edgecolor=border_col, facecolor='none', alpha=0.6)
            ax.add_patch(glow)
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=8",
                                      linewidth=1.8, edgecolor=border_col, facecolor=bg_col)
        ax.add_patch(card)
        
        # Category badge
        ax.text(x + 18, y + h - 22, category.upper(), fontsize=8.5, fontweight='bold', color=border_col)
        # Title
        ax.text(x + 18, y + h - 45, title, fontsize=13, fontweight='bold', color='#ffffff')
        # Sub-badge pill
        pill = patches.FancyBboxPatch((x + w - 110, y + h - 35), 95, 20, boxstyle="round,pad=3",
                                      linewidth=0, facecolor=border_col, alpha=0.25)
        ax.add_patch(pill)
        ax.text(x + w - 62, y + h - 25, badge, fontsize=8, fontweight='bold', color=border_col, ha='center', va='center')

        # Bullets (large, readable)
        y_pos = y + h - 78
        for b in bullets:
            ax.text(x + 18, y_pos, b, fontsize=9.5, color='#cbd5e1', va='center')
            y_pos -= 22

    # -------------------------------------------------------------
    # STAGE 1: RAW INPUT (Left)
    # -------------------------------------------------------------
    s1_active = (phase < 1.0)
    draw_card(40, 220, 210, 260, 
              "Stage 1: Speech", "Input Audio", "45-60s .wav",
              ["* Candidate Interview", "* 16 kHz Mono Audio", "* Voice Waveforms", "* 769 Training Audios", "* 216 Test Audios"],
              '#38bdf8', '#111827', is_active=s1_active)

    # -------------------------------------------------------------
    # STAGE 2: DUAL BRANCHES (The Two Ears)
    # -------------------------------------------------------------
    # Top Branch: HOW THEY SOUND (Acoustic)
    s2a_active = (0.7 <= phase < 2.0)
    draw_card(320, 360, 270, 180,
              "How It Sounds", "Branch A: Acoustic", "218 Features",
              ["* Speaking Pace (WPM)", "* Pause & Hesitation Regularity", "* Pitch (F0) & Intonation", "* Voice Energy & Dynamics"],
              '#60a5fa', '#111827', is_active=s2a_active)

    # Bottom Branch: WHAT THEY SAID (Linguistic)
    s2b_active = (0.9 <= phase < 2.2)
    draw_card(320, 150, 270, 180,
              "What Is Said", "Branch B: Linguistic", "50 Features",
              ["* Verbatim Transcription (Whisper)", "* Sentence Structure & Clauses", "* Grammar & Word Choices (POS)", "* Readability & Grade Level"],
              '#34d399', '#111827', is_active=s2b_active)

    # -------------------------------------------------------------
    # STAGE 3: MULTIMODAL COMBINATION
    # -------------------------------------------------------------
    s3_active = (1.8 <= phase < 2.8)
    draw_card(660, 220, 210, 260,
              "Stage 3: Fusion", "Full Profile", "268 Features",
              ["* Sound + Text Merged", "* 10-Fold Stratified Split", "* No Data Leakage", "* Robust Normalization", "* Complete Picture"],
              '#c084fc', '#111827', is_active=s3_active)

    # -------------------------------------------------------------
    # STAGE 4: MODEL ENSEMBLE & FINAL SCORE (Right)
    # -------------------------------------------------------------
    s4_active = (2.6 <= phase < 3.8)
    draw_card(930, 220, 230, 260,
              "Stage 4: Decision", "Ensemble Panel", "0.0 to 5.0",
              ["* CatBoost (33.9%): Stable trees", "* XGBoost (24.2%): Deep patterns", "* Ridge (22.8%): Linear anchor", "* LightGBM (19.1%): Edge cases", "* Noise Override: Static -> 0.0"],
              '#f59e0b', '#111827', is_active=s4_active)

    # -------------------------------------------------------------
    # RESULT SCORECARD (Bottom Highlight Bar)
    # -------------------------------------------------------------
    res_card = patches.FancyBboxPatch((40, 45), 1120, 75, boxstyle="round,pad=8",
                                      linewidth=1.8, edgecolor='#10b981', facecolor='#111827')
    ax.add_patch(res_card)
    
    ax.text(180, 82, "FINAL PREDICTION", fontsize=11, fontweight='bold', color='#ffffff', ha='center')
    ax.text(180, 62, "Continuous MOS Likert Score", fontsize=9, color='#94a3b8', ha='center')

    ax.text(450, 85, "Validation Pearson (r)", fontsize=9.5, color='#94a3b8', ha='center')
    ax.text(450, 65, "0.8424", fontsize=13, fontweight='bold', color='#38bdf8', ha='center')

    ax.text(680, 85, "Leaderboard Loss (1 - r)", fontsize=9.5, color='#94a3b8', ha='center')
    ax.text(680, 65, "0.1576  (Rank 1: 0.3064)", fontsize=13, fontweight='bold', color='#34d399', ha='center')

    ax.text(950, 85, "Compulsory Train RMSE", fontsize=9.5, color='#94a3b8', ha='center')
    ax.text(950, 65, "0.2022", fontsize=13, fontweight='bold', color='#f59e0b', ha='center')

    # -------------------------------------------------------------
    # SIGNAL FLOW ARROWS & PULSING ENERGY PACKETS
    # -------------------------------------------------------------
    def draw_flow_arrow(p1, p2, color, window):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.2, mutation_scale=16))
        # Animated traveling photon pulse
        if window[0] <= phase <= window[1]:
            rel = (phase - window[0]) / (window[1] - window[0])
            px = p1[0] + rel * (p2[0] - p1[0])
            py = p1[1] + rel * (p2[1] - p1[1])
            ax.plot(px, py, 'o', color='#ffffff', markersize=7)
            ax.plot(px, py, 'o', color=color, markersize=14, alpha=0.45)

    # Input -> Branch A & B
    draw_flow_arrow((250, 380), (320, 440), '#60a5fa', (0.2, 1.1))
    draw_flow_arrow((250, 320), (320, 240), '#34d399', (0.3, 1.2))

    # Branch A & B -> Fusion
    draw_flow_arrow((590, 440), (660, 380), '#60a5fa', (1.3, 2.1))
    draw_flow_arrow((590, 240), (660, 320), '#34d399', (1.4, 2.2))

    # Fusion -> Decision Ensemble
    draw_flow_arrow((870, 350), (930, 350), '#c084fc', (2.2, 3.1))

    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0b0f19', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    frames.append(img)

# Save as optimized GIF
gif_path = 'pipeline_architecture.gif'
frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=85, loop=0, optimize=True)
print(f"Successfully generated clean, intuitive animation: {gif_path} ({len(frames)} frames)")
