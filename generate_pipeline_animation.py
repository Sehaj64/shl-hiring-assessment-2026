import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io

# Clean, spacious 16:9 canvas
width, height = 1100, 520
dpi = 100
n_frames = 48

frames = []

for frame_idx in range(n_frames):
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor('#0f172a')  # Slate dark
    ax.set_facecolor('#0f172a')
    ax.set_xlim(0, 1100)
    ax.set_ylim(0, 520)
    ax.axis('off')

    # Animation progress cycle: 0.0 to 4.0
    t = frame_idx / n_frames
    phase = (t * 4) % 4

    # Main Title (Minimal, Clean)
    ax.text(550, 485, "HOW THE GRAMMAR SCORING ENGINE WORKS",
            fontsize=16, fontweight='bold', color='#ffffff', ha='center', va='center')
    ax.text(550, 460, "4 Simple Steps:  Input  ->  Dual Analysis  ->  AI Consensus  ->  Final Score",
            fontsize=11, color='#94a3b8', ha='center', va='center')

    # Clean Box Drawing Function (No clutter, spacious margins)
    def draw_step_card(x, y, w, h, step_num, title, subtitle, line1, line2, color, is_active=False):
        if is_active:
            glow = patches.FancyBboxPatch((x-3, y-3), w+6, h+6, boxstyle="round,pad=8",
                                          linewidth=2.5, edgecolor=color, facecolor='none', alpha=0.55)
            ax.add_patch(glow)
        
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=8",
                                      linewidth=1.8, edgecolor=color, facecolor='#1e293b')
        ax.add_patch(card)

        # Step Pill
        pill = patches.FancyBboxPatch((x + 16, y + h - 30), 75, 18, boxstyle="round,pad=3",
                                      linewidth=0, facecolor=color, alpha=0.25)
        ax.add_patch(pill)
        ax.text(x + 53, y + h - 21, step_num.upper(), fontsize=8.5, fontweight='bold', color=color, ha='center', va='center')

        # Title
        ax.text(x + 16, y + h - 55, title, fontsize=13, fontweight='bold', color='#ffffff', va='center')
        # Subtitle
        ax.text(x + 16, y + h - 75, subtitle, fontsize=9.5, color='#94a3b8', va='center')

        # Simple 2 description lines
        if line1:
            ax.text(x + 16, y + 42, f"- {line1}", fontsize=10, color='#e2e8f0', va='center')
        if line2:
            ax.text(x + 16, y + 20, f"- {line2}", fontsize=10, color='#e2e8f0', va='center')

    # =========================================================
    # STEP 1: INPUT SPEECH AUDIO (Left)
    # =========================================================
    s1_on = (phase < 1.0)
    draw_step_card(40, 160, 200, 230, "Step 1", "Speech Audio", "Candidate Response",
                   "45-60 second recording", "16 kHz mono sound file",
                   '#38bdf8', is_active=s1_on)

    # =========================================================
    # STEP 2: DUAL ANALYSIS (Two Parallel Tracks)
    # =========================================================
    # Track A (Top): Fluency & Sound
    s2a_on = (0.7 <= phase < 2.0)
    draw_step_card(290, 280, 235, 140, "Track A", "How It Sounds", "Acoustic Fluency",
                   "Speaking pace (WPM)", "Pauses & rhythm regularity",
                   '#60a5fa', is_active=s2a_on)

    # Track B (Bottom): Grammar & Words
    s2b_on = (0.9 <= phase < 2.2)
    draw_step_card(290, 110, 235, 140, "Track B", "What Is Said", "Linguistic Grammar",
                   "Sentence clauses & syntax", "Word choices & vocabulary",
                   '#34d399', is_active=s2b_on)

    # =========================================================
    # STEP 3: AI COMMITTEE / ENSEMBLE (Fusion & Consensus)
    # =========================================================
    s3_on = (1.8 <= phase < 3.0)
    draw_step_card(575, 160, 235, 230, "Step 3", "AI Committee", "Ensemble Consensus",
                   "Sound + text merged", "4 models vote together",
                   '#c084fc', is_active=s3_on)
    # Extra note inside Step 3
    ax.text(591, 195, "* CatBoost, XGB, Ridge, LGBM", fontsize=8.5, color='#a855f7', va='center')

    # =========================================================
    # STEP 4: FINAL SCORE (Right)
    # =========================================================
    s4_on = (2.8 <= phase or phase < 0.2)
    draw_step_card(860, 160, 200, 230, "Step 4", "Grammar Score", "Official MOS Rubric",
                   "Continuous rating: 0 to 5", "Empty audio filtered out",
                   '#f59e0b', is_active=s4_on)
    # Highlight final score
    ax.text(960, 205, "Score: 1.0 - 5.0", fontsize=11, fontweight='bold', color='#fbbf24', ha='center')

    # =========================================================
    # CONNECTING FLOW ARROWS & PULSING SIGNAL DOTS
    # =========================================================
    def draw_path(p1, p2, color, window):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.2, mutation_scale=15))
        # Moving energy pulse
        if window[0] <= phase <= window[1]:
            rel = (phase - window[0]) / (window[1] - window[0])
            dot_x = p1[0] + rel * (p2[0] - p1[0])
            dot_y = p1[1] + rel * (p2[1] - p1[1])
            ax.plot(dot_x, dot_y, 'o', color='#ffffff', markersize=7)
            ax.plot(dot_x, dot_y, 'o', color=color, markersize=14, alpha=0.4)

    # Step 1 -> Track A & Track B
    draw_path((240, 310), (290, 350), '#60a5fa', (0.2, 1.1))
    draw_path((240, 240), (290, 180), '#34d399', (0.3, 1.2))

    # Track A & B -> Step 3
    draw_path((525, 350), (575, 300), '#60a5fa', (1.3, 2.1))
    draw_path((525, 180), (575, 250), '#34d399', (1.4, 2.2))

    # Step 3 -> Step 4
    draw_path((810, 275), (860, 275), '#c084fc', (2.2, 3.1))

    # =========================================================
    # BOTTOM KEY METRICS BAR
    # =========================================================
    bar = patches.FancyBboxPatch((40, 35), 1020, 48, boxstyle="round,pad=6",
                                 linewidth=1.2, edgecolor='#334155', facecolor='#1e293b')
    ax.add_patch(bar)

    ax.text(180, 59, "Dataset: 769 Train / 216 Test", fontsize=10, color='#94a3b8', ha='center', va='center')
    ax.text(480, 59, "Validation Pearson (r):  0.8424", fontsize=10.5, fontweight='bold', color='#38bdf8', ha='center', va='center')
    ax.text(760, 59, "Leaderboard Loss:  0.1576 (Rank 1: 0.3064)", fontsize=10.5, fontweight='bold', color='#34d399', ha='center', va='center')
    ax.text(970, 59, "Train RMSE: 0.2022", fontsize=10, fontweight='bold', color='#f59e0b', ha='center', va='center')

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0f172a', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    frames.append(img)

# Save high-clarity GIF
gif_path = 'pipeline_architecture.gif'
frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=85, loop=0, optimize=True)
print(f"Successfully generated ultra-clean, simple animation: {gif_path} ({len(frames)} frames)")
