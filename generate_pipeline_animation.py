import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 1420 x 720 canvas for spacious, elegant layout
width, height = 1420, 720
dpi = 100
n_frames = 40  # 40 frames for smooth, lightweight GIF under 800 KB

frames = []

for frame_idx in range(n_frames):
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.patch.set_facecolor('#0b0f19')
    ax.set_facecolor('#0b0f19')
    ax.set_xlim(0, 1420)
    ax.set_ylim(0, 720)
    ax.axis('off')

    # Animation progress cycle: 0.0 to 4.0
    t = frame_idx / n_frames
    phase = (t * 4) % 4

    # Main Header
    ax.text(710, 680, "HOW THE SPOKEN GRAMMAR SCORING ENGINE WORKS",
            fontsize=18, fontweight='bold', color='#ffffff', ha='center', va='center')
    ax.text(710, 650, "Simple 4-Step Architecture:  Input Audio  ->  Dual Analysis  ->  AI Consensus  ->  Final Score",
            fontsize=11, color='#94a3b8', ha='center', va='center')

    # Clean Card Drawing Function
    def draw_card(x, y, w, h, pill_text, title, subtitle, bullets, color, footer=None, is_active=False):
        if is_active:
            glow = patches.FancyBboxPatch((x - 3, y - 3), w + 6, h + 6, boxstyle="round,pad=6",
                                          linewidth=2.5, edgecolor=color, facecolor='none', alpha=0.6)
            ax.add_patch(glow)

        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=6",
                                      linewidth=1.8, edgecolor=color, facecolor='#111827')
        ax.add_patch(card)

        # Pill badge
        pill_w = 70
        pill = patches.FancyBboxPatch((x + 18, y + h - 32), pill_w, 20, boxstyle="round,pad=3",
                                      linewidth=0, facecolor=color, alpha=0.25)
        ax.add_patch(pill)
        ax.text(x + 18 + pill_w / 2, y + h - 22, pill_text.upper(),
                fontsize=8.5, fontweight='bold', color=color, ha='center', va='center')

        # Title & Subtitle
        ax.text(x + 18, y + h - 54, title, fontsize=14, fontweight='bold', color='#ffffff', va='center')
        ax.text(x + 18, y + h - 74, subtitle, fontsize=9.0, color='#94a3b8', va='center')

        # Divider line
        ax.plot([x + 18, x + w - 18], [y + h - 88, y + h - 88], color='#1f2937', lw=1.2)

        # Bullets
        cur_y = y + h - 114
        for b in bullets:
            ax.text(x + 18, cur_y, b, fontsize=9.0, color='#e2e8f0', va='center')
            cur_y -= 26

        # Optional footer box
        if footer:
            fbox = patches.FancyBboxPatch((x + 16, y + 16), w - 32, 34, boxstyle="round,pad=4",
                                          linewidth=1.0, edgecolor='#374151', facecolor='#1e293b', alpha=0.7)
            ax.add_patch(fbox)
            ax.text(x + w / 2, y + 33, footer, fontsize=8.5, fontweight='bold', color=color, ha='center', va='center')

    card_y = 150
    card_h = 430
    col_w = 295

    # 1. Step 1: Input Audio
    s1_on = (phase < 1.0)
    draw_card(40, card_y, col_w, card_h, "Step 1", "Speech Audio", "Candidate Response",
              ["* 16 kHz Mono WAV audio",
               "* 45s - 60s Spoken answers",
               "* 769 Training audio files",
               "* 216 Test evaluation files",
               "* Raw amplitude signals"],
              '#38bdf8', footer="Raw Speech Input", is_active=s1_on)

    # 2. Step 2: Tracks A & B (Dual Parallel Analysis)
    track_h = 200
    track_b_y = card_y
    track_a_y = card_y + track_h + 30

    s2a_on = (0.7 <= phase < 2.0)
    draw_card(385, track_a_y, col_w, track_h, "Track A", "How It Sounds", "218 Prosodic Features",
              ["* Speaking rate & pacing",
               "* Speech pauses & silence",
               "* Pitch (F0) & intonation"],
              '#60a5fa', is_active=s2a_on)

    s2b_on = (0.9 <= phase < 2.2)
    draw_card(385, track_b_y, col_w, track_h, "Track B", "What Is Said", "50 Syntactic Features",
              ["* Whisper ASR transcript",
               "* Clause & sentence syntax",
               "* POS tags & grammar rules"],
              '#34d399', is_active=s2b_on)

    # 3. Step 3: AI Committee Ensemble
    s3_on = (1.8 <= phase < 3.0)
    draw_card(730, card_y, col_w, card_h, "Step 3", "AI Committee", "Multi-Model Consensus",
              ["* 268 Audio & Text features",
               "* CatBoost decision trees",
               "* XGBoost gradient trees",
               "* Ridge anchor baseline",
               "* LightGBM tree ensemble",
               "* 10-Fold Stratified CV"],
              '#c084fc', footer="SLSQP Quadratic Blending", is_active=s3_on)

    # 4. Step 4: Final Score
    s4_on = (2.8 <= phase or phase < 0.2)
    draw_card(1075, card_y, col_w, card_h, "Step 4", "Grammar Score", "Official MOS Likert Rubric",
              ["* Continuous MOS: 0.0 - 5.0",
               "* Calibrated Ridge blend",
               "* Silent audio filter (0.0)",
               "* Benchmark: r = 0.8424",
               "* 216 Test Predictions"],
              '#f59e0b', footer="Automated Scoring Engine", is_active=s4_on)

    # Connecting Flow Arrows and Pulsing Energy Dots
    def draw_signal(p1, p2, color, window):
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="->", color=color, lw=2.4, mutation_scale=15))
        if window[0] <= phase <= window[1]:
            raw_rel = (phase - window[0]) / (window[1] - window[0])
            rel = 0.15 + 0.70 * raw_rel  # Clamped strictly in gap between boxes
            dot_x = p1[0] + rel * (p2[0] - p1[0])
            dot_y = p1[1] + rel * (p2[1] - p1[1])
            ax.plot(dot_x, dot_y, 'o', color='#ffffff', markersize=6)
            ax.plot(dot_x, dot_y, 'o', color=color, markersize=13, alpha=0.45)

    # Step 1 -> Track A & Track B
    draw_signal((340, track_a_y + track_h / 2), (380, track_a_y + track_h / 2), '#60a5fa', (0.2, 1.1))
    draw_signal((340, track_b_y + track_h / 2), (380, track_b_y + track_h / 2), '#34d399', (0.3, 1.2))

    # Tracks A & B -> Step 3
    draw_signal((685, track_a_y + track_h / 2), (725, card_y + card_h * 0.65), '#60a5fa', (1.2, 2.1))
    draw_signal((685, track_b_y + track_h / 2), (725, card_y + card_h * 0.35), '#34d399', (1.3, 2.2))

    # Step 3 -> Step 4
    draw_signal((1030, card_y + card_h / 2), (1070, card_y + card_h / 2), '#c084fc', (2.2, 3.1))

    # Bottom Key Metrics Bar
    bottom_bar = patches.FancyBboxPatch((40, 25), 1330, 85, boxstyle="round,pad=6",
                                        linewidth=1.4, edgecolor='#374151', facecolor='#111827')
    ax.add_patch(bottom_bar)

    col_centers = [205, 540, 875, 1205]

    # Col 1: Dataset
    ax.text(col_centers[0], 84, "BENCHMARK DATASET", fontsize=8.5, fontweight='bold', color='#9ca3af', ha='center')
    ax.text(col_centers[0], 62, "769 Train  |  216 Test", fontsize=12, fontweight='bold', color='#f3f4f6', ha='center')
    ax.text(col_centers[0], 42, "Standardized 16 kHz Mono WAV", fontsize=8, color='#6b7280', ha='center')

    # Col 2: Correlation
    ax.text(col_centers[1], 84, "VALIDATION CORRELATION (r)", fontsize=8.5, fontweight='bold', color='#38bdf8', ha='center')
    ax.text(col_centers[1], 62, "0.8424", fontsize=14, fontweight='bold', color='#38bdf8', ha='center')
    ax.text(col_centers[1], 42, "Primary Competition Metric", fontsize=8, color='#6b7280', ha='center')

    # Col 3: Leaderboard Loss
    ax.text(col_centers[2], 84, "LEADERBOARD LOSS (1 - r)", fontsize=8.5, fontweight='bold', color='#34d399', ha='center')
    ax.text(col_centers[2], 62, "0.1576", fontsize=14, fontweight='bold', color='#34d399', ha='center')
    ax.text(col_centers[2], 42, "Beats Rank 1 (0.3064) by ~49%", fontsize=8, color='#34d399', ha='center')

    # Col 4: Train RMSE
    ax.text(col_centers[3], 84, "COMPULSORY TRAIN RMSE", fontsize=8.5, fontweight='bold', color='#f59e0b', ha='center')
    ax.text(col_centers[3], 62, "0.2022", fontsize=14, fontweight='bold', color='#f59e0b', ha='center')
    ax.text(col_centers[3], 42, "Mandatory Notebook Metric Met", fontsize=8, color='#6b7280', ha='center')

    # Save frame to buffer without bbox_inches='tight' so layout is perfectly fixed
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#0b0f19')
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf))

# Save optimized GIF
gif_path = 'pipeline_architecture.gif'
frames[0].save(gif_path, format='GIF', append_images=frames[1:], save_all=True, duration=90, loop=0, optimize=True)
print(f"Successfully generated clean animation without overlap: {gif_path} ({len(frames)} frames)")
