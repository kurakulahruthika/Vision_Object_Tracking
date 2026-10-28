import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def create_architecture_diagram(output_path):
    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Background canvas
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Title Banner
    title_box = patches.FancyBboxPatch(
        (0.8, 11.0), 14.4, 0.8, boxstyle="round,pad=0.1",
        facecolor="#1e3a8a", edgecolor="none", zorder=2
    )
    ax.add_patch(title_box)
    ax.text(8.0, 11.4, "SCALE-ADAPTIVE HYBRID OBJECT TRACKING & MOTION ANALYSIS ARCHITECTURE",
            color="white", fontsize=15, fontweight="bold", ha="center", va="center")

    # Helper function for drawing rounded process blocks
    def draw_block(x, y, w, h, title, subtitle, color, text_color="white", edge_color="none", lw=1):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.12",
            facecolor=color, edgecolor=edge_color, linewidth=lw, zorder=3
        )
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w / 2, y + h * 0.62, title, color=text_color, fontsize=10, fontweight="bold", ha="center", va="center", zorder=4)
            ax.text(x + w / 2, y + h * 0.32, subtitle, color=text_color, fontsize=8.5, ha="center", va="center", zorder=4)
        else:
            ax.text(x + w / 2, y + h * 0.5, title, color=text_color, fontsize=10, fontweight="bold", ha="center", va="center", zorder=4)

    # 1. Input & Initialization Phase
    draw_block(0.8, 9.6, 3.2, 0.9, "Input Video / Webcam", "cv2.VideoCapture (File / Live)", "#0f766e")
    draw_block(4.4, 9.6, 3.4, 0.9, "Interactive Target ROI", "User Mouse Box [x, y, w, h]", "#0f766e")
    draw_block(8.2, 9.6, 3.6, 0.9, "Shi-Tomasi Feature Seeding", "min(lambda_1, lambda_2) > thresh", "#0369a1")
    draw_block(12.2, 9.6, 3.0, 0.9, "Appearance Extraction", "Reference 2D HSV Histogram", "#0369a1")

    # Connector Arrows Row 1
    arrow_props = dict(arrowstyle="->", color="#334155", lw=2, mutation_scale=15)
    ax.annotate("", xy=(4.4, 10.05), xytext=(4.0, 10.05), arrowprops=arrow_props)
    ax.annotate("", xy=(8.2, 10.05), xytext=(7.8, 10.05), arrowprops=arrow_props)
    ax.annotate("", xy=(12.2, 10.05), xytext=(11.8, 10.05), arrowprops=arrow_props)

    # Split down arrows
    ax.annotate("", xy=(2.6, 8.6), xytext=(2.6, 9.6), arrowprops=arrow_props)
    ax.annotate("", xy=(10.0, 8.6), xytext=(10.0, 9.6), arrowprops=arrow_props)

    # 2. Parallel Processing Columns (Kalman vs Optical Flow)
    # Left: Kalman Column
    draw_block(0.8, 7.5, 3.6, 1.1, "Discrete Kalman Filter", "Constant Velocity Model\nState: [x, y, vx, vy]^T", "#4338ca")
    draw_block(0.8, 5.9, 3.6, 1.1, "Boundary-Aware Damping", "If near edge: vx <= 0\nPrevents out-of-bounds drift", "#6366f1")

    # Right: Optical Flow Column
    draw_block(7.8, 7.5, 4.4, 1.1, "Bidirectional Pyramidal LK Flow", "Forward Step: Frame t -> t+1\nBackward Step: Frame t+1 -> t", "#0284c7")
    draw_block(7.8, 5.9, 4.4, 1.1, "Forward-Backward Validation", "Filter: ||P_orig - P_back|| < 1.5 px\nDiscards non-rigid aperture drift", "#0284c7")

    ax.annotate("", xy=(2.6, 7.0), xytext=(2.6, 7.5), arrowprops=arrow_props)
    ax.annotate("", xy=(10.0, 7.0), xytext=(10.0, 7.5), arrowprops=arrow_props)

    # Connecting Flow to RANSAC
    draw_block(7.8, 4.3, 4.4, 1.1, "RANSAC 2D Affine Consensus", "Fits partial affine matrix M\nIsolates rigid target inliers", "#0d9488")
    ax.annotate("", xy=(10.0, 5.4), xytext=(10.0, 5.9), arrowprops=arrow_props)

    # 3. Dynamic Scale Adaptation (Center Node)
    draw_block(7.8, 2.7, 4.4, 1.1, "Dynamic Affine Scale Morphing", "Transforms 4 corners via M\nScale Clamping: [0.70x, 1.40x]", "#b45309")
    ax.annotate("", xy=(10.0, 3.8), xytext=(10.0, 4.3), arrowprops=arrow_props)

    # 4. Gating & Decision Engine
    draw_block(4.4, 1.2, 7.2, 1.1, "Multi-Factor Gating & Decision Engine", "Confidence >= 0.42  OR  Strong Geometry (Inliers >= 8 & FB < 0.9px)\nOverrides Kalman lag during sharp maneuvers & edge U-turns", "#be123c")

    # Flow arrows entering Decision Engine
    ax.annotate("", xy=(6.0, 2.3), xytext=(2.6, 5.9),
                arrowprops=dict(arrowstyle="->", color="#4338ca", lw=2, connectionstyle="arc3,rad=-0.15", mutation_scale=15))
    ax.annotate("", xy=(8.0, 2.3), xytext=(10.0, 2.7),
                arrowprops=dict(arrowstyle="->", color="#b45309", lw=2, connectionstyle="arc3,rad=0.15", mutation_scale=15))

    # 5. Two Outcomes: Tracking vs Recovery
    draw_block(0.8, 2.7, 3.2, 1.0, "[ACCEPTED: TRACKING]", "Kalman Correct (Update State)\nRetain RANSAC inliers", "#15803d")
    draw_block(12.0, 2.7, 3.2, 1.0, "[REJECTED: RECOVERY]", "Fall back to Kalman Predict\nAdaptive small-target re-seed", "#b91c1c")

    ax.annotate("", xy=(2.4, 2.7), xytext=(4.4, 1.75),
                arrowprops=dict(arrowstyle="->", color="#15803d", lw=2.5, connectionstyle="arc3,rad=-0.2", mutation_scale=15))
    ax.annotate("", xy=(13.6, 2.7), xytext=(11.6, 1.75),
                arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=2.5, connectionstyle="arc3,rad=0.2", mutation_scale=15))

    # 6. Outputs & Telemetry (Bottom Bar)
    draw_block(0.8, 0.1, 14.4, 0.75, "Real-Time Telemetry & Visual Output", "Speed in km/h  |  Speed-Colored Fading Trail  |  Per-Frame CSV Logging  |  30+ FPS Pure CPU", "#334155")
    ax.annotate("", xy=(2.4, 0.85), xytext=(2.4, 2.7), arrowprops=arrow_props)
    ax.annotate("", xy=(13.6, 0.85), xytext=(13.6, 2.7), arrowprops=arrow_props)

    # Boundary boxes for visual grouping
    k_group = patches.FancyBboxPatch((0.5, 5.6), 4.2, 3.3, boxstyle="round,pad=0.1",
                                    facecolor="none", edgecolor="#818cf8", linestyle="--", linewidth=1.5)
    ax.add_patch(k_group)
    ax.text(0.7, 8.7, "Kinematic Modeling", color="#4338ca", fontsize=9, fontweight="bold")

    v_group = patches.FancyBboxPatch((7.5, 2.4), 5.0, 6.5, boxstyle="round,pad=0.1",
                                    facecolor="none", edgecolor="#38bdf8", linestyle="--", linewidth=1.5)
    ax.add_patch(v_group)
    ax.text(7.7, 8.7, "Geometric Visual Pipeline", color="#0284c7", fontsize=9, fontweight="bold")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Architecture diagram successfully saved to: {output_path}")

if __name__ == "__main__":
    out1 = "C:/Users/KALYAN/.gemini/antigravity/scratch/Vision_Object_Tracking/results/system_architecture.png"
    out2 = "C:/Users/KALYAN/.gemini/antigravity/brain/9b0599f9-607e-49fd-8f2d-795bf190a6bb/system_architecture.png"
    create_architecture_diagram(out1)
    create_architecture_diagram(out2)
