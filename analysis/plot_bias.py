import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.config import BASE_RESULTS_DIR

def plot_cognitive_biases(model_name, condition):
    target_dir = os.path.join(BASE_RESULTS_DIR, model_name, condition, "everystep")
    data_path = os.path.join(target_dir, "bias_indicators.csv")
    
    if not os.path.exists(data_path):
        print("❌ Error: bias_indicators.csv 파일이 없습니다. eval_indicators.py를 먼저 실행하세요.")
        return
        
    df = pd.read_csv(data_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sns.set_theme(style="whitegrid")
    
    # -----------------------------------------------------------------
    # 📊 1. TrueBelief (시야 한계 무시) - Group별
    # -----------------------------------------------------------------
    sns.barplot(data=df, x='group_desc', y='tb_delta', ax=axes[0], color='#E63946', errorbar='ci')
    axes[0].axhline(0, color='black', linewidth=1.5)
    axes[0].set_title('TrueBelief Bias at t=1\n(Δ Belief of Occluded Target)', fontweight='bold')
    axes[0].set_ylabel('Overestimation (+) / Underestimation (-)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # -----------------------------------------------------------------
    # 📊 2. MotionHeuristic (근접성 편향) - Group별
    # -----------------------------------------------------------------
    sns.barplot(data=df, x='group_desc', y='mh_delta', ax=axes[1], color='#457B9D', errorbar='ci')
    axes[1].axhline(0, color='black', linewidth=1.5)
    axes[1].set_title('Motion Heuristic Bias at t=1\n(Δ Absolute Desire Difference)', fontweight='bold')
    axes[1].set_ylabel('')
    axes[1].tick_params(axis='x', rotation=45)

    # -----------------------------------------------------------------
    # 📊 3. NoCost (비용-보상 무시) - Path Length 구간별
    # -----------------------------------------------------------------
    # Path length를 3구간(Short, Medium, Long)으로 나누기
    df['length_bin'] = pd.qcut(df['path_length'], q=3, labels=['Short', 'Medium', 'Long'])
    
    sns.barplot(data=df, x='length_bin', y='nc_delta', ax=axes[2], color='#ACCB20', errorbar='ci')
    axes[2].axhline(0, color='black', linewidth=1.5)
    axes[2].set_title('NoCost Bias at Final Step\n(Δ Desire by Path Length)', fontweight='bold')
    axes[2].set_xlabel('Path Length Category')
    axes[2].set_ylabel('')

    plt.suptitle(f"Cognitive Bias Magnitudes: {model_name.upper()} vs BToM", fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(target_dir, "plot_bias_magnitudes.png")
    plt.savefig(out_path, dpi=300)
    print(f"✅ 편향 시각화 완료: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_cognitive_biases("gpt-4o", "vanilla")