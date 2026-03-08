import os
import argparse
from src.config import HUMAN_PKL_PATH, BASE_RESULTS_DIR, REFERENCE_PKL_DIR
from analysis.plot_bars import plot_comparison_bars
from analysis.plot_scatter import plot_scatter_analysis
from analysis.plot_rmse_corr import plot_combined_metrics
from analysis.plot_rsa import plot_rsa_analysis
from src.prepare_everystep import run_prepare_everystep
from analysis.plot_everystep import run_plot_everystep

def resolve_pkl_path(name, condition, mode):
    """이름(model/human/btom)에 따라 정확한 Pickle 경로를 반환하는 헬퍼 함수"""
    name_lower = name.lower()
    if name_lower == "human":
        return HUMAN_PKL_PATH
    elif name_lower in ["btom", "truebelief", "nocost", "motionheuristic"]:
        return os.path.join(REFERENCE_PKL_DIR, name_lower, f"{name_lower}_data.pkl")
    else:
        if mode == "everystep":
            return os.path.join(BASE_RESULTS_DIR, name_lower, condition, "everystep", "model_data.pkl")
        else:
            return os.path.join(BASE_RESULTS_DIR, name_lower, condition, "model_data.pkl")

def get_base_dir(name, condition, mode):
    """
    Target 모델이 btom이나 human일 경우 condition 폴더 없이 최상위에 저장합니다.
    """
    name_lower = name.lower()
    if name_lower in ["human", "btom", "truebelief", "nocost", "motionheuristic"]:
        return os.path.join(BASE_RESULTS_DIR, name_lower)
    else:
        if mode == "everystep":
            return os.path.join(BASE_RESULTS_DIR, name_lower, condition, "everystep")
        else:
            return os.path.join(BASE_RESULTS_DIR, name_lower, condition)

def run_analysis(model_name, condition, mode, baseline="human", analysis_type="all"):
    print(f"🚀 분석 시작: Target=[{model_name}], Baseline=[{baseline.upper()}]")
    if model_name.lower() not in ["human", "btom"]:
        print(f"   -> Cond=[{condition}], Mode=[{mode}], Type=[{analysis_type.upper()}]")
    
    # 1. 경로 자동 설정
    target_pkl = resolve_pkl_path(model_name, condition, mode)
    baseline_pkl = resolve_pkl_path(baseline, condition, mode)
    
    if not os.path.exists(target_pkl):
        print(f"❌ Error: Target data not found at {target_pkl}. 변환을 먼저 수행하세요.")
        return
    if not os.path.exists(baseline_pkl):
        print(f"❌ Error: Baseline data not found at {baseline_pkl}. 변환을 먼저 수행하세요.")
        return

    # 동적으로 계산된 저장 폴더
    base_dir = get_base_dir(model_name, condition, mode)
    os.makedirs(base_dir, exist_ok=True)

    # 2. 분석 함수 호출
    # (A) Bar Plots
    if analysis_type in ["all", "bar"]:
        print(f"\n--- [1] Drawing Comparison Bar Plot ({model_name} vs {baseline.upper()}) ---")
        bar_img = os.path.join(base_dir, f"comparison_bars_vs_{baseline}.png")
        plot_comparison_bars(target_pkl, bar_img, model_name=f"{model_name} ({condition}, {mode})")
    
    # (B) Scatter Correlation Plots
    if analysis_type in ["all", "scatter"]:
        print(f"\n--- [2] Drawing Scatter Plot ({model_name} vs {baseline.upper()}) ---")
        scatter_img = os.path.join(base_dir, f"scatter_plot_vs_{baseline}.png")
        plot_scatter_analysis(
            x_pkl_path=target_pkl, 
            y_pkl_path=baseline_pkl, 
            output_img_path=scatter_img, 
            x_name=f"{model_name} ({condition}, {mode})", 
            y_name=f"{baseline.upper()}"
        )
        
    # (C) RMSE & Correlation Plots
    if analysis_type in ["all", "rmse"]:
        print(f"\n--- [3] Drawing Performance Metrics ({model_name} vs {baseline.upper()}) ---")
        perf_img = os.path.join(base_dir, f"summary_performance_vs_{baseline}.png")
        plot_combined_metrics(target_pkl, perf_img, model_name=f"{model_name} ({condition}, {mode})")

    # (D) RSA Plots
    if analysis_type in ["all", "rsa"]:
        print(f"\n--- [4] Drawing RSA Heatmaps ({model_name} vs {baseline.upper()}) ---")
        rsa_img = os.path.join(base_dir, f"rsa_vs_{baseline}.png")
        plot_rsa_analysis(
            x_pkl_path=target_pkl,
            y_pkl_path=baseline_pkl,
            output_img_path=rsa_img,
            x_name=f"{model_name} ({condition}, {mode})",
            y_name=f"{baseline.upper()}"
        )

    # (E) Phase Plots
    if analysis_type in ["all", "phase"]:
        if mode == "everystep":
            print("\n--- [5] Drawing Phase Trajectory Plots (Everystep Only) ---")
            
            # 1. 데이터 전처리 및 완벽한 피험자(Perfect Subjects) 동적 추출
            selected_subjects = run_prepare_everystep(model_name, condition)
            
            if not selected_subjects:
                print("❌ Error: 선별된 피험자가 없습니다. Phase Plot을 그릴 수 없습니다.")
            else:
                print(f"  -> Drawing Everystep Plots for Selected Subjects: {selected_subjects}")
                run_plot_everystep(model_name, condition, selected_subjects)
                
        else:
            # 사용자가 normal 모드인데 강제로 --type phase를 요청한 경우 방어
            if analysis_type == "phase":
                print("\n❌ Warning: 'phase' 분석은 '--mode everystep'에서만 가능합니다. 건너뜁니다.")

    print("\n✨ 모든 분석 및 시각화 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model 인자는 그대로 두되, btom이나 human일 때는 뒤의 condition을 굳이 치지 않아도 됩니다.
    parser.add_argument("--model", type=str, required=True, help="Target model (e.g., gpt-4o, btom, human)")
    parser.add_argument("--baseline", type=str, default="human", 
                        choices=["human", "btom", "truebelief", "nocost", "motionheuristic"], help="Baseline to compare against (default: human)")
    
    # condition의 default를 세팅해두면 굳이 타이핑 안 해도 에러가 나지 않습니다.
    parser.add_argument("--condition", type=str, default="vanilla", 
                        choices=["vanilla", "reasoning", "oneshot"],
                        help="Experiment condition")
    parser.add_argument("--mode", type=str, default="normal", 
                        choices=["normal", "everystep"], 
                        help="Experiment option")
    parser.add_argument("--type", type=str, default="all",
                        choices=["all", "bar", "scatter", "rmse", "rsa", "phase"],
                        help="Specific analysis to run (default: all)")
    args = parser.parse_args()
    
    run_analysis(args.model, args.condition, args.mode, args.baseline, args.type)