with open('C:/Users/KALYAN/.gemini/antigravity/scratch/Vision_Object_Tracking/stage8g_fair_final_eval.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_block = """TRACKERS = {
    "Stage 5": "results/stage5_tracking_data.csv",
    "Original Stage 6A": "results/stage6A_tracking_data.csv",
    "Corrected Stage 6A": "results/stage6A_corrected_tracking_data.csv",
}"""

new_block = """TRACKERS = {
    "Stage 5 (Baseline)": "results/stage5_tracking_data.csv",
    "Stage 6A (Original)": "results/stage6A_tracking_data.csv",
    "Stage 6A (Corrected)": "results/stage6A_corrected_tracking_data.csv",
    "Stage 9A (Transition)": "results/stage9A_transition_aware_data.csv",
    "Stage 10 (Master)": "results/stage10_master_tracking_data.csv",
}"""

code = code.replace(old_block, new_block)
with open('C:/Users/KALYAN/.gemini/antigravity/scratch/Vision_Object_Tracking/stage8g_fair_final_eval.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("stage8g updated successfully!")
