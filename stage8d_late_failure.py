
import pandas as pd

path = "results/stage8/common_frame_comparison.csv"
df = pd.read_csv(path)

print("Available columns:")
print(df.columns.tolist())

print("\nLate-frame comparison (frame >= 260):")
late = df[df["frame"] >= 260]

with pd.option_context(
    "display.max_rows", None,
    "display.max_columns", None,
    "display.width", 220
):
    print(late.to_string(index=False))
