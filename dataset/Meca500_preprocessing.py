import os
import pandas as pd

view = 'front'
cams = 'leftcam'

# 사전에 계산됨
Tvec = [0, -0.01, 0.75]
Rvec = [96, 98,-45]
summary = []
summary.append([view, cams, *Tvec, *Rvec])
columns = ["view", "cam", "tvec_x", "tvec_y", "tvec_z", "rvec_x", "rvec_y", "rvec_z"]
df = pd.DataFrame(summary, columns=columns)

output_dir = "./Meca500"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "aruco_pose_summary.json")
df.to_json(output_path, orient="records", indent=2)