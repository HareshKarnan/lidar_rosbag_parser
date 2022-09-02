import pandas as pd
import sys
import subprocess
#later if time, make path not hard coded and create directory if it doesn't exist
# subprocess.run(["cd","/hdd/luisamao/robodata/Large_Crowds_Data"])
excel_file = "/hdd/luisamao/robodata/SCAND.xlsx"
all_rosbags = pd.read_excel(excel_file, sheet_name=1)
save_path = "/hdd/luisamao/robodata/Large_Crowds_Data"
robot = "Spot"
tag = "Navigating Through Large Crowds" #if None else sys.argv[1]
end = len(all_rosbags)
count = 0
for i in range(0,end):
    if str(all_rosbags["Tags"][i]).find(tag) != -1 and str(all_rosbags["Robot"][i])==robot:
        count+=1
        # print(all_rosbags["Link to rosbag"][i])
        # print(str(all_rosbags["Tags"][i]))
        subprocess.run(["wget", "-O", count, str(all_rosbags["Link to rosbag"][i]), "-P", save_path])
