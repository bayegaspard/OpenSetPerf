import pandas as pd
from torch import randint

LEFTOUT=randint(low = 0,high = 61,size = (1,)).item()
print(f"sample {LEFTOUT} is leftout")


text = pd.DataFrame(columns=["place","type"])

for sample in range(62):
    if sample!=LEFTOUT:
        for x in range(1016):
            text.loc[len(text.index)] = [f"English/Fnt/Sample{sample+1:03d}/img{sample+1:03d}-{x+1:05d}.png",sample]
        print("Sample "+str(sample)+" is done")


text.to_csv("BasicUnknowns/list.txt", index=False, header=False)

text = pd.DataFrame(columns=["place","type"])

for x in range(1016):
    text.loc[len(text.index)] = [f"English/Fnt/Sample{LEFTOUT+1:03d}/img{LEFTOUT+1:03d}-{x+1:05d}.png",LEFTOUT]
print("Leftout is done")

text.to_csv("BasicUnknowns/list2.txt", index=False, header=False)