import os
path = '/home/yash/Desktop/TENSORBROS/New/Train/Cookery'
files = os.listdir(path)
i = 1

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, str(i)+'.jpg'))
    i = i+1
