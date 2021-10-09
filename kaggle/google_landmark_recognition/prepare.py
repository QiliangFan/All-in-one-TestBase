import pandas as pd

submission_csv = "../input/landmark-recognition-2021/sample_submission.csv"
train_csv = "../input/landmark-recognition-2021/train.csv"

submission = pd.read_csv(submission_csv)
train = pd.read_csv(train_csv)

print(f"submission: {submission.columns.values}")
print(f"train: {train.columns.values}")


from glob import glob
import os

train_root = "../input/landmark-recognition-2021/train"
test_root = "../input/landmark-recognition-2021/test"


train_files = []
train_labels = []

for idx, (_id, landmark_id) in train.iterrows():
    img = glob(os.path.join(train_root, "**", f"{_id}.jpg"), recursive=True)[0]
    train_files.append(img)
    train_labels.append(landmark_id)

test_files = glob(os.path.join(test_root, "**", f"*.jpg"), recursive=True)