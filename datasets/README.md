## Dataset taken from : https://www.unb.ca/cic/datasets/ids-2017.html
- link to dataset without registering or filling the form : http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/
- Download the zip file from the dataset folder using the follwoing command:
```
wget http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/MachineLearningCSV.zip
```
- Unzip the `GeneratedLabelledFlows.zip` and place  `*.csv` files in the `datasets` folder.

- Dataloader.py will generate one of two folders. Clustered or normal. The clustered folder keeps its own counts.csv while the normal folder just generates a counts.csv in the datasets directory.