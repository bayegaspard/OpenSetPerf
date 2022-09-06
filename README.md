### OpenSetPerf
A Practitioner's Guide to the Performance of Deep-Learning Based Open Set Recognition Algorithms for Network Intrusion Detection Systems
#### Usage
- Use the command below to download the Payload-Byte NIDS dataset.
```
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aP3y7UsF8dS46i6hq7aZaUdmuT9MNobG' -O Payload_Byte_NIDS_Data.csv
```
Original source of dataset can be found here: https://github.com/Yasir-ali-farrukh/Payload-Byte/tree/main/Data
 
- Navigate to the root folder and place the downloaded CSV in the `dataset` folder. New structure will be `dataset\Payload_Byte_NIDS_Data.csv`
- Install required packages using the command below:
```
pip3 install -r requirements.txt
```
## Note: If you don't have pip3 installed, you can use the command below to install one.
sudo apt-get install python3-pip
- Navigate to the root folder and run the following command:
```
python3 main.py
```
