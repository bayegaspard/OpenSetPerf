## OpenSetPerf
A Practitioner's Guide to the Performance of Deep-Learning Based Open Set Recognition Algorithms for Network Intrusion Detection Systems

### Steps to run:

- Clone the repository using :

`git clone https://github.com/bayegaspard/OpenSetPerf.git`
- Download the [Payload-Byte NIDS Dataset](https://github.com/Yasir-ali-farrukh/Payload-Byte/tree/main/Data) 
- Navigate to the root folder and place the downloaded CSV file in the `dataset` folder. New structure will be `dataset\Payload_data_CICIDS2017.csv`
- If you don't have pip3 installed, you can use the command below to install one.

`sudo apt-get install python3-pip
`
- Navigate to the `src` directory using the command `cd OpenSetPerf\src` directory.
- It is recommended to perform this test in a virtual environment. This step is optional.
```
pip3 install virtualenv
virtualenv opensetperf
source opensetper/bin/activate
```
- Install required packages using the command below:
`pip3 install -r requirements.txt
`

##### Items in `src` folder: 

- `requirements.txt`
  - File containing the version numbers of the required external libraries
- `OpenMax`
  - This is an implementation of OpenMax
  - It has a few extra display files because I wanted to see what happens when other varables change.

- `EnergyOOD`
  - This is an implementation of Energy based OOD
  - It has a file EvaluationDisplay that is a modified version of Evaluation to display things easier.

- `datasets`
  - We place the NIDS dataset in this folder.

