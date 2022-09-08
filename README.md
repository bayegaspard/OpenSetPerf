## OpenSetPerf
A Practitioner's Guide to the Performance of Deep-Learning Based Open Set Recognition Algorithms for Network Intrusion Detection Systems

### Steps to run:

- Clone the repository using :

`git clone https://github.com/bayegaspard/OpenSetPerf.git`
- Download the [Payload-Byte NIDS Dataset](https://github.com/Yasir-ali-farrukh/Payload-Byte/tree/main/Data) 
- Navigate to the root folder and place the downloaded CSV file in the `dataset` folder. New structure will be `dataset\Payload_data_CICIDS2017.csv`
##### Note: If you don't have pip3 installed, you can use the command below to install one.

`sudo apt-get install python3-pip
`
- Navigate to the `src` directory using the command `cd OpenSetPerf\src` directory.
- Install required packages using the command below:
`pip3 install -r requirements.txt
`
- Choose either `BasicCNN`, `BasicUnknowns`, `OpenMax`, `combined` or `EnergyOOD`
- Within the folder in `OpenSetPerf\src`, run the main program using the command: `python3 `"step_6_folder_name"`/main.py`

##### Items in `src` folder: 

- `requirements.txt`
  - File containing the version numbers of the required external libraries

- `BasicCNN`
  - This is the CNN all the rest are based off of, it does not have any unknowns or unknown evaluation.

- `BasicUnknowns`
  - This is a version of the CNN that is being passed unknowns and gives a report about them.
  - It uses softmax

- `OpenMax`
  - This is an implementation of OpenMax
  - It has a few extra display files because I wanted to see what happens when other varables change.

- `EnergyOOD`
  - This is an implementation of Energy based OOD
  - It has a file EvaluationDisplay that is a modified version of Evaluation to display things easier.

- `ODIN`
  - Implements ODIN
  - Also has a legacy EvaulationDisplay

- `Combined`
  - Implements SoftMax, OpenMax, EnergyOOD, and ODIN.
  - Main will save CSVs of results in the parent folder.
- `Plots`
  - Contains the `spider_plots.py` which contain code to draw a spider plot for qualitative algorithmic evaluation.
- `datasets`
  - We place the NIDS dataset in this folder.

