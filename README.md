## OpenSetPerf
A Practitioner's Guide to the Performance of Deep-Learning Based Open Set Recognition Algorithms for Network Intrusion Detection Systems

### Steps to run:

- Clone the repository using : `git clone https://github.com/bayegaspard/OpenSetPerf.git`
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


##### Notes:
Some folders have an extra runable file called `Display.py` that uses `matplotlib` to draw a graph like the ones in the original paper.
Unfortunately the graphs for `OpenMax` do not turn out exactly like the paper's, it is currently unknown why.


##### Items in main folder: 

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

### Items inside each folder:
- `Buildinglist.py`
  - This is the file that updates and recreates the lists for all of the names of the images.
  - Change this file and re run it if you want to split the data diffrently

- `list.txt`
  - This is the file that contains a the names of all the images that you want to classify and what class they are part of.

- `list2.txt`
  - This version of list contains the values of items you want to be classified as Unknown.

- `LoadImages.py`
  - This contains a viersion of Dataloader specifically for images and the format of list.txt

- `ModelLoader.py`
  -This contains the model class and structure.

- `External code file`
  - Any one of "OpenMaxByMaXu", "EnergyCodeByWetliu", or "OdinCodeByWetliu"
  - These files contain mainly code that was not made by me. I did my best not to modify it at all and say where it came from.
  - They mainly serve as the implementations of the formulas found in papers.

- `Evaluation.py`
  - This is the big file. It handels the output of the model and can give you back reports on the data.
  - It needs to be created with the correctValCounter class, the name is old.
  - If you want it to be able to save a confusion matrix for you, it needs confusionMat set to true.
  - Use EvalN to evaluate any one of the 4 model interpreters, 
  ```
  type= 
  --"Soft" for SoftMax
  --"Open" for OpenMax, needs a weibull model to be set with setWeibull()
  --"Energy" for Energy OOD
  --"Odin" for ODIN, needs odinSetup(param) to be run
  ```
  - I have done my best to keep all versions of Evaluation.py to be the same in all folders for ease of comparison.

- `main.py`
  - This is the file that runs the training and testing loop.
  - The only big diffrence is that Energy OOD has an addition to the training loss function.
  - This file is very similar to all other versions.

- `checkpoint.pth`
  - This is a saved model.
  - The saves automatically load.
  - The models usually save every 5 epocs and overwrite the old save.
  - All saves are using torch seed 0 so that they split off the same training and testing sets. (Beware contaimination)

- `Dispaly.py`
  - Does not exist in all folders
  - Does not train a model. A checkpoint is needed.
  - Either displays data in a matplotlib pyplot or saves data in a CSV
