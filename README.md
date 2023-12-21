## OpenSetPerf
A Practitioner's Guide to the Performance of Deep-Learning Based Open Set Recognition Algorithms for Network Intrusion Detection Systems

### Steps to run:

- Clone the repository using the following commands:

`git clone https://github.com/bayegaspard/OpenSetPerf.git`
- Make sure you do that from the `dev` branch.
- Download the [Payload-Byte NIDS Dataset](https://github.com/Yasir-ali-farrukh/Payload-Byte/tree/main/Data) 
- Navigate to the root folder and place the downloaded CSV file in the `dataset` folder. New structure will be `dataset\Payload_data_CICIDS2017.csv` for the CIC dataset and `dataset\UNSW-NB15`.
- If you don't have pip3 installed, you can use the command below to install one.

`sudo apt-get install python3-pip
`
- Navigate to the `src` directory using the command `cd OpenSetPerf\src` directory.
- It is recommended to perform this test in a virtual environment. This step is optional.
```
pip3 install virtualenv
virtualenv opensetperf
source opensetperf/bin/activate
```
- Install required packages using the command below:
`pip3 install -r requirements.txt
`
- Navigate up one directory `cd ..` into the root directory for the Repo.
- You may need to create some folders in the Saves directory namely, `OpenSetPerf\Saves\conf`,`OpenSetPerf\Saves\models`, and `OpenSetPerf\Saves\roc`.
- Run the model using `python3 src\main\main.py`.
- Get hyperparameter information by using `python3 src\main\main.py -h`.
- Alternatively, you can run `chmod +x ./threeRuns.sh` and then `.\threeRuns.sh` to run the model three times.
- Saves and model outputs will generate in the `Saves` folder.

- Edit the `src/main/Config.py` file to change the hyperparameters for the model. More information in `src/main/README.md`. You can also edit the hyperparameters using command line parameters, see `python3 src\main\main.py -h` for more details.



##### Items in root folder: 


- `requirements.txt`
  - File containing the version numbers of the required external libraries

- `src`
  - This is the folder that contains all of the code from this project
  - `main`
    - Folder containing all of the central aspects for running the model.
      - `main.py` - The control file for the entire model.
      - `Config.py` - This is the file that controls all of the hyperparameters of the model.
      - `Dataload.py` - This file gets the whole dataset and splits it up into chunks that the model can read.
      - `FileHandeling.py` - This file controls dealing with files.
      - `ModelStruct.py` - This file defines the model and its structure. But it does not implement the different algorithms.
      - `EndLayer.py` - This file works with the folder `CodeFromImplementations` to implement each of the different algorithms.
      - `plots.py` - This file generates 4 png files of different matplotlib graphs.
      - `GPU.py` - This file helps run the model on different GPUs or move tensors from one GPU to the CPU.
      - `GenerateImages.py` - This file reads the save file and generates images in `Saves/images/` to visualize the data.
      - `helperFunctions.py` - This file contains all other functions that are not contained in another file.
      - `Distance_Types.py` - This file contains the distance calculations for logits.

  - `CodeFromImplementations`
    - This is the code we used to implement the different algorithms including:
      - OpenMax
      - Energy OOD
      - Competitive Overcomplete Output Layer (COOL)
      - Deep Open Classification (DOC)
      - OpenNet (iiMod) Created from the equations in the related paper.
    - This code is modified as minimally as possible.
    - All of the implementations list a link at the top of where they were sourced from.

- `Saves`
  - This is the output file that will save all of metrics from the model.
  - It has many different types of files such as:
    - Data/DataTest - This saves the specific dataloaders from the last time the model was run including the train/test split as to not contaminate the model if it is run again.
    - Scoresall.csv - This file saves the model state and all standard metrics from the model. 
    - models - This file stores the most trained models for each of the algorithms
    - images - this is a folder that stores graphical representations of the `Scoresall.csv`
    - ROC_data - these are files that store the receiver operator characteristic data. It can be used to inform about the settings of threshold. The first line is false positives, the second is true positives, and the third is thresholds to achieve those positives.
  - The following files are still generated but are not used:
    - fscore - this saves the Config parameters and the associated f-score that those parameters got to.
    - history/history{Algorithm} - These save all of the output measures from each of the algorithms after each epoch. 
    - phase - Unused from a previous  refactor, it used to be a save of where in the models training we last got to.
    - scores/scores{Algorithm} - Unused from a previous refactor. It is now unknown what is being stored.
    - hyperparam - saves the Config of the last time the file was run.
    - unknown - saves which classes were unknown from the last time the file was run.
    - batch - saves information about each batch that has been run. NOTE: this file can break if it is saved to too many times, you may need to delete it and allow it to regenerate.
    
- `Tests`
  - This is a folder of pytest tests to ensure the model works properly.
  - test_endlayer.py - Tests that the endlayer outputs correctly given a valid input.
  - test_fullRun.py - Runs the whole model on minimal settings with No-save mode so that it does not mess up the save file.
  - test_LongTests.py - Runs the code on the whole loop. Takes a long time so it is not tested. Does not work.
  - test_loops.py - tests if the loop functions in main.py work.
  - test_other.py - tests things not in the other categories.

- `datasets`
  - We place the NIDS dataset in this folder.
  - Up to two more folders will automatically generate. If you get a warning that a file does not exist, try deleting the generated folders and allowing them to regenerate.

- `build_number.txt`
  - This is a number that is included in `Saves/Scoresall.csv` to inform about which version of the code was used to generate the outputs.

- `threeRuns.sh`
  - This is a simple and small shell script to run the model three times.

### Data Used:
  -The data located at `Saves/Scoresall-ArchivedForPaper.csv` is the data generated for a paper. Not all of the data in the file was used for the paper as some of it is old. Look at the Version column, the values of version numbers greater than or equal to 422 were used with the `src/main/Generateimages.py` file to create the data for the paper. 
