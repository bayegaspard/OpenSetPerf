## Hyperparameter Modification
This is a description of how to modify the hyperparameters in the `Config.py` file.

### Steps to modify the unknown classes

- First you need to open the file `Config.py`
- Find the definition of the dictionary `parameters`
- Inside the definition you will find a key called `"unknowns_clss"` (the reason for class to be spelled clss is unknown)
- To add classes to the unknowns you can simply add numbers 0 to 14 in the associated array. The key to which classes are which number can be found in line 10 of the file `Dataload.py`.

### Alternatively

- When the code is run with `python src/main/main.py` you can modify the parameters by adding arguments. See `python src/main/main.py -h` for details.

### Steps to modify the hyperparameters

- First you need to open the file `Config.py`
- Then you need to find the definition of `parameters`
- There is a brief description of how the parameters are organized in the first few lines.
    - Each line is it's own parameter.
    - Each line has three sections.
    - The first section before the colon (`:`) is the name of the parameter. Don't change it.
    - The second section is after the colon and open square bracket (`:[`) and before the comma (`,`) This is the section that you can change.
    - The last section is between the comma (`,`) and the ending close (`],`). This section is just the description, change it if you want to.
- There are currently 20 hyperparameters those being:
    - batch_size - This is the size of chunks that are fed into the model. Bigger chunks are better but will have memory usage. Pick an integer between 100 and 100000.
    - num_workers - This is the hyperparameter that determines the number of threads working to build the chunks. They will help the model run faster to a point but increase memory. Pick an integer from 2 to 8 or 0 to disable entirely.
    - attemptLoad - This is to load the model from the most recent file in `Saves/Epoch{xxx}.pth`. It might not be working as it has not been tested in a while. pick either 0 or 1.
    - testlength - This is the percent of the training set will be split off into the test set. The test set will also have all of the unknown classes as well as this percent that is split off. Can be any number in the range [0,1)
    - MaxPerClass - This is the limit of training examples for every class. It can be any positive number, bigger numbers will make a better model but be much slower to train. Also, dendrogram limit will have this cap the number of examples in each subclass instead meaning that this number should be lower if you are using dendrogram limit.
    - num_epochs - The number of epochs the model trains. Bigger numbers means more training. More training means a better model, but it happens slower. And it can be any natural number.
    - learningRate - This is the effect of each epoch's training. Bigger numbers mean the training can get better faster but also have the possibility of getting much worse faster as well. Learning rates should be numbers far smaller than 1.
    - threshold - Threshold determines what the cutoff activation is for finding unknowns. It is not used for all of the algorithms. Can be any positive number.
    - model - Select either a convolutional model or a fully connected model.
    - OOD Type - Select which algorithm you want to use to find the unknowns. Five are currently avalible and are listed in the description.
    - Dropout - Dropout is a method to prevent overfitting on large models. For small models you want dropout to be quite low but if you are making a bigger model you can increase it a bit. It can be any number from [0,1)
    - Datagrouping - This is stating if you want the data to be split by methods for dendrograms before loading it or not. If you are running the model on the whole dataset this parameter does not matter. It does matter if you set the MaxPerClass to a low number. Note: Each method generates it's own folder to contain the grouped datasamples.
    - optimizer - Not really a parameter at the moment but this is the optimizer we are using.
    - Unknowns - This is just a note to see the unknows section because lists do not save well in CSV files.
    - CLASSES - This is also not a parameter to modify. It is just here to tell things how many classes they should expect.
    - Temperature - Energy OOD specific parameter. It modifies how scaling works.
    - Degree of Overcompleteness - COOL specific parameter. It modifies how many nodes there are per class. A value of 1 turns COOL into Softmax. It can take any positive integer.
    - Number of Layers - The number of extra fully connected layers to add into the model. Each extra layer needs more training but can improve the model's performance in the long run. It can be any integer ≥0
    - Nodes - The number of nodes in the added layers. This also is the number of nodes before the last layer, so it has an effect even if the number of layers is zero. Default is 256.
    - Activation - This is the activation function to use. Currently the only option is ReLU but if you are using bigger models you should add leakyReLU.
    - LOOP - This is a hyper-hyperparameter that if it is 1 the model loops over all of the algorithms in order and saves the values.
    - Dataset - This is a hyperparameter that selects which dataset to use. Currently can only be "Payload_data_CICIDS2017" or "Payload_data_UNSW"
    - SchedulerStepSize - This states the number of epochs that are run with the current learning rate before it is reduced.
    - SchedulerStep - This is the multiplier for the learning rate every `SchedulerStepSize` epochs.
    - ApplyPrelimSoft - Not used any more, but it applies a thresholded softmax and only uses OOD if that value is less than the threshold.
    - ItemLogitData - Saves the final logits and labels for those logits in `Saves/items.csv`
    - SaveBatchData - Saves the values and distances from each batch for further study.
    - Var_filtering_threshold - Threshold for the var filtering method. This can either be a single value or a pair of values in an array, if it is a pair, the known values are the ones that fall outside of the range.
    - Experimental_bitConvolution - A side project to use 2 dimensional convolutions over the bits instead of 1 dimensional convolutions over the bytes.
