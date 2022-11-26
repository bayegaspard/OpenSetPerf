import torch

opt_func = {"Adam":torch.optim.Adam,"SGD":torch.optim.SGD}
helper_variables = {
    "phase" : -1,
    "startphase" : 0,
    "unknowns_clss": {"unknowns":[2,3]},

    "e": 0
}
parameters = {
    "batch_size":[100, "Number of items per batch"],
    "num_workers":[3, "Number of threads working on building batches"],
    "attemptLoad":[0, "0: do not use saves\n1:use saves"],
    "testlength":[1/4, "[0,1) percentage of training to test with"],
    "num_epochs":[10,"Number of times it trains on the whole trainset"],
    "learningRate":[0.001, "a modifier for training"],
    "threshold":[5,"When to declare something to be unknown"],
    "optimizer":opt_func["Adam"],
    "Unknowns":"refer to unknowns.CSV",
    "CLASSES":[15,"Number of classes, do not change"],
    "Temperature":[1,"Energy OOD scaling parameter"]
}

