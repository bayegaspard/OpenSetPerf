from torch.utils.data import DataLoader
import numpy as np
import torch
import time

# user defined modules
import GPU, FileHandling
from EndLayer import EndLayers
import plots
import Dataload
import ModelStruct
import Config
import os
import helperFunctions
import pandas as pd
import plotly
import plotly.express as px

root_path = os.getcwd()

if __name__ == "__main__":
    print(torch.__version__)


#useful variables
opt_func = Config.parameters["optimizer"]
device = GPU.get_default_device() # selects a device, cpu or gpu

def old_run_model(set,last=None,start=0):
    global fscores


    #This refreshes all of the copies of the Config files, the copies can be used to find the current config if something breaks.
    #At this point these copies are mostly redundent.
    FileHandling.refreshFiles(root_path)

    FileHandling.generateHyperparameters(root_path) # generate hyper parameters copy files if they did not exist.

    #This is an example of how we get the values from Config now.
    knownVals = Config.parameters["Knowns_clss"][0]

    #This just helps translate the config strings into model types. It is mostly unnesisary.
    model_list = {"Convolutional":ModelStruct.Conv1DClassifier,"Fully_Connected":ModelStruct.FullyConnected}
    model = model_list[Config.parameters["model"][0]]() # change index to select a specific architecture.

    if not last is None:
        model.store=last

    #This initializes the data-parallization which hopefully splits the training time over all of the connected GPUs
    model = ModelStruct.ModdedParallel(model)

    #This selects what algorithm you are using.
    model.end.type = Config.parameters["OOD Type"][0]

    #This selects the default cutoff value
    model.end.cutoff = Config.parameters["threshold"][0]

    model.loadPoint("Saves/models")

    model.end.prepWeibull(set1,device,model)

    model.eval()

    

    print("length of Stage1",len(set1),"\nlength of Stage2",len(set2))
    
    run = []    

    for batch in set:
        model.validation_step(batch)
        f1, recall, precision, accuracy = helperFunctions.getFscore(model.store)
        run.append(f1)
    
    # fig = px.scatter(fscores)
    # fig.show()

    turningPoint = len(run)

    runDF = pd.DataFrame([run],columns=range(start,start+turningPoint))
    
    fscores = pd.concat([fscores,runDF])

    return turningPoint,model.store
    
def old_staticDataset():
    knowns, unknowns = FileHandling.getDatagroup()

    #These lines initialize the loaders for the datasets.
    knownsset = DataLoader(knowns, 10, num_workers=Config.parameters["num_workers"][0],shuffle=True, pin_memory=True)
    unknownsset = DataLoader(unknowns, 10, shuffle=True, num_workers=Config.parameters["num_workers"][0],pin_memory=True)



    
    knownsset = Dataload.recreateDL(knownsset)
    unknownsset = Dataload.recreateDL(unknownsset)

    return knownsset, unknownsset

def old_main():
    """
    The main function
    This is what is run to start the model.
    Takes no arguements and returns nothing.
    
    """
    #Finds the current working directory.
    global root_path
    #If the current working directory is in the wrong location it changes the current working directory and prints an error.
    while (os.path.basename(root_path) == "main.py" or os.path.basename(root_path) == "main" or os.path.basename(root_path) == "src"):
        #checking that you are running from the right folder.
        print(f"Please run this from the source of the repository not from {os.path.basename(root_path)}. <---- Look at this!!!!")
        os.chdir("..")
        root_path=os.getcwd()

    #This is what the diffrent plots have overriding their names. If it is a loop it changes for every iteration
    plots.name_override = "Config File settings"
    
    global set1
    global set2
    set1, set2 = old_staticDataset()
    global fscores
    fscores = pd.DataFrame()

    #names= ["Softmax Closedset","Softmax Openset","Openmax Closedset","Openmax Openset","Energy Closedset","Energy Openset","COOL Closedset","COOL Openset","DOC Closedset","DOC Openset"]
    names= ["Softmax Closedset","Softmax Openset","Openmax Openset","Energy Openset","COOL Openset","DOC Openset"]

    #Runs the model
    turningPoint, last = old_run_model(set1)
    old_run_model(set2,last=last,start=turningPoint)
    Config.parameters["OOD Type"][0] = "Open"

    #turningPoint, last = run_model(set1)
    old_run_model(set2,last=last,start=turningPoint)
    Config.parameters["OOD Type"][0] = "Energy"

    #turningPoint, last = run_model(set1)
    old_run_model(set2,last=last,start=turningPoint)
    Config.parameters["OOD Type"][0] = "COOL"

    #turningPoint, last = run_model(set1)
    old_run_model(set2,last=last,start=turningPoint)
    Config.parameters["OOD Type"][0] = "DOC"

    #turningPoint, last = run_model(set1)
    old_run_model(set2,last=last,start=turningPoint)

    fig = px.scatter(y=fscores.iloc[0])
    for x in range(len(fscores)):
        trace = plotly.graph_objs.Scatter(y=fscores.iloc[x],mode="markers",name=names[x])
        fig.add_trace(trace)
    fig.add_vline(x=turningPoint-0.5)
    fig.update_xaxes(title_text='Batch of 10')
    fig.update_yaxes(title_text='Fscore')
    fig.update_yaxes(range=[0, 1])
    fig.show()
    fig.write_image("Saves/GeneratedImages.png")
    
#Tells what Config section each possibility maps to. (so that that specific column can be checked)
valueLocations = {
    "Convolutional": "model",
    "Fully_Connected": "model",
    "Payload_data_CICIDS2017": "Dataset",
    "Payload_data_UNSW": "Dataset"
}

def traceLines(trace:plotly.graph_objs.Trace):
    """
    Changes the trace lines so the graphs are consistant.
    """
    if trace.name == "Soft":
        trace.update({"line":{"dash":'solid',"width":4}})
    elif trace.name == "iiMod":
        trace.update({"line":{"dash":"dashdot","width":4}})
    elif trace.name == "COOL":
        trace.update({"line":{"dash":"longdash","width":4}})
    elif trace.name == "DOC":
        trace.update({"line":{"dash":"dash","width":4}})
    elif trace.name == "Energy":
        trace.update({"line":{"dash":"dot","width":4}})
    elif trace.name == "Open":
        trace.update({"line":{"dash":"dot","width":4}})
    else:
        trace.update({"line":{"dash":'solid',"width":4}})
    
    if trace.name == "Soft":
        trace.update({"marker":{"symbol":'circle',"size":8}})
    elif trace.name == "iiMod":
        trace.update({"marker":{"symbol":'square',"size":8}})
    elif trace.name == "COOL":
        trace.update({"marker":{"symbol":'x',"size":8}})
    elif trace.name == "DOC":
        trace.update({"marker":{"symbol":'cross',"size":8}})
    elif trace.name == "Energy":
        trace.update({"marker":{"symbol":'diamond',"size":8}})
    elif trace.name == "Open":
        trace.update({"marker":{"symbol":'star-open',"size":8}})
    else:
        trace.update({"marker":{"symbol":'circle-x',"size":8}})

def keepFirstThree(df:pd.DataFrame):
    """
    Keeps only the three latest values for any model/dataset pair. 
    """
    df.drop_duplicates(inplace=True)
    df["Version"] = pd.to_numeric(df["Version"])
    df.sort_values(by="Version",inplace=True,kind="mergesort")#I want a stable sort so I am using mergesort.
    final = pd.DataFrame()
    for x in range(3):
        # print(f"The length is {len(df)}")
        temp = df.drop_duplicates(subset=["Currently Modifying","model","Dataset"],keep="last")
        final = pd.concat([final,temp])
        df = pd.concat([df,temp])
        # print(f"The length is (should be more) {len(df)}")
        df.drop_duplicates(inplace=True,keep=False)
    assert len(final)%3 ==0
    return final


def main(save=True,show=False, minimumVersion=None, bysection=False, latex=False):
    """
    Creates pivot tables based on the data in Scoresall.csv. It then displays the data in up to three different ways.
    MinimumVersion deturmines what rows of the datatable to use based on the Version column. 
    If no version is specified then the current version of the code base is used.

    If Save is true then main() creates graphs based on the data and saves the graphs in Saves/images/
    If Show is true then main() opens the graphs made by Save and shows them in an internet window
    If latex is true then main() will save the pivot table as a latex table in Saves/images/

    """
    if minimumVersion is None:
        #Getting version number
        #https://gist.github.com/sg-s/2ddd0fe91f6037ffb1bce28be0e74d4e
        f = open("build_number.txt","r")
        minimumVersion = int(f.read().strip())

    if not os.path.exists("Saves/images/"):
        os.mkdir("Saves/images")
    
    whole_table = pd.read_csv("Saves/Scoresall.csv")
    if "Type of modification" in whole_table:
        whole_table = whole_table[whole_table["Version"]!="OLD"] #Sort out all unreadable versions
        whole_table = whole_table[whole_table["Version"].astype(int)>=minimumVersion] #Only accept version numbers above value
        whole_table["Unknowns"] = whole_table["Unknowns"].str.replace(" Unknowns","") #Remove unnessisary text
        whole_table = whole_table[whole_table["Type of modification"]!="Default"] #Remove Defaults
        whole_table = whole_table[whole_table["Type of modification"].notna()]  #Also remove defaluts

        whole_table = keepFirstThree(whole_table)
        


        for z1 in ["Convolutional","Fully_Connected"]:
            part_tabel = whole_table[whole_table[valueLocations[z1]]==z1].copy()
            for z2 in ["Payload_data_CICIDS2017","Payload_data_UNSW"]:
                part_tabel2 = part_tabel[part_tabel[valueLocations[z2]]==z2].copy()
                counts = part_tabel2["Currently Modifying"].value_counts()
                if counts.min() != counts.max():
                    print(f"{z1}-{z2} has an unbalenced number of samples, {counts.min()}!={counts.max()}")
                if graphTabel(part_tabel2,show=show,save=bysection) == -1:
                    print(f"{z1}-{z2} was unable to find samples for graphs")


        whole_table[whole_table[valueLocations["Convolutional"]]=="Convolutional"]
        whole_table[whole_table[valueLocations["Payload_data_CICIDS2017"]]=="Payload_data_CICIDS2017"]
        graphTabel(whole_table,show=show,save=save,latex=latex)

    
    

def graphTabel(df:pd.DataFrame,show=False,save=True,latex=False,extrapath=""):
    """
    This function creates a series of tables based on different pivot tables created from the dataframe df.


    If Save is true then main() creates graphs based on the data and saves the graphs in Saves/images/
    If Show is true then main() opens the graphs made by Save and shows them in an internet window
    If latex is true then main() will save the pivot table as a latex table in Saves/images/
    """
    if len(df) <2:
        print("Dataframe not enough values")
        return -1

    for prefix in ["","AUTOTHRESHOLD_","AUTOTHRESHOLD2_"]:
        for y in [f"Test_F1",f"Val_F1",f"Test_Found_Unknowns"]:
            for x in set(df["Type of modification"]):

                part_table = pd.pivot_table(df[df["Type of modification"]==x],values=f"{prefix}{y}",index=[f"{x}"],columns=["OOD Type"],aggfunc=np.mean)
                # print(part_table)
                
                if x in ["Activation"]:
                    fig = px.scatter(part_table)
                elif x in ["Datagrouping","Unknowns"]:
                    fig = px.line(part_table,markers=True)
                else:
                    fig = px.line(part_table,markers=True,log_x=True)
                xaxis = x
                if xaxis == "MaxPerClass":
                    xaxis = "Datapoints per class"
                fig.update_layout(yaxis_title=y,xaxis_title=xaxis,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"size":18,"color":"rgba(0,0,0,255)"},legend_title_text='Algorithm')
                fig.update_yaxes(range=[0, 1],gridcolor="rgba(200,200,200,50)",zerolinecolor="rgba(200,200,200,50)",zerolinewidth=1)
                fig.update_xaxes(gridcolor="rgba(200,200,200,50)",zerolinecolor="rgba(200,200,200,50)",zerolinewidth=1,exponentformat='power')
                    
                fig.for_each_trace(traceLines)

                if show:
                    fig.show()
                if save:
                    fig.write_image(f"Saves/images/{extrapath}{prefix}{y}{x}.png",scale=4)
                if latex:
                    part_table.to_latex(f"Saves/images/{extrapath}{prefix}{y}{x}",float_format="%.2f")

if __name__ == '__main__':
    main(minimumVersion=422, latex=True)




