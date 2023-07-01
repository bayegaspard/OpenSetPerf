
import pandas as pd

import plotly
import plotly.express as px

rev=False
bar=False
logX=False
scatter = False
row_labels = True

# File: ScoresAll(autorecovered)2/sheet1
# fileNames = ["TestF1","TestAccuracy","TestRecall","TestPrecision","FoundUnknowns","KnownsF1","KnownsAccuracy","KnownsRecall","KnownsPrecision"]
# axisNames = ["F1 Score","Accuracy","Recall","Precision","Percent of Unknowns Found","F1 Score","Accuracy","Recall","Precision"]
# rev = True

#File: ScoresLoop/toPrint
# fileNames = ["TestF1_Activation","FoundUnknowns_Activation","KnownsF1_Activation","TestF1_LR","FoundUnknowns_LR","KnownsF1_LR","TestF1_DPPerClass","FoundUnknowns_DPPerClass","KnownsF1_DPPerClass"]
# axisNames = ["F1 Score","Percentage of Unknowns Found","F1 Score","F1 Score","Percentage of Unknowns Found","F1 Score","F1 Score","Percentage of Unknowns Found","F1 Score"]
# bar = True
# scatter = True
# graphNumber = 0

# fileNames = ["TestF1_LR","FoundUnknowns_LR","KnownsF1_LR"]
# axisNames = ["F1 Score","Percentage of Unknowns Found","F1 Score"]
# logX=True

#File: ScoresLoop/toPrint2
# fileNames = ["AccuracyTest_Activation","RecallTest_Activation","PrecisionTest_Activation","AccuracyVal_Activation","RecallVal_Activation","PrecisionVal_Activation","AccuracyTest_LR","RecallTest_LR","PrecisionTest_LR","AccuracyVal_LR","RecallVal_LR","PrecisionVal_LR","AccuracyTest_DPPerClass","RecallTest_DPPerClass","PrecisionTest_DPPerClass","AccuracyVal_DPPerClass","RecallVal_DPPerClass","PrecisionVal_DPPerClass"]
# axisNames = ["Testing Accuracy","Testing Recall","Testing Precision","Validation Accuracy","Validation Recall","Validation Precision","Testing Accuracy","Testing Recall","Testing Precision","Validation Accuracy","Validation Recall","Validation Precision","Testing Accuracy","Testing Recall","Testing Precision","Validation Accuracy","Validation Recall","Validation Precision"]
# bar = True
# scatter = True
# graphNumber = 0

#File: Current scores Loop
fileNames = ["TestF1_BatchSize","FoundUnknowns_BatchSize","KnownsF1_BatchSize","TestF1_Grouping","FoundUnknowns_Grouping","KnownsF1_Grouping","TestF1_LR","FoundUnknowns_LR","KnownsF1_LR","TestF1_DPPerClass","FoundUnknowns_DPPerClass","KnownsF1_DPPerClass","TestF1_Epochs","FoundUnknowns_Epochs","KnownsF1_Epochs"]
axisNames = ["F1 Score","Percentage of Unknowns Found","F1 Score","F1 Score","Percentage of Unknowns Found","F1 Score","F1 Score","Percentage of Unknowns Found","F1 Score"]
row_labels = False

for graphNumber in range(len(fileNames)):
    input(f"{fileNames[graphNumber]} please.")
    df = pd.read_clipboard(index_col=0)

    if not bar:
        fig = px.line(df,markers=True,log_x=logX)
    elif scatter:
        fig = px.scatter(df)
    else:
        fig = px.bar(df)
    if graphNumber>4 and rev:
        fig.update_xaxes(autorange="reversed")
    fig.update_layout(yaxis_title=axisNames[graphNumber%len(axisNames)],xaxis_title="",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",font={"size":18,"color":"rgba(0,0,0,255)"},legend_title_text='Algorithm')
    fig.update_yaxes(range=[0, 1],gridcolor="rgba(200,200,200,50)",zerolinecolor="rgba(200,200,200,50)",zerolinewidth=1)
    fig.update_xaxes(gridcolor="rgba(200,200,200,50)",zerolinecolor="rgba(200,200,200,50)",zerolinewidth=1,exponentformat='power')
    #from: https://www.datacamp.com/cheat-sheet/plotly-express-cheat-sheet
    #fig.update_traces(patch={"line":{"dash":"dot","width":4}})

    #From: https://plotly.com/python/creating-and-updating-figures/?_gl=1*9mey7b*_ga*ODY1NzkwODMuMTY4MzkyODAyNQ..*_ga_6G7EE0JNSC*MTY4NDc3MjkwMy45LjEuMTY4NDc3NjM0My4wLjAuMA..#updating-traces
    # fig.for_each_trace(
    #     lambda trace: trace.update(marker_symbol="square") if trace.name == "setosa" else (),
    # )

    def traceLines(trace:plotly.graph_objs.Trace):
        if not bar:
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
            else:
                trace.update({"line":{"dash":'solid',"width":4}})
        elif scatter:
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
            else:
                trace.update({"marker":{"symbol":'circle-x',"size":8}})

    fig.for_each_trace(traceLines)

    #fig.show()
    fig.write_image(f"/Users/abroggi/Desktop/images/V3/{fileNames[graphNumber]}.png",scale=4)

    if graphNumber>=2:
        bar=False
    