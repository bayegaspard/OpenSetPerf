import torch
import Config
import math

#Code from the iiMod file
def distance_measures(Z:torch.Tensor,means:list,Y:torch.Tensor,distFunct)->torch.Tensor:

    intraspread = torch.tensor(0,dtype=torch.float32)
    N = len(Y)
    K = range(len(Config.parameters["Knowns_clss"][0]))
    # K = range(len([0,1,2]))
    #For each class in the knowns
    for j in K:
        #The mask will only select items of the correct class
        mask = (Y==Config.parameters["Knowns_clss"][0][j]).cpu()
        # mask = Y==[0,1,2][j]
        
        #torch.flatten(x,start_dim=1,end_dim=-1)
        intraspread += distFunct(means[j].cpu(),Z.cpu()[mask.cpu()])
    
    return intraspread/N

#Equation 2 from iiMod file
def class_means(Z:torch.Tensor,Y:torch.Tensor):
    means = []
    for y in Config.parameters["Knowns_clss"][0]:
    # for y in [0,1,2]:
        #Technically only this part is actually equation 2 but it seems to want to output a value for each class.
        mask = (Y==y)
        Cj = mask.sum().item()
        sumofz = Z[mask].sum(dim=0)
        means.append(sumofz/Cj)
    return means

def class_means_from_loader(weibulInfo):
    #Note, this masking is only due to how we are handling model outputs.
    #If I was to design things again I would have designed the model outputs not to need this masking.
    data_loader = weibulInfo["loader"]
    model = weibulInfo["net"]

    totalout = []
    totallabel = []
    classmeans = None
    for num,(X,Y) in enumerate(data_loader):
        #Getting the correct column (Nessisary for our label design)
        y = Y[:,0]
        Z = model(X)    #Step 2
        if classmeans is None:
            classmeans = class_means(Z,y)
        elif len(Z) == Config.parameters["batch_size"][0]:
            classmeans = [(x * num + y) / (num + 1) for x, y in zip(classmeans, class_means(Z, y))]
        else:
            classmeans = [(x * num * Config.parameters["batch_size"][0] + y) / (num * Config.parameters["batch_size"][0] + len(y)) for x, y in zip(classmeans, class_means(Z, y))]


    return classmeans

#Derived from ChatGPT. Apparently.
def euclidean_distance(point1, point2):
    if len(point1) != len(point2):
        raise ValueError("Points must have the same dimensions")

    squared_distance = sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2))
    if squared_distance.dim()>0: #ADDED LINE
        distance = [math.sqrt(x) for x in squared_distance]#ADDED LINE
    else:#ADDED LINE
        distance = math.sqrt(squared_distance)
    
    
    return distance


class forwardHook():
    def __init__(self):
        self.distances = {}
        self.class_vals = None #these are the final classifications for each row in the batch
        self.means = {}
        self.distFunct = "intra_spread"
    
    def __call__(self,module:torch.nn.Module,input:torch.Tensor,output:torch.Tensor):
        # print("Forward hook called")
        name = f"{module._get_name()}_{output[0].size()}"
        if self.class_vals is None:
            if output.ndim == 2:
                self.class_vals = output.argmax(dim=1).cpu()
            else:
                self.class_vals = output.cpu()
        else:
            if not name in self.means.keys():
                self.means[name] = class_means(output,self.class_vals)
            if not name in self.distances.keys():
                self.distances[name] = distance_measures(output,self.means[name],self.class_vals,dist_types_dict[self.distFunct])
            else:
                self.distances[name] += distance_measures(output,self.means[name],self.class_vals,dist_types_dict[self.distFunct])
    

dist_types_dict = {
    "Cosine_dist": lambda x1,x2: 1-torch.nn.functional.cosine_similarity(x1,x2[:,:len(x1)]).sum(),
    "intra_spread": lambda x,y:torch.linalg.norm(x-y[:,:len(x)],dim=0).sum(),
    "Euclidean Distance": lambda x1,x2: torch.tensor([euclidean_distance(x1,y2) for y2 in x2[:,:len(x1)]]).sum()
}

# torch.nn.modules.module.register_module_forward_hook(forwardHook())