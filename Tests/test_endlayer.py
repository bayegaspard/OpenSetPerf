#three lines from https://xxx-cook-book.gitbooks.io/python-cook-book/content/Import/import-from-parent-folder.html
import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
sys.path.append(f"{root_folder}/src/main")
sys.path.append(f"{root_folder}/src/CodeFromImplementations")
import src.main.EndLayer as EndLayer
import src.main.Config as Config
import src.main.helperFunctions as helperFunctions
import torch




class sampleNet(torch.nn.Module):
    """This is just an empty neural network"""
    def testingTensor(self):
        testingTensor = [[0]*Config.parameters["CLASSES"][0] for x in range(Config.parameters["CLASSES"][0])]
        for x in range(Config.parameters["CLASSES"][0]):
            testingTensor[x][x] = 1
        testingTensor = torch.tensor(testingTensor)
        return testingTensor, torch.tensor(list(zip(testingTensor.argmax(dim=1),testingTensor.argmax(dim=1))))
    def forward(self, x: torch.Tensor):
        return x

def testtype():
    """
    Tests that endlayer outputs a tensor
    """
    end = EndLayer.EndLayers(Config.parameters["CLASSES"][0])
    example_tensor = torch.Tensor([range(Config.parameters["CLASSES"][0])]*2)
    targets = torch.Tensor([[4,4],[5,Config.parameters["CLASSES"][0]]])
    test= end.endlayer(example_tensor,targets)
    # print(test)
    assert isinstance(test,torch.Tensor)

def testShape():
    """
    Tests the shape of the endlayer
    """
    end = EndLayer.EndLayers(Config.parameters["CLASSES"][0])
    example_tensor = torch.Tensor([range(Config.parameters["CLASSES"][0])]*2)
    targets = torch.Tensor([[4,4],[5,Config.parameters["CLASSES"][0]]])
    test= end.endlayer(example_tensor,targets)
    shape = test.shape[-1]

    assert shape == Config.parameters["CLASSES"][0]+1

def testIfSoftUnknown():
    """
    Tests that softmax does not output unknonws
    """
    end = EndLayer.EndLayers(Config.parameters["CLASSES"][0])
    example_tensor = torch.Tensor([range(Config.parameters["CLASSES"][0])]*2)
    targets = torch.Tensor([[4,4],[5,Config.parameters["CLASSES"][0]]])
    before_argmax = end.endlayer(example_tensor,targets)
    after_argmax = before_argmax.argmax()
    assert torch.all(after_argmax!=Config.parameters["CLASSES"][0])
                     

def testAllEndlayers():
    """
    Tests if all of the endlayers run without chrashes.
    """
    net = sampleNet()
    end = EndLayer.EndLayers(Config.parameters["CLASSES"][0])
    end.prepWeibull([net.testingTensor()],torch.device('cpu'),net)
    for x in ["Soft","Open","Energy","COOL","DOC","iiMod", "SoftThresh"]:
        end.type = x
        example_tensor = torch.Tensor([range(Config.parameters["CLASSES"][0])]*2)
        targets = torch.Tensor([[4,4],[5,Config.parameters["CLASSES"][0]]])
        if x == "COOL":
            example_tensor = torch.cat([example_tensor,example_tensor,example_tensor],dim=-1)
        before_argmax = end.endlayer(example_tensor,targets)
        after_argmax = before_argmax.argmax(dim=1)
        assert isinstance(after_argmax,torch.Tensor)
        assert len(after_argmax) == 2


def testConsecutaveDimentions_a():
    """
    Test if making things consecutive works on the current settings
    """
    net = sampleNet()
    example_tensor, labels = net.testingTensor()
    consecutive_tensor, consecutive_labels = helperFunctions.renameClassesLabeled(example_tensor,labels)
    assert len(example_tensor[0]) != len(consecutive_tensor[0]) or len(Config.class_split["knowns_clss"])==Config.parameters["CLASSES"][0]
    assert consecutive_labels.max() < len(Config.class_split["knowns_clss"])
    assert torch.all(consecutive_tensor[consecutive_tensor.max(dim=1)[0].gt(0)].argmax(dim=1)==consecutive_labels[consecutive_tensor.max(dim=1)[0].gt(0),0])

def testConsecutaveDimentions_b():
    """
    Test if making things consecutive works with no unknowns
    """
    Config.class_split["unknowns_clss"] = []
    Config.class_split["knowns_clss"] = Config.loopOverUnknowns(Config.class_split["unknowns_clss"])
    helperFunctions.Config.class_split = Config.class_split.copy()
    helperFunctions.setrelabel()
    net = sampleNet()
    example_tensor, labels = net.testingTensor()
    consecutive_tensor, consecutive_labels = helperFunctions.renameClassesLabeled(example_tensor,labels)
    assert len(example_tensor[0]) != len(consecutive_tensor[0]) or len(Config.class_split["knowns_clss"])==Config.parameters["CLASSES"][0]
    assert consecutive_labels.max() < len(Config.class_split["knowns_clss"])
    assert torch.all(consecutive_tensor[consecutive_tensor.max(dim=1)[0].gt(0)].argmax(dim=1)==consecutive_labels[consecutive_tensor.max(dim=1)[0].gt(0),0])


def testConsecutaveDimentions_c():
    """
    Test if making things consecutive works with some unknowns
    """
    Config.class_split["unknowns_clss"] = [1,2,3,4]
    Config.class_split["knowns_clss"] = Config.loopOverUnknowns(Config.class_split["unknowns_clss"])
    helperFunctions.Config.class_split = Config.class_split.copy()
    helperFunctions.setrelabel()
    net = sampleNet()
    example_tensor, labels = net.testingTensor()
    consecutive_tensor, consecutive_labels = helperFunctions.renameClassesLabeled(example_tensor,labels)
    assert len(example_tensor[0]) != len(consecutive_tensor[0]) or len(Config.class_split["knowns_clss"])==Config.parameters["CLASSES"][0]
    assert consecutive_labels.max() < len(Config.class_split["knowns_clss"])
    assert torch.all(consecutive_tensor[consecutive_tensor.max(dim=1)[0].gt(0)].argmax(dim=1)==consecutive_labels[consecutive_tensor.max(dim=1)[0].gt(0),0])
