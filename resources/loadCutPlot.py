import uproot4 as up
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import sys
import copy

'''
ChangeLog:
08-06 Version:LHCb Style plots added and the mooseDir and strippingDir added

15-06 Version: 	Abillity to load mulptiple files at once/indicate which version we have loaded, flatten array of multiple
				+ whatever changes required for functionality to save plot
				creation/editing in an external file and run that.

17-06 Version:  Realised best way to do multifile is to have a "load file function" that takes a file location input and makes a corresponding dictionary... 
                can then switch between modes to decide what dataset being plotted on

14/10: Move from SBStudy to Main Repo, rename to "loadCutPlot.py" designed for loading in a tuple and doing small checks and cuts on it!

'''

####
#Setup
#need to import all the possible tuples, can't access new variables within this module apparently
#also want to setup an empty list to use that maybe would allow things to be accessed!

#from tupleLoc import * actually don't need this,
#only issue with imports is when an eval is going on, an eval only has scope of that program

storageList=[] #list for placing objects in that are needed for later ecal statements


#leftoverDecayFiles
nameList=[r"$\Lambda_b^0 \rightarrow \Lambda (1520) \mu\mu$ Data"]
#nameList=[r"$\Lambda_b^0 \rightarrow \Lambda (1520) \mu\mu$ Simulation"]
#nameList=[r"$\Lambda_b^0 \rightarrow \Lambda (1520) \mu e$"+"\n Gen-Level Simulation"]

#for strippingtests
#nameList =[r"$\Sigma_b^{\pm} \rightarrow \Lambda_b^0 (\rightarrow \Lambda (1520) \mu X) \pi^{\pm}$"]

#LbBlindedRegion=[5000,5800]
LbBlindedRegion=[5200,5800]
dataLbMRange=[3700,6900,300]

sampNum=0

pkmumuDataType=""
MC=False

nameList=[r"$\Lambda_b^0 \rightarrow \Lambda (1520) \mu e$ Simulation"]

if pkmumuDataType=="":
    pass
#some default loading mechanisms
elif pkmumuDataType=="MC":
    fileList =[mooseDirMC+"job133-CombDVntuple-15114001-MC2016MD-1520mumu-MC.root"]
    nameList=[r"$\Lambda_b^0 \rightarrow \Lambda (1520) \mu\mu$ Simulation"]
    MC=True
elif pkmumuDataType=="Data":
    fileList =[mooseDirData+"job118-CombDVntuple-quarter2016MD-pKmumu-Data.root"]
    nameList=[r"$\Lambda_b^0 \rightarrow \Lambda (1520) \mu\mu$ Data"]
    MC=False


targetParticles=["Lb","p","K","L1","L2"]
#L1 is always muon and L2 is always electron!!!

#remember when doing mass substitution:
#M1 is the accompanying muon of Lb_M013... and M2 is for LB_m012!!! M1 is mu+ & 3 ,
# M2 is mu- and 2 I should check I understand what p and K are too... p is 0, K is 1

#this may mean that L1 is 3.. depends on charge
#02 will always have opposite charge.
# will change identity dependent on 

#not too worried about substitutions, I can do recalculations mysled. momentum is unchanged
#just mass calculations, can do that manually

openFileCheck=[False]

#treeString="MCDecayTreeTuple/MCDecayTree"
treeString="DTT1520mumu/DecayTree"

#for stripping
#treeString= "DTTSb/DecayTree"

#everything is stored in this LbData dictionary... if we had two open at once these would compete
#ideally want two separate caches.. how about a dictionary of dictionaries! This should be on the
#secondary file side though!
#Na do it this side with a "mode" variable... can have default names set.. also I guess have fullList diff variable

fullListDict={}
fileLocationDict={}
dataDict={}
decayNameDict={}
MCDict={}
currMode=["default"]


def loadNTuple(fileLoc,treeID=treeString,datasetName="default",decayName=nameList[0],MCDataset=True):

    #with up.open(f"{fileList[sampNum]}:DTT1520mumu/DecayTree") as f1:
    with up.open(f"{fileLoc}:{treeID}",num_workers=4) as f1:
        #print(f1.show())
        fullListDict[datasetName]=np.array(f1.keys())


        #goodList=f1.show()
        #targetVarList=[]
        #for targetParticle in targetParticles:
        #    targetVarList+=fullList[[(True if targetParticle in a else False) for a in fullList]]
        #print(fullList[[(True if "Lb" in a else False) for a in fullList]])

        
        #simple way to load in all Lb_ info, leads to very large files
        #LbData=f1.arrays(fullList[[(True if "Lb_" in a else False) for a in fullList]],library="np")

        #LbData.update(f1.arrays(fullList[[(True if "LStar_" in a else False) for a in fullList]],library="np"))


        #LbData.update(f1.arrays(fullList[[(True if "JPs_" in a else False) for a in fullList]],library="np"))

    dataDict[datasetName]={} #stores actual variables in dictionaries within dictionary
    decayNameDict[datasetName]=decayName #decay descriptor storage for easy plot presentation
    MCDict[datasetName]=MCDataset #informs which datasets are montecarlo
    fileLocationDict[datasetName]=fileLoc+":"+treeID #location for loading further variables
    currMode[0]=datasetName #activates loaded dataset

# def cutProcess(inpVar):
#     #convert "Lb_M" to LbData["Lb_M"], allows data to still be in separate dictionaries
#     #but cuts can be called with simply the variable name

#written to use cache instead

#cache variables when used for first time
def cacheData(searchString):
    with up.open(fileLocationDict[currMode[0]],num_workers=4) as f1:
         dataDict[currMode[0]].update(f1.arrays(fullListDict[currMode[0]][[(True if searchString in a else False) for a in fullListDict[currMode[0]]]],library="np"))

#allows user to search through a tuple
def listSearch(containsString):
    for a in fullListDict[currMode[0]]:
        if containsString in a:
            print(a)

#facillity to speed up loading times by only opening a tuple once when loading
#multiple variables in one calculation
#also performs slicing on jagged variables with different length elements for diff events
def LbDataGet(var=None,sliceNum=0):
    if var is None:
        with up.open(fileLocationDict[currMode[0]],num_workers=4) as f1:
            print(f1.show())
        return None

    if var in dataDict[currMode[0]].keys():
        pass
    elif openFileCheck[0]:
        global f2
        f2=up.open(fileLocationDict[currMode[0]],num_workers=4)
        openFileCheck[0]=False
        dataDict[currMode[0]].update(f2.arrays([var],library="np"))

    else:
        try:
            if not f2.closed:
                dataDict[currMode[0]].update(f2.arrays([var],library="np"))
            else:
                with up.open(fileLocationDict[currMode[0]],num_workers=4) as f1:
                    dataDict[currMode[0]].update(f1.arrays([var],library="np"))
        except:
            with up.open(fileLocationDict[currMode[0]],num_workers=4) as f1:
                dataDict[currMode[0]].update(f1.arrays([var],library="np"))
    if dataDict[currMode[0]][var].dtype=='O':
        print("WARNING: Var is type 'object', could be a jagged array.")
        if sliceNum==0:
            print("Taking 1st value for each candidate.")
        if sliceNum != "flatten":
            objList=[sliceNum]
            for g in range(1,len(dataDict[currMode[0]][var])):
                objList.append(objList[g-1]+len(dataDict[currMode[0]][var][g-1]))
            #print(objList)
            return np.hstack(dataDict[currMode[0]][var])[objList]
        else:
            return np.hstack(dataDict[currMode[0]][var])
        
    else:
        return dataDict[currMode[0]][var]
            
#process to convert a string of cut into a mask
def evalCut(cutString,plotArgs,dataToPlot):
    #print(cutString)
    openFileCheck[0]=True
    cutProcessed=eval(cutString)
    if not openFileCheck[0]:
        f2.close()
    openFileCheck[0]=False 
    return cutProcessed


#runs the blinding check to ensure a human must say yes to show any data variable
def userCheckBlind(var):
    while True:
        a=input("Signal region not blinded, is variable '%s' ok? (must answer 'yes')"%var)
        if a=="yes":
            return True
        elif a=="n" or a=="no":
            print("Blinding...")
            return False


#converts input "varName" into something that will actually retrieve variable with cuts
def calculateVar(var):
    #convert variable in form "Lb_M01" into LbDataGet and return this...
    #at the start of LbShow Func Will calculate plotting values once for easier and quicker manipulation
    #also want to allow calculation and combination of values such as "Lb_M23"**2 + whatever
    #for multiple items/calculations.. need to be in '' single quotes
    if var.find('"')==-1:
    	if var.find('LbDataGet')!=-1:
    		return evalCut(var,None,None)
    	else:
        	return evalCut('LbDataGet("'+var+'")',None,None)

    else:
        charPos=var.find('"',0)
        switchStart=True
        while charPos!=-1:
            if switchStart:
                var=var[:charPos]+"LbDataGet("+var[charPos:]
                charPos=var.find('"',charPos+11)
            else:
                var=var[:charPos+1]+")"+var[charPos+1:]
                charPos=var.find('"',charPos+2)
            switchStart=not switchStart
        
        return evalCut(var,None,None)

# plotting dictionary 
def plotDecDict(xlab="default",ylab="Candidates",xUnit="MeV",histCol="b",fillHist=False,xscale="linear",
                yscale="linear",logHist=False,customLabel="",customTitle="",density=False,step=False,
                plotLHCbStyle=False,yUnit="MeV",colormap="viridis",zlab="Candidates",alpha=1,zscale="linear"):
    return {"xlab":xlab,"ylab":ylab,"xUnit":xUnit,"histCol":histCol,"fillHist":fillHist,"xscale":xscale,
            "yscale":yscale,"logHist":logHist,"customLabel":customLabel,"customTitle":customTitle,
            "density":density,"step":step,"plotLHCbStyle":plotLHCbStyle,"yUnit":yUnit,"colormap":colormap,
            "zlab":zlab,"alpha":alpha,"zscale":zscale}

#previously LbShow
#creates a histogram with given cuts and can return said histogram
#allows customaisation of plots and provides blinding
def create1DHist(var=None,TCut=("default",None),plotArgs=None,blindSignalRegion=True,plotDecor=plotDecDict(),overlay=False,debug=False,outputHist=True):
    if not isinstance(TCut,(tuple)):
    	TCut=(TCut,TCut)


    if var is None:
        print(dataDict[currMode[0]].keys())
        return None
    #var=var.astype(str)
    cut = []
    showEventYields=True
    if blindSignalRegion:
        #cut+=LbBlindedSignalRegionCut
        cut+=["(dataToPlot<%i)|(dataToPlot>%i)"%(LbBlindedRegion[0],LbBlindedRegion[1])]
        showEventYields=False
    elif not MCDict[currMode[0]]:
        if not userCheckBlind(var):
            #cut+=LbBlindedSignalRegionCut
            cut+=["(dataToPlot<%i)|(dataToPlot>%i)"%(LbBlindedRegion[0],LbBlindedRegion[1])]
            showEventYields=False

    dataToPlot=calculateVar(var)
    if plotArgs is not None:
        if len(plotArgs)==3:
            plotArgs=(plotArgs[0],plotArgs[1],plotArgs[2]+1)
            bins=np.linspace(*plotArgs)
            #cut+=["dataToPlot>plotArgs[0]","dataToPlot<plotArgs[1]"]
        elif len(plotArgs)==2:
            #need to replace with range
            #cut+=["dataToPlot>plotArgs[0]","dataToPlot<plotArgs[1]"]
            bins="auto"
        else:
            bins=plotArgs[0]
    else:
        bins="auto"

    if TCut[1] is not None:
        cut+= TCut[1]
    #cut+= TCut if TCut is not None else []

    cut= "("+")&(".join(map(str,cut))+")"
    #print(cut)
    cutString=cut
    if debug:
        print(cut)
    cut=evalCut(cut,plotArgs,dataToPlot)
    #print(bins)

    
    if not overlay:
        plt.figure()
    else:
        if plotArgs is not None:
            bins=plotArgs
    lbHist=plt.hist(dataToPlot if len(cut)==0 else dataToPlot[cut] ,
        bins=bins,  fill=plotDecor["fillHist"],   ec=plotDecor["histCol"],
        density=plotDecor["density"],
        histtype="step" if plotDecor["step"] else "bar",
        log=plotDecor["logHist"], alpha=plotDecor["alpha"],
        label=plotDecor["customLabel"]+((" Entries:%i"%len(dataToPlot if len(cut)==0 else dataToPlot[cut])) if showEventYields else " Data outside\nblinded region"))

    #only update labels etc if new plot
    if not overlay:
        plt.ylabel("%s / %.2g %s "%(plotDecor["ylab"],lbHist[1][1]-lbHist[1][0],plotDecor["xUnit"]))
        plt.xlabel("%s [%s], %i bins"%(var if plotDecor["xlab"]=="default" else plotDecor["xlab"] ,plotDecor["xUnit"],len(lbHist[0])))
        #lbHist[0][(lbHist[1][1:]>5000)&(lbHist[1][:-1]<5800)]=0
        if plotDecor["customTitle"]=="":
            plt.title(("MC" if MCDict[currMode[0]] else "Data") + " sample of "+decayNameDict[currMode[0]] +(" with cuts:\n"+ "\n".join(map(str,TCut[0])) if isinstance(TCut[0],list) else  ((" with cuts:\n"+TCut[0]) if TCut[0] != "default" else "")))
        else:
            plt.title(plotDecor["customTitle"])
        plt.xscale(plotDecor["xscale"])
        plt.yscale(plotDecor["yscale"])
    plt.legend()
    plt.show()
    if plotDecor["plotLHCbStyle"]:
        plt.figure()
        plt.errorbar((lbHist[1][1:]+lbHist[1][:-1])/2,lbHist[0],fmt='kx',
            xerr=(lbHist[1][1:]-lbHist[1][:-1])/2,yerr=np.sqrt(lbHist[0]),
            capsize=2,alpha=plotDecor["alpha"],
            label=plotDecor["customLabel"]+((" Entries:%i"%len(dataToPlot if len(cut)==0 else dataToPlot[cut])) if showEventYields else "Data outside\nblinded region"))
        if not overlay:
            plt.ylabel("%s / %.2g %s "%(plotDecor["ylab"],lbHist[1][1]-lbHist[1][0],plotDecor["xUnit"]))
            plt.xlabel("%s [%s], %i bins"%(var if plotDecor["xlab"]=="default" else plotDecor["xlab"] ,plotDecor["xUnit"],len(lbHist[0])))
            #lbHist[0][(lbHist[1][1:]>5000)&(lbHist[1][:-1]<5800)]=0
            if plotDecor["customTitle"]=="":
                plt.title(("MC" if MCDict[currMode[0]] else "Data") + " sample of "+decayNameDict[currMode[0]] +(" with cuts:\n"+ "\n".join(map(str,TCut[0])) if isinstance(TCut[0],list) else  ((" with cuts:\n"+TCut[0]) if TCut[0] != "default" else "")))
            else:
                plt.title(plotDecor["customTitle"])
            plt.xscale(plotDecor["xscale"])
            plt.yscale(plotDecor["yscale"])
        plt.legend()
        plt.show()

    #plt.show()
    if outputHist:
        return lbHist

#now I want to make a 2d and potentially then 3d version of this... along with maybe an unbinned setting?
#easier to make a new function rather than trying to make the 1D one intelligent?
#no need for lhcbstyle, the whole thing is lhcb style, and theres no multi data plotting with a 2d histogram either.
#plotdecdict can be over general and include things that the 1D plot doesnt even allow cos its a dictionary


#2d version of above
def create2DHist(var=[None,None],TCut=("default",None),plotArgs=[None,None],blindSignalRegion=[True,True],plotDecor=plotDecDict(),debug=False,outputHist=True):
    if not isinstance(TCut,(tuple)):
        TCut=(TCut,TCut)
    TCut=(TCut,TCut)


    if var is None:
        print(dataDict[currMode[0]].keys())
        return None
    #var=var.astype(str)
    cut = [[],[]]
    dataToPlot=[np.empty(1),np.empty(1)]
    bins=[np.empty(1),np.empty(1)]
    for axis in [0,1]:
        showEventYields=[True,True]
        if blindSignalRegion[axis]:
            #cut+=LbBlindedSignalRegionCut
            #print(cut[axis-1])
            #print(axis)
            #print(["(dataToPlot[%i]<%i)|(dataToPlot[%i]>%i)"%(axis,LbBlindedRegion[0],axis,LbBlindedRegion[1])])
            cut[axis]+=["(dataToPlot[%i]<%i)|(dataToPlot[%i]>%i)"%(axis,LbBlindedRegion[0],axis,LbBlindedRegion[1])]
            cut[axis-1]+=["(dataToPlot[%i]<%i)|(dataToPlot[%i]>%i)"%(axis,LbBlindedRegion[0],axis,LbBlindedRegion[1])]
            showEventYields[axis]=False
        elif not MCDict[currMode[0]]:
            if not userCheckBlind(var[axis]):
                #cut+=LbBlindedSignalRegionCut
                #print(cut[axis-1])
                cut[axis]+=["(dataToPlot[%i]<%i)|(dataToPlot[%i]>%i)"%(axis,LbBlindedRegion[0],axis,LbBlindedRegion[1])]
                cut[axis-1]+=["(dataToPlot[%i]<%i)|(dataToPlot[%i]>%i)"%(axis,LbBlindedRegion[0],axis,LbBlindedRegion[1])]
                showEventYields[axis]=False


        dataToPlot[axis]=calculateVar(var[axis])
        if plotArgs[axis] is not None:
            if len(plotArgs[axis])==3:
                plotArgs[axis]=(plotArgs[axis][0],plotArgs[axis][1],plotArgs[axis][2]+1)
                bins[axis]=np.linspace(*(plotArgs[axis]))
                #cut+=["dataToPlot>plotArgs[0]","dataToPlot<plotArgs[1]"]
            elif len(plotArgs[axis])==2:
                pass
                #cut+=["dataToPlot>plotArgs[0]","dataToPlot<plotArgs[1]"]
                #need to replace with range
                bins[axis]="auto"
            else:
                bins[axis]=plotArgs[axis][0]
        else:
            bins[axis]=10

        if TCut[axis][1] is not None:
            cut[axis]+= TCut[axis][1]
        #cut+= TCut if TCut is not None else []
    #print(cut)

    for axis in [0,1]:
        cut[axis]= "("+")&(".join(map(str,cut[axis]))+")"
        #print(cut)
        cutStr=["",""]
        cutStr[axis]+=cut[axis]
        if debug:
            print(cut[axis])
        cut[axis]=evalCut(cut[axis],plotArgs,dataToPlot)


    plt.figure()
    # print(*[(dataToPlot[axis] if len(cut[axis])==0 else (dataToPlot[axis])[cut[axis]]) for axis in [0,1]],
    #     bins, plotDecor["density"],plotDecor["colormap"],plotDecor["alpha"])
    #print(cut)

    #print(cutStr)

    #print([len(dataToPlot[axis]) if len(cut[axis])==0 else len((dataToPlot[axis])[cut[axis]]) for axis in [0,1]])
    #print(dataToPlot[0] if len(cut[0])==0 else (dataToPlot[0])[cut[0]],dataToPlot[1] if len(cut[1])==0 else (dataToPlot[1])[cut[1]])
    #lb2dHist=plt.hist2d(*[dataToPlot[axis] if len(cut[axis])==0 else (dataToPlot[axis])[cut[axis]] for axis in [0,1]],
    #    bins=bins, density=plotDecor["density"],cmap=plotDecor["colormap"],alpha=plotDecor["alpha"])

    lb2dHist=plt.hist2d(dataToPlot[0] if len(cut[0])==0 else (dataToPlot[0])[cut[0]],dataToPlot[1] if len(cut[1])==0 else (dataToPlot[1])[cut[1]],
        bins=bins, density=plotDecor["density"],cmap=plotDecor["colormap"],alpha=plotDecor["alpha"])

    #only update labels etc if new plot
    #plt.zlabel("%s / (%.2g * %.2g %s*%s) "%(plotDecor["zlab"],lb2dHist[1][1]-lb2dHist[1][0],lb2dHist[2][1]-lb2dHist[2][0],plotDecor["xUnit"],plotDecor["yUnit"]))
    plt.ylabel("%s [%s], %i bins"%(var[1] if plotDecor["ylab"]=="default" else plotDecor["ylab"] ,plotDecor["yUnit"],lb2dHist[0].shape[1]))
    plt.xlabel("%s [%s], %i bins"%(var[0] if plotDecor["xlab"]=="default" else plotDecor["xlab"] ,plotDecor["xUnit"],lb2dHist[0].shape[0]))
    #lbHist[0][(lbHist[1][1:]>5000)&(lbHist[1][:-1]<5800)]=0
    if plotDecor["customTitle"]=="":
        plt.title(("MC" if MCDict[currMode[0]] else "Data") + " sample of "+decayNameDict[currMode[0]]+(" with cuts:\n"+ "\n".join(map(str,
                      TCut[0])) if isinstance(TCut[0][0],list) else  ((" with cuts:\n"+TCut[0][0]) if TCut[0][0] != "default" else "")))

        
    else:
        plt.title(plotDecor["customTitle"])
    plt.xscale(plotDecor["xscale"])
    plt.yscale(plotDecor["yscale"])
    #plt.zscale(plotDecor["zscale"])
    cb=plt.colorbar()
    cb.set_label("%s / (%.2g * %.2g %s*%s) "%(plotDecor["zlab"],lb2dHist[1][1]-lb2dHist[1][0],lb2dHist[2][1]-lb2dHist[2][0],plotDecor["xUnit"],plotDecor["yUnit"]))
    plt.show()

    if outputHist:
        return lb2dHist

    #print(bins)

# def LStarShow(var=None,TCut=None,plotArgs=None):
#     if var is None:
#         print(LStarData.keys())
#         return None
#     #var=var.astype(str)
#     cut = []
#     if blindSignalRegion:
#         cut+=LbBlindedSignalRegionCut
#     elif not MC:
#         if not userCheckBlind(var):
#             cut+=LbBlindedSignalRegionCut

#     if plotArgs is not None:
#         if len(plotArgs)==3:
#             bins=np.linspace(*plotArgs)
#             cut+=["LStarData[var]>plotArgs[0]","LStarData[var]<plotArgs[1]"]
#         if len(plotArgs)==2:
#             cut+=["LStarData[var]>plotArgs[0]","LStarData[var]<plotArgs[1]"]
#             bins="auto"
#         if len(plotArgs)==1:
#             bins=plotArgs[0]
#     else:
#         bins="auto"

#     if TCut is not None:
#         cut+= TCut

#     cut= "("+")&(".join(map(str,cut))+")"
#     cutString=cut
#     cut=evalCut(cut)

#     plt.figure()
#     LStarHist=plt.hist(LStarData[var][cut],bins=bins,fill=False,ec="b",label="Entries:%i"%len(LStarData[var][cut]))
#     plt.ylabel("Candidates / %.1f MeV "%(LStarHist[1][1]-LStarHist[1][0]))
#     plt.xlabel(r"$\Lambda$(1520) "+", %s [MeV], %i bins"%(var,len(LStarHist[0])))
#     plt.title(f"{var} for "+ "MC" if MC else "data" " sample of "+nameList[0] +(" with cuts:\n"+ "\n".join(map(str,TCut)) if TCut is not None else ""))
#     plt.legend()
#     plt.show()

#does multiple plots at once to see what truth matching does... a little redundant now
def LbTruthCompare(var=None,TCut=None,plotArgs=None,noCut=False,blindSignalRegion=True):
    if not MC:
        print("Truth Comparison only works for true data")
        return None
    if var is None:
        print(dataDict[currMode[0]].keys())
        return None
    #var=var.astype(str)
    cut = []
    if blindSignalRegion:
        cut+=LbBlindedSignalRegionCut
    elif not MC:
        if not userCheckBlind(var):
            cut+=LbBlindedSignalRegionCut
    if plotArgs is not None:
        if len(plotArgs)==3:
            bins=np.linspace(*plotArgs)
            cut+=["LbDataGet(var)>plotArgs[0]","LbDataGet(var)<plotArgs[1]"]
        if len(plotArgs)==2:
            cut+=["LbDataGet(var)>plotArgs[0]","LbDataGet(var)<plotArgs[1]"]
            bins="auto"
        if len(plotArgs)==1:
            bins=plotArgs[0]
    else:
        bins="auto"

    
    

    noCutCut= cut.copy()
    truthCutList=noCutCut.copy()
    if TCut is not None:
        cut+= TCut
    truthwCutsList=cut.copy()
    cut= "("+")&(".join(map(str,cut))+")"
    cutString=cut
    print(cutString)
    cut=evalCut(cut)
    
    plt.figure()
    lbHist=plt.hist(LbDataGet(var) if cutString == "()" else LbDataGet(var)[cut],bins=bins,color="b",label="Cuts\nEntries:%i"%len(LbDataGet(var) if cutString=="()" else LbDataGet(var)[cut]))
    if noCut:
        noCutCut="("+")&(".join(map(str,noCutCut))+")"
        noCutCutString=noCutCut
        print(noCutCutString)
        noCutCut=evalCut(noCutCutString)

        lbHistNoCut=plt.hist(LbDataGet(var) if noCutCutString=="()" else LbDataGet(var)[noCutCut],bins=lbHist[1],color="k",zorder=-2,label="No Cuts\nEntries:%i"%len(LbDataGet(var) if noCutCutString=="()" else LbDataGet(var)[noCutCut]))

        truthCutList+=truthMatching(["10","50","60"])
        cut= "("+")&(".join(map(str,truthCutList))+")"
        cutString=cut
        cut=evalCut(cut)
        lbHistTruthNoCut=plt.hist(LbDataGet(var) if cutString=="()" else LbDataGet(var)[cut],bins=lbHist[1],zorder=-1,color="r",label="Truth matched - no cuts\nEntries:%i"%len(LbDataGet(var) if cutString=="()" else LbDataGet(var)[cut]))

    truthwCutsList+=truthMatching(["10","50","60"])
    cut= "("+")&(".join(map(str,truthwCutsList))+")"
    cutString=cut
    cut=evalCut(cut)
    lbHistTruthCut=plt.hist(LbDataGet(var) if cutString=="()" else LbDataGet(var)[cut],bins=lbHist[1],color="g",label="Truth matched - w cuts\nEntries:%i"%len(LbDataGet(var) if cutString=="()" else LbDataGet(var)[cut]))

    plt.ylabel("Candidates / %.1f MeV "%(lbHist[1][1]-lbHist[1][0]))
    plt.xlabel(r"$\Lambda_b^0$ "+", %s [MeV], %i bins"%(var,len(lbHist[0])))
    plt.title(f"{var} for "+ "MC" if MC else "data" " sample of "+decayNameDict[currMode[0]]+(" with cuts:\n"+ "\n".join(map(str,TCut)) if TCut is not None else ""))
    plt.legend()
    plt.show()

#performs truthmatching by motherDaughter and also by BKGCAT numbering
#by default will use motherDaughter Matching
def truthMatching(bkgcat=None,useMotherDaughterMatching=True):
    #can ask for a certain bckcat, if none asked, simply each particle and their mothers matched
    #also or instead?? I will start with also
    if useMotherDaughterMatching:
        TMCut=[f"LbDataGet('{name}_TRUEID')==LbDataGet('{name}_ID')"for name in targetParticles[1:]]+[f"LbDataGet('{name}_MC_MOTHER_ID')==LbDataGet('LStar_ID')" for name in targetParticles[1:3]]+[f"LbDataGet('{name}_MC_MOTHER_ID')==LbDataGet('Lb_ID')" for name in targetParticles[3:]]
    else:
        TMCut=[]
    #"("+")&(".join(map(str,cut))+")"
    if bkgcat is not None:
        TMCut+=["(LbDataGet('Lb_BKGCAT')=="+ ")|(LbDataGet('Lb_BKGCAT')==".join(map(str,bkgcat))+")"]

    return TMCut




#some random useful variables that came in handy for prev work...
##print Lb plots##
fl32="float32"
# should write some basic unit tests?
rpKBGSelec=["LbDataGet('Lb_M012')>2320",
                "LbDataGet('Lb_M013_Subst01_Kp~2pK')>2320",
                "LbDataGet('Lb_M123')<5200",
                "LbDataGet('Lb_M023_Subst0_p2K')<5200",
                "np.abs(LbDataGet('Lb_M01_Subst0_p2K')-1020)>12",
                "np.abs(LbDataGet('Lb_M12_Subst2_mu2pi')-1865)>20",
                "np.abs(LbDataGet('Lb_M13_Subst3_mu2pi')-1865)>20",
                "np.abs(LbDataGet('Lb_M12_Subst12_Kmu2muK')-3097)>35"
                ]

rpKPID = ["LbDataGet('p_P')>10e3","LbDataGet('K_P')>2e3","LbDataGet('L1_P')>3e3","LbDataGet('L2_P')>3e3",
"LbDataGet('p_PT')>1e3","LbDataGet('K_PT')>250","LbDataGet('L1_PT')>800","LbDataGet('L2_PT')>800",
"LbDataGet('p_P')<150e3","LbDataGet('K_P')<150e3","LbDataGet('L1_P')<150e3","LbDataGet('L2_P')<150e3",
"LbDataGet('p_MC15TuneV1_ProbNNp')>0.3","LbDataGet('p_MC15TuneV1_ProbNNk')<0.8","LbDataGet('p_MC15TuneV1_ProbNNpi')<0.7",
"LbDataGet('K_MC15TuneV1_ProbNNk')>0.2","LbDataGet('K_MC15TuneV1_ProbNNp')<0.8",
"LbDataGet('L1_MC15TuneV1_ProbNNmu')>0.1","LbDataGet('L2_MC15TuneV1_ProbNNmu')>0.1"]

L1520Cut=["LbDataGet('Lb_M01')>1500","LbDataGet('Lb_M01')<1540"]
L1520Cut1550=["LbDataGet('Lb_M01')>1500","LbDataGet('Lb_M01')<1550"]

rpKRareRegion=["LbDataGet('Lb_M23')**2>0.1e6","LbDataGet('Lb_M23')**2<6e6"]
JPsiCut=["(LbDataGet('Lb_M23')**2<8e6)|(LbDataGet('Lb_M23')**2>11e6)"]
psi2SCut=["(LbDataGet('Lb_M23')**2<12.5e6)|(LbDataGet('Lb_M23')**2>15e6)"]

LbSignalRegionCut=["(LbDataGet('Lb_M')<5620-45)|(LbDataGet('Lb_M')>5620+45)"]
LbPlotRegion=["(LbDataGet('Lb_M')>5300)","(LbDataGet('Lb_M')<5950)"]

LbBlindedSignalRegionCut=["(LbDataGet('Lb_M')<%i)|(LbDataGet('Lb_M')>%i)"%(LbBlindedRegion[0],LbBlindedRegion[1])]

selfConstructPK='(("p_PE"+"K_PE")**2-("p_PX"+"K_PX")**2-("p_PY"+"K_PY")**2-("p_PZ"+"K_PZ")**2)**0.5'

#BestPV=np.array([[LbDataGet("PVX")[i][0],LbDataGet("PVY")[i][0],LbDataGet("PVZ")[i][0]] for i in range(len(LbDataGet("PVZ")))])
#BestPV=np.array([LbDataGet("PVZ"),LbDataGet("PVY"),LbDataGet("PVZ")]).T