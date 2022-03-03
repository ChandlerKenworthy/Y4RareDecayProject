#cuts for any pre-selection applied after trigger requirements
# will include:
# - L1520 region cut
# - PID cuts
# - jpsi veto
# - potential lambdac veto
# - acceptance cuts

###opening angle calc
numToParticleOpeningAng={"0":"p","1":"K","2":"L1","3":"L2",
                "p":"p","K":"K","L1":"L1","L2":"L2",
                "LStar":"LStar","JPs":"JPs","Lb":"Lb"} #second line to convert something already converted, a


#inputs are in format ["L2/p/K/L1","L2/p/K/L1","L2/p/K/L1"], massComb can either be "012" or "pKL1", as can input/output
#outputs must be sensible (pi,mu,e,K,p)
def openingAngleCosineCalc(inputParticles):
    
    inputParticlesTemp=[]
    for a in range(len(inputParticles)):
        inputParticlesTemp.append(numToParticleOpeningAng[inputParticles[a]])
    inputParticles=inputParticlesTemp
        
    

    ppSum=["LbDataGet('"+"_PX')*LbDataGet('".join(map(str,inputParticles))+"_PX')",
    "LbDataGet('"+"_PY')*LbDataGet('".join(map(str,inputParticles))+"_PY')",
    "LbDataGet('"+"_PZ')*LbDataGet('".join(map(str,inputParticles))+"_PZ')"]
    
    dotProd='+'.join(map(str,ppSum))
    
    cosTheta=f"({dotProd})/("+"LbDataGet('"+"_P')*LbDataGet('".join(map(str,inputParticles))+"_P'))"

    return cosTheta


def openingAngleCalc(inputParticles):
    
    inputParticlesTemp=[]
    for a in range(len(inputParticles)):
        inputParticlesTemp.append(numToParticleOpeningAng[inputParticles[a]])
    inputParticles=inputParticlesTemp
        
    

    ppSum=["LbDataGet('"+"_PX')*LbDataGet('".join(map(str,inputParticles))+"_PX')",
    "LbDataGet('"+"_PY')*LbDataGet('".join(map(str,inputParticles))+"_PY')",
    "LbDataGet('"+"_PZ')*LbDataGet('".join(map(str,inputParticles))+"_PZ')"]
    
    dotProd='+'.join(map(str,ppSum))
    
    cosTheta=f"({dotProd})/("+"LbDataGet('"+"_P')*LbDataGet('".join(map(str,inputParticles))+"_P'))"

    return f"(np.arccos({cosTheta}))"

#subMassCalculator
numToParticle={"0":"p","1":"K","2":"L1","3":"L2",
                "p":"p","K":"K","L1":"L1","L2":"L2"} #second line to convert something already converted, a

inputToOutputParticles={"p":"p","K":"K","L1":"mu","L2":"e"} #converts from internal meanings (pKL1L2) to actual fs paticles

particleMasses={"mu":105.65837121,"e":0.51100003,"p":938.27208207,"K":493.6770041,"pi":139.57061}

#inputs are in format ["L2/p/K/L1","L2/p/K/L1","L2/p/K/L1"], massComb can either be "012" or "pKL1", as can input/output
#outputs must be sensible (pi,mu,e,K,p)
def subMassCalc(massComb=None,inputParticles=None,outputParticles=None):
    if massComb is None:
        return None
    elif outputParticles is None or inputParticles is None:
        inputParticles=[numToParticle[char] for char in massComb]
        outputParticles=[inputToOutputParticles[a] for a in inputParticles]
    elif len(inputParticles)!=len(outputParticles):
        return None

    massCombTemp=[]
    for a in range(len(massComb)):
        massCombTemp.append(numToParticle[massComb[a]])
    massComb=massCombTemp
    for a in range(len(inputParticles)):
        inputParticles[a]=numToParticle[inputParticles[a]]

    newEnergies={inputParticles[i]:f"({particleMasses[outputParticles[i]]**2:.8f}+LbDataGet('{inputParticles[i]}_P')**2)**0.5" for i in range(len(inputParticles))}
    for a in massComb:
        if a not in newEnergies.keys():
            newEnergies[a]=f"LbDataGet('{a}_PE')"

    PSum=["LbDataGet('"+"_PX')+LbDataGet('".join(map(str,massComb))+"_PX')",
    "LbDataGet('"+"_PY')+LbDataGet('".join(map(str,massComb))+"_PY')",
    "LbDataGet('"+"_PZ')+LbDataGet('".join(map(str,massComb))+"_PZ')"]

    newInvMass=(f"((({'+'.join(map(str,[newEnergies[a] for a in massComb]))})**2"
        f"-({PSum[0]})**2"
        f"-({PSum[1]})**2"
        f"-({PSum[2]})**2)**0.5)")

    return newInvMass


##Trigger cuts
## cuts for the trigger lines. Investigation completed in ~/tuples/triggerTests.

triggerL0mandL0eOLD="((LbDataGet('L1_L0MuonDecision_TOS'))|(LbDataGet('L2_L0ElectronDecision_TOS')))"
#cut for L0m or L0e, determined to be about 67% efficient on TM L1520mue simulation

triggerL0mandL0eandL0TISOLD="((LbDataGet('L1_L0MuonDecision_TOS'))|(LbDataGet('L2_L0ElectronDecision_TOS'))|(LbDataGet('Lb_L0Global_TIS')))"
#also includong global L0 TIS increased efficiency by 10%, but could lead to issues when determining trigger efficiency later
triggerL0m="(LbDataGet('L1_L0MuonDecision_TOS'))"
#new trigger, L0Muon only, as L0electron trigger thresholds often vary
# and using this would require intensive correction studies that vary within run periods
# as well as between runs

triggerHLT1TrkMVAOLD="(LbDataGet('Lb_Hlt1TrackMVADecision_TOS'))"
#encompassing hlt1 trigger constant between normalisation and control channels...95% efficient
triggerHLT1MVAandMuon="((LbDataGet('Lb_Hlt1TrackMVADecision_TOS'))|(LbDataGet('Lb_Hlt1TrackMuonDecision_TOS')))"#new trigger, additional M
#new trigger, with additional Track Muon decision, slight increase in stats

triggerHLT2Topo234OLD="(LbDataGet('Lb_Hlt2Topo2BodyDecision_TOS')|LbDataGet('Lb_Hlt2Topo3BodyDecision_TOS')|LbDataGet('Lb_Hlt2Topo4BodyDecision_TOS'))"
#95% efficient on TM MC, very general, works for norm and signal
triggerHLT2Topo234andMu234="(LbDataGet('Lb_Hlt2Topo2BodyDecision_TOS')|LbDataGet('Lb_Hlt2Topo3BodyDecision_TOS')|LbDataGet('Lb_Hlt2Topo4BodyDecision_TOS')|LbDataGet('Lb_Hlt2TopoMu2BodyDecision_TOS')|LbDataGet('Lb_Hlt2TopoMu3BodyDecision_TOS')|LbDataGet('Lb_Hlt2TopoMu4BodyDecision_TOS'))"
#new trigger, identical for signal and normalisation mode

triggerCombinedSelec=triggerL0m+"&"+triggerHLT1MVAandMuon+"&"+triggerHLT2Topo234andMu234
#updated with newTriggers

#useful for constructing cuts
#in OS sample, eMinusCut and MuPlusCut are the same cut, but separated for convienience when using SS case
eMinusCut="(((LbDataGet('L2_ID')>0)&(LbDataGet('p_ID')>0))|((LbDataGet('L2_ID')<0)&(LbDataGet('p_ID')<0)))"
muMinusCut="(((LbDataGet('L1_ID')>0)&(LbDataGet('p_ID')>0))|((LbDataGet('L1_ID')<0)&(LbDataGet('p_ID')<0)))"
ePlusCut="(((LbDataGet('L2_ID')<0)&(LbDataGet('p_ID')>0))|((LbDataGet('L2_ID')>0)&(LbDataGet('p_ID')<0)))"
muPlusCut="(((LbDataGet('L1_ID')<0)&(LbDataGet('p_ID')>0))|((LbDataGet('L1_ID')>0)&(LbDataGet('p_ID')<0)))"


L1520RegionCut="((LbDataGet('LStar_M')>1448)&(LbDataGet('LStar_M')<1591))"


vetoPhiRegion="((LbDataGet('Lb_M01_Subst0_p2K')<1019.461-12)|(LbDataGet('Lb_M01_Subst0_p2K')>1019.461+12))"

#DsVetoes
vetoDs_KKe=f"({subMassCalc('013',inputParticles=['p'],outputParticles=['K'])}>2000)"
vetoDs_KKmu=f"({subMassCalc('012',inputParticles=['p'],outputParticles=['K'])}>2000)"
vetoDs_both=f"({vetoDs_KKe}&{vetoDs_KKmu})"

#Lambda c vetoes

#unless I include the extra bits, then I would be able to apply an and to the last 
vetoLambdac_pKe=f"((({subMassCalc('013')}>2320)&{ePlusCut})|{muPlusCut})"
vetoLambdac_pKmu=f"((({subMassCalc('012')}>2320)&{muPlusCut})|{ePlusCut})"
vetoLambdac_both=f"({vetoLambdac_pKe}&{vetoLambdac_pKmu})"

#also pK swap vetoes WIP ADD A EMINUS AND MUMINUS/EPLUS MUPLUS GENERAL CUT TO REUSE 
vetoLambdac_pKe_pKswap=f"((({subMassCalc('013',inputParticles=['p','K'],outputParticles=['K','p'])}>2320)&{eMinusCut})|{muMinusCut})"
vetoLambdac_pKmu_pKswap=f"((({subMassCalc('012',inputParticles=['p','K'],outputParticles=['K','p'])}>2320)&{muMinusCut})|{eMinusCut})"
vetoLambdac_both_pKswap=f"({vetoLambdac_pKe_pKswap}&{vetoLambdac_pKmu_pKswap})"

#pKpi/KKpi vetoes, not expecting to use all of them. potentially just the mu ones

vetoLcRes_pKpi_eplus=f"(((({subMassCalc('013',inputParticles=['L2'],outputParticles=['pi'])}>2286.46+25)|({subMassCalc('013',inputParticles=['L2'],outputParticles=['pi'])}<2286.46-25))&{ePlusCut})|{muPlusCut})"

vetoLcRes_pKpi_muplus=f"(((({subMassCalc('012',inputParticles=['L1'],outputParticles=['pi'])}>2286.46+25)|({subMassCalc('012',inputParticles=['L1'],outputParticles=['pi'])}<2286.46-25))&{muPlusCut})|{ePlusCut})"

vetoDsRes_KKpi_e=f"((({subMassCalc('013',inputParticles=['p','L2'],outputParticles=['K','pi'])}>1968.35+25))|(({subMassCalc('013',inputParticles=['p','L2'],outputParticles=['K','pi'])}<1968.35-25)))"

vetoDsRes_KKpi_mu=f"((({subMassCalc('012',inputParticles=['p','L1'],outputParticles=['K','pi'])}>1968.35+25))|(({subMassCalc('012',inputParticles=['p','L1'],outputParticles=['K','pi'])}<1968.35-25)))"


#charmonium vetoes
#sqrt 9 - sqrt 10.1
vetoJPsiRes_q = "((LbDataGet('Lb_M23')>3178.05)|(LbDataGet('Lb_M23')<3000))"

#sqrt 13 - sqrt 14
vetoPsi2SRes_q = "((LbDataGet('Lb_M23')>3741.66)|(LbDataGet('Lb_M23')<3605.55))"


#D0 vetoes, written for both K-e+ and K-mu+

vetoD0Res_Kpi_eplus = (f"(((({subMassCalc('13',inputParticles=['L2'],outputParticles=['pi'])}>1865+20)|({subMassCalc('13',inputParticles=['L2'],outputParticles=['pi'])}<1865-20))"
                        f"&{ePlusCut})|"
                        f"{muPlusCut})")

vetoD0Res_Kpi_muplus = (f"(((({subMassCalc('12',inputParticles=['L1'],outputParticles=['pi'])}>1865+20)|({subMassCalc('12',inputParticles=['L1'],outputParticles=['pi'])}<1865-20))"
                        f"&{muPlusCut})|"
                        f"{ePlusCut})")

vetoD0Res_Kpi_Kell_both=f"({vetoD0Res_Kpi_eplus}&{vetoD0Res_Kpi_muplus})"


#also D0 vetoes with p2K ell2pi
vetoD0Res_Kpi_pmu = (f"(((({subMassCalc('02',inputParticles=['p','L1'],outputParticles=['K','pi'])}>1865+20)|({subMassCalc('02',inputParticles=['p','L1'],outputParticles=['K','pi'])}<1865-20))"
                      f"&{muMinusCut})|"
                       f"{eMinusCut})")

vetoD0Res_Kpi_pe = (f"(((({subMassCalc('03',inputParticles=['p','L2'],outputParticles=['K','pi'])}>1865+20)|({subMassCalc('03',inputParticles=['p','L2'],outputParticles=['K','pi'])}<1865-20))"
                    f"&{eMinusCut})|"
                    f"{muMinusCut})")

vetoD0Res_Kpi_pell_both=f"({vetoD0Res_Kpi_pmu}&{vetoD0Res_Kpi_pe})"

# jpsi vetoes for kmu/e: mu looks needed, maybe not e. e should be widened if used

vetoJpsiVeto_mumu_Kmu = (f"(((({subMassCalc('12',inputParticles=['K'],outputParticles=['mu'])}>3097+35)|({subMassCalc('12',inputParticles=['K'],outputParticles=['mu'])}<3097-35))"
                        f"&{muPlusCut})|"
                        f"{ePlusCut})")


vetoJpsiVeto_ee_Ke = (f"(((({subMassCalc('13',inputParticles=['K'],outputParticles=['e'])}>3097+35)|({subMassCalc('13',inputParticles=['K'],outputParticles=['e'])}<3097-70))"
                      f"&{ePlusCut})|"
                      f"{muPlusCut})")


###analogous jpsi vetoes for pmu/e: shouldn't be needed, no obvious peak.

vetoJpsiVeto_mumu_pmu = (f"(((({subMassCalc('02',inputParticles=['p'],outputParticles=['mu'])}>3097+35)|({subMassCalc('02',inputParticles=['p'],outputParticles=['mu'])}<3097-35))"
                        f"&{muMinusCut})|"
                        f"{eMinusCut})")


vetoJpsiVeto_ee_pe = (f"(((({subMassCalc('03',inputParticles=['p'],outputParticles=['e'])}>3097+35)|({subMassCalc('03',inputParticles=['p'],outputParticles=['e'])}<3097-70))"
                      f"&{eMinusCut})|"
                      f"{muMinusCut})") 


#input multicandidate selection type strings here...


#clone tracks strings, using the cosine form to save overall calculation time
vetoCloneTracks_ellh = (f"(({openingAngleCosineCalc('02')}<np.cos(1e-3))&({openingAngleCosineCalc('03')}<np.cos(1e-3))"
                        f"&({openingAngleCosineCalc('12')}<np.cos(1e-3))&({openingAngleCosineCalc('13')}<np.cos(1e-3)))")

vetoCloneTracks_hh = f"({openingAngleCosineCalc('01')}<np.cos(1e-3))"

vetoCloneTracks_ellell = f"({openingAngleCosineCalc('23')}<np.cos(1e-3))"

#not using ellell for now

#input pkswap and ep and mup D0 resonance veto, consider emu D0 veto
####
#combine current preselec progress in collision data and MC cut files


#cuts only to be applied to data

# blinding cut and any ...


##### Collision Data Cuts

blindDataRegionCut="((LbDataGet('Lb_M')>5800)|(LbDataGet('Lb_M')<5200))"

currPreselecDataPreFormat = ("{vetoPhiRegion}&{vetoDs_both}&{vetoLambdac_both}&"
                                "{vetoJPsiRes_q}&{vetoD0Res_Kpi_eplus}&"
                                "{vetoJpsiVeto_mumu_Kmu}&{vetoD0Res_Kpi_pe}&"
                                "{vetoCloneTracks_ellh}&{vetoCloneTracks_hh}")
currPreselecData = (f"({vetoPhiRegion}&{vetoDs_both}&{vetoLambdac_both}&"
                    f"{vetoJPsiRes_q}&{vetoD0Res_Kpi_eplus}&"
                    f"{vetoJpsiVeto_mumu_Kmu}&{vetoD0Res_Kpi_pe}&"
                    f"{vetoCloneTracks_ellh}&{vetoCloneTracks_hh})")

### this is actual cut for data

######## MonteCarlo Cuts

#File to hold cuts specific to the L1520mue simulation, mainly corresponding to truth matching/reweighting cuts
targetParticles=["Lb","p","K","L1","L2"]


def stringListToCut(stringInput):
    return "("+")&(".join(map(str,stringInput))+")"

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

truthMatchCat1050=stringListToCut(truthMatching(bkgcat=[10,50],useMotherDaughterMatching=False))
truthMatchIDMatch=stringListToCut(truthMatching())



currPreselecMCPreFormat = "{currPreselecDataPreFormat}&{triggerCombinedSelec}&{L1520RegionCut}&{truthMatchCat1050}"
currPreselecMC = f"({currPreselecData}&{triggerCombinedSelec}&{L1520RegionCut}&{truthMatchCat1050})"

#actual cut for MC