from dataflow import Flow
from datetime import datetime, date

###########################################
### ONLY CHANGE THESE VALUES AT RUNTIME ###
###########################################

sname = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/MC/2016MD/fullSampleOct2021/job207-CombDVntuple-15314000-MC2016MD_Full-pKmue-MC.root"
fname = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/realData/2016MD/halfSampleOct2021/blindedTriggeredL1520Selec-collision-firstHalf2016MD-pKmue_Fullv9.root"
version = '6.0.0'
features = list(dict.fromkeys([
    'Lb_PT', 'Lb_IPCHI2_OWNPV', 'Lb_ENDVERTEX_CHI2', 'Lb_HOP',
    'L2_IPCHI2_OWNPV', 'L1_IPCHI2_OWNPV', 'LStar_ORIVX_CHI2', 'LStar_DIRA_OWNPV',
    'JPs_FD_ORIVX', 'p_PZ', 'p_P', 'K_PZ', 'K_P', 'K_PY', 'K_PT', 'p_PY', 'p_PT',
    'K_P', 'Lb_L1_cmult_0.5TrkISO', 'Lb_L2_cmult_0.5TrkISO', 'Lb_p_cmult_0.5TrkISO',
    'Lb_K_cmult_0.5TrkISO', 'Lb_L1_cc_asy_PT_0.5ConeISO', 'Lb_L2_cc_asy_PT_0.5ConeISO',
    'Lb_p_cc_asy_PT_0.5ConeISO', 'Lb_K_cc_asy_PT_0.5ConeISO', 'Lb_P', 'Lb_PT',
    'Lb_MINIPCHI2', 'Lb_DIRA_OWNPV', 'p_ETA', 'K_ETA', 'L1_ETA', 'L2_ETA',
    'Lb_IP01', 'Lb_IP23', 'JPs_DIRA_TOPPV', 'Lb_IP_OWNPV', 'p_TRACK_VeloCHI2NDOF',
    'Lb_TAUERR'
]))

new_features = {
    'ABS_ARTANH_PZ_P': "np.abs(np.arctanh( p_PZ / p_P )-np.arctanh( K_PZ / K_P ))",
    'MAG_ARSINH_PY_PT': "np.sqrt((np.arcsinh( K_PY / K_PT )-np.arcsinh( p_PY / p_PT ))**2+(np.arcsinh( K_P / K_PT )-np.arcsinh( p_P / p_PT ))**2)",
    'SUM_CONE_ISO': " Lb_L1_cc_asy_PT_0.5ConeISO + Lb_L2_cc_asy_PT_0.5ConeISO + Lb_p_cc_asy_PT_0.5ConeISO + Lb_K_cc_asy_PT_0.5ConeISO ",
    'SUM_LIPCHI2': " L2_IPCHI2_OWNPV + L1_IPCHI2_OWNPV ",
    #'LB_TRACKISO': " Lb_L1_cmult_0.5TrkISO + Lb_L2_cmult_0.5TrkISO + Lb_p_cmult_0.5TrkISO + Lb_K_cmult_0.5TrkISO ",
    'LN_COS_THETA': "np.log(1-np.cos(np.arcsin( Lb_PT / Lb_P )))",
    'LN_LB_MINIPCHI2': "np.log( Lb_MINIPCHI2 )",
    'LN_COS_LBDIRA': "np.log(1-np.cos( Lb_DIRA_OWNPV ))",
    'LN_JPs_DIRA_TOPPV': "np.log( JPs_DIRA_TOPPV )"
}

apply_preselection = True
event_ratio = 1

###########################################
### ONLY CHANGE THESE VALUES AT RUNTIME ###
###########################################


data = Flow(features, sname, fname)
data.set_simulated_preselection("(((( Lb_M01_Subst0_p2K <1019.461-12)|( Lb_M01_Subst0_p2K >1019.461+12))&((((((243716.98437715+ p_P **2)**0.5+ K_PE + L2_PE )**2-( p_PX + K_PX + L2_PX )**2-( p_PY + K_PY + L2_PY )**2-( p_PZ + K_PZ + L2_PZ )**2)**0.5)>2000)&(((((243716.98437715+ p_P **2)**0.5+ K_PE + L1_PE )**2-( p_PX + K_PX + L1_PX )**2-( p_PY + K_PY + L1_PY )**2-( p_PZ + K_PZ + L1_PZ )**2)**0.5)>2000))&((((((((880354.49999197+ p_P **2)**0.5+(243716.98437715+ K_P **2)**0.5+(0.26112103+ L2_P **2)**0.5)**2-( p_PX + K_PX + L2_PX )**2-( p_PY + K_PY + L2_PY )**2-( p_PZ + K_PZ + L2_PZ )**2)**0.5)>2320)&((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))|((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))&(((((((880354.49999197+ p_P **2)**0.5+(243716.98437715+ K_P **2)**0.5+(11163.69140675+ L1_P **2)**0.5)**2-( p_PX + K_PX + L1_PX )**2-( p_PY + K_PY + L1_PY )**2-( p_PZ + K_PZ + L1_PZ )**2)**0.5)>2320)&((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))|((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0)))))&(( Lb_M23 >3178.05)|( Lb_M23 <3000))&((((((( K_PE +(19479.95517577+ L2_P **2)**0.5)**2-( K_PX + L2_PX )**2-( K_PY + L2_PY )**2-( K_PZ + L2_PZ )**2)**0.5)>1865+20)|(((( K_PE +(19479.95517577+ L2_P **2)**0.5)**2-( K_PX + L2_PX )**2-( K_PY + L2_PY )**2-( K_PZ + L2_PZ )**2)**0.5)<1865-20))&((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))|((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))&((((((((11163.69140675+ K_P **2)**0.5+ L1_PE )**2-( K_PX + L1_PX )**2-( K_PY + L1_PY )**2-( K_PZ + L1_PZ )**2)**0.5)>3097+35)|(((((11163.69140675+ K_P **2)**0.5+ L1_PE )**2-( K_PX + L1_PX )**2-( K_PY + L1_PY )**2-( K_PZ + L1_PZ )**2)**0.5)<3097-35))&((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))|((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))&((((((((243716.98437715+ p_P **2)**0.5+(19479.95517577+ L2_P **2)**0.5)**2-( p_PX + L2_PX )**2-( p_PY + L2_PY )**2-( p_PZ + L2_PZ )**2)**0.5)>1865+20)|(((((243716.98437715+ p_P **2)**0.5+(19479.95517577+ L2_P **2)**0.5)**2-( p_PX + L2_PX )**2-( p_PY + L2_PY )**2-( p_PZ + L2_PZ )**2)**0.5)<1865-20))&((( L2_ID >0)&( p_ID >0))|(( L2_ID <0)&( p_ID <0))))|((( L1_ID >0)&( p_ID >0))|(( L1_ID <0)&( p_ID <0))))&((( p_PX * L1_PX + p_PY * L1_PY + p_PZ * L1_PZ )/( p_P * L1_P )<np.cos(1e-3))&(( p_PX * L2_PX + p_PY * L2_PY + p_PZ * L2_PZ )/( p_P * L2_P )<np.cos(1e-3))&(( K_PX * L1_PX + K_PY * L1_PY + K_PZ * L1_PZ )/( K_P * L1_P )<np.cos(1e-3))&(( K_PX * L2_PX + K_PY * L2_PY + K_PZ * L2_PZ )/( K_P * L2_P )<np.cos(1e-3)))&(( p_PX * K_PX + p_PY * K_PY + p_PZ * K_PZ )/( p_P * K_P )<np.cos(1e-3)))&( L1_L0MuonDecision_TOS )&(( Lb_Hlt1TrackMVADecision_TOS )|( Lb_Hlt1TrackMuonDecision_TOS ))&( Lb_Hlt2Topo2BodyDecision_TOS | Lb_Hlt2Topo3BodyDecision_TOS | Lb_Hlt2Topo4BodyDecision_TOS | Lb_Hlt2TopoMu2BodyDecision_TOS | Lb_Hlt2TopoMu3BodyDecision_TOS | Lb_Hlt2TopoMu4BodyDecision_TOS )&(( LStar_M >1448)&( LStar_M <1591))&(( Lb_BKGCAT ==10)|( Lb_BKGCAT ==50)))")
data.set_real_preselection("((( Lb_M01_Subst0_p2K <1019.461-12)|( Lb_M01_Subst0_p2K >1019.461+12))&((((((243716.98437715+ p_P **2)**0.5+ K_PE + L2_PE )**2-( p_PX + K_PX + L2_PX )**2-( p_PY + K_PY + L2_PY )**2-( p_PZ + K_PZ + L2_PZ )**2)**0.5)>2000)&(((((243716.98437715+ p_P **2)**0.5+ K_PE + L1_PE )**2-( p_PX + K_PX + L1_PX )**2-( p_PY + K_PY + L1_PY )**2-( p_PZ + K_PZ + L1_PZ )**2)**0.5)>2000))&((((((((880354.49999197+ p_P **2)**0.5+(243716.98437715+ K_P **2)**0.5+(0.26112103+ L2_P **2)**0.5)**2-( p_PX + K_PX + L2_PX )**2-( p_PY + K_PY + L2_PY )**2-( p_PZ + K_PZ + L2_PZ )**2)**0.5)>2320)&((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))|((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))&(((((((880354.49999197+ p_P **2)**0.5+(243716.98437715+ K_P **2)**0.5+(11163.69140675+ L1_P **2)**0.5)**2-( p_PX + K_PX + L1_PX )**2-( p_PY + K_PY + L1_PY )**2-( p_PZ + K_PZ + L1_PZ )**2)**0.5)>2320)&((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))|((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0)))))&(( Lb_M23 >3178.05)|( Lb_M23 <3000))&((((((( K_PE +(19479.95517577+ L2_P **2)**0.5)**2-( K_PX + L2_PX )**2-( K_PY + L2_PY )**2-( K_PZ + L2_PZ )**2)**0.5)>1865+20)|(((( K_PE +(19479.95517577+ L2_P **2)**0.5)**2-( K_PX + L2_PX )**2-( K_PY + L2_PY )**2-( K_PZ + L2_PZ )**2)**0.5)<1865-20))&((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))|((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))&((((((((11163.69140675+ K_P **2)**0.5+ L1_PE )**2-( K_PX + L1_PX )**2-( K_PY + L1_PY )**2-( K_PZ + L1_PZ )**2)**0.5)>3097+35)|(((((11163.69140675+ K_P **2)**0.5+ L1_PE )**2-( K_PX + L1_PX )**2-( K_PY + L1_PY )**2-( K_PZ + L1_PZ )**2)**0.5)<3097-35))&((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))|((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))&((((((((243716.98437715+ p_P **2)**0.5+(19479.95517577+ L2_P **2)**0.5)**2-( p_PX + L2_PX )**2-( p_PY + L2_PY )**2-( p_PZ + L2_PZ )**2)**0.5)>1865+20)|(((((243716.98437715+ p_P **2)**0.5+(19479.95517577+ L2_P **2)**0.5)**2-( p_PX + L2_PX )**2-( p_PY + L2_PY )**2-( p_PZ + L2_PZ )**2)**0.5)<1865-20))&((( L2_ID >0)&( p_ID >0))|(( L2_ID <0)&( p_ID <0))))|((( L1_ID >0)&( p_ID >0))|(( L1_ID <0)&( p_ID <0))))&((( p_PX * L1_PX + p_PY * L1_PY + p_PZ * L1_PZ )/( p_P * L1_P )<np.cos(1e-3))&(( p_PX * L2_PX + p_PY * L2_PY + p_PZ * L2_PZ )/( p_P * L2_P )<np.cos(1e-3))&(( K_PX * L1_PX + K_PY * L1_PY + K_PZ * L1_PZ )/( K_P * L1_P )<np.cos(1e-3))&(( K_PX * L2_PX + K_PY * L2_PY + K_PZ * L2_PZ )/( K_P * L2_P )<np.cos(1e-3)))&(( p_PX * K_PX + p_PY * K_PY + p_PZ * K_PZ )/( p_P * K_P )<np.cos(1e-3)))")

if apply_preselection:
    data.apply_preselection(keep_regions=[[4500, 5200], [5800, 6500]])
data.combine_data()

for key, value in new_features.items():
    data.generate_feature(key, value, verbose=0)

cols = list(data.get_combined_data().columns)
feats = list(dict.fromkeys([
    'ABS_ARTANH_PZ_P', 'MAG_ARSINH_PY_PT', 'SUM_CONE_ISO', 'LN_COS_THETA',
    'SUM_LIPCHI2', 'LB_TRACKISO', 'JPs_FD_ORIVX', 'LStar_DIRA_OWNPV',
    'LStar_ORIVX_CHI2', 'JPs_DOCA12', 'Lb_HOP', 'Lb_ENDVERTEX_CHI2',
    'Lb_IPCHI2_OWNPV', 'Lb_PT', 'LN_LB_MINIPCHI2', 'p_ETA', 'K_ETA',
    'L1_ETA', 'L2_ETA', 'Lb_IP01', 'Lb_IP23', 'LN_COS_LBDIRA', 
    'LN_JPs_DIRA_TOPPV', 'Lb_IP_OWNPV', 'p_TRACK_VeloCHI2NDOF', 
    'Lb_TAUERR', 'category'
]))

keeps = [j for j in cols if j not in feats]

data.drop_features(keeps, target='all')
data.set_event_ratio(event_ratio)
csv_path = f'data_files/{version}.csv'
data.to_csv(csv_path)
print(f"Output to CSV: {csv_path}\nGenerating metadata file...")

f = open(f'outputs/csv_metadata/{version}.txt', 'w')
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
today = date.today()
d1 = today.strftime("%d/%m/%Y")
f.write(f"Version: {version}\nCompile Date: {d1}\nCompile Time: {current_time}\nPreselection Applied: {apply_preselection}\n")
f.write(f"Event Ratio: {event_ratio}\nFeatures Included:\n")
for i, p in enumerate(feats):
    string = f"{i}. {p}\n"
    f.write(string)
f.write("Custom Feature Expressions:\n")
for key, value in new_features.items():
    f.write(f"{key} = {value}\n")
f.close()

print(f"INFO: Metadata file generated at outputs/csv_metadata/{version}.txt'")