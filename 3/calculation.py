import sys
sys.path.insert(1, '/home/lcm/Desktop/PH3244/scripts')
import labtools
labtools.import_data()     
labtools.unit_check()      
labtools.unit_converter()  
labtools.make_parsed_data()


ds = ['V_DS = -5.0V',
'V_DS = -2.65 V' ] 

gs = ['V_GS = 1.558 V',
'V_GS = 0V'      ,
'V_GS = 1 V']

for OBS_NAME in ds:    
    x, y = "VGS", "ID"
    plot_path = labtools.plot(OBS_NAME, x_col=x, y_col=y,        title=r"$V_{GS}$ vs $I_D$ | " +f"{OBS_NAME.split(',')[0]}"
    )
    print(f"\nSaved figure: {plot_path}")

for OBS_NAME in gs:    
    x, y = "VDS", "IGS"
    plot_path = labtools.plot(OBS_NAME, x_col=x, y_col=y,        title=r"$V_{DS}$ vs $I_{GS}$ |" + f" {OBS_NAME.split(',')[0]}"
    )
    print(f"\nSaved figure: {plot_path}")

