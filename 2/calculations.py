import numpy as np
import matplotlib.pyplot as plt


input_char_1_Vbe = np.array([3, 198, 358, 583, 634, 636, 639, 644, 649, 656, 667]) #V # for Vce = 1.00 V
input_char_2_Vbe = np.array([3, 280, 441, 632, 663, 672, 675, 677, 679, 683, 689]) #V # for Vce = 4.00 V
input_char_3_Vbe = np.array([3, 335, 553, 636, 661, 672, 696, 701, 704, 706, 710]) #V # for Vce = 10.00 V
input_char_common_Ib = [0, 0, 0, 5, 10, 15, 30, 50, 80, 120, 200] #micro amps

###
a= np.array([0.004, 0.057, 0.106, 0.177, 0.327, 0.529, 0.823, 1.191, 1.958, 3.205, 5.66, 9.68, 16.76, 22.65])
b= np.array([0.005, 0.058, 0.134, 0.208, 0.405, 0.606, 0.844, 1.134, 2.231, 3.262, 4.86, 7.32])
c= np.array([0.004, 0.012, 0.067, 0.118, 0.243, 0.338, 0.540, 0.798, 1.214, 1.922, 2.829, 4.91, 9.11, 32.53])

a1= np.array([0.00, 2.05, 7.13, 13.84, 17.01, 18.73, 20.32, 20.95, 21.80, 23.09, 25.96, 29.76, 34.46, 38.19])
b1= np.array([0.00, 1.24, 7.19, 10.25, 11.04, 12.29, 12.38, 12.44, 12.67, 12.94, 13.54, 14.05])
c1= np.array([0.00, 0.02, 0.46, 1.64, 2.88, 2.97, 2.97, 2.97, 2.98, 2.99, 3.01, 3.07, 3.16, 4.59])


#### input characteri
# standard dev = sigma /sqrt 12
input_char_Vbe_err = np.full(shape=11, fill_value=1)/np.sqrt(12)
input_char_common_Ib_err = np.full(shape=11, fill_value=5)/np.sqrt(12)

plt.plot(input_char_1_Vbe, input_char_common_Ib, marker='o', color='red', label=r'$V_{\text{CE}}$ = 1.00 V', ls=':')
plt.plot(input_char_2_Vbe, input_char_common_Ib, marker='o', color='blue', label=r'$V_{\text{CE}}$ = 4.00 V', ls='-')
plt.plot(input_char_3_Vbe, input_char_common_Ib, marker='o', color='green',label=r'$V_{\text{CE}}$ = 10.00 V', ls='--')

plt.xlabel(r'$V_{\text{BE}}$ (mV)')
plt.ylabel(r'$I_{\text{B}}$ ($\mu$A)')
plt.legend()

plt.savefig('inputchar.png', dpi=1200 ) 
plt.show()
plt.close() 

#OUTPU3T CHARACTERISCS


plt.plot(a, a1, marker='o', color='red', ls=':', label=r'$I_\text{B}$ = 80 $\mu$A')
plt.plot(b, b1, marker='o', color='blue', ls='-', label=r'$I_\text{B}$ = 50 $\mu$A')
plt.plot(c, c1, marker='o', color='green', ls='--', label=r'$I_\text{B}$ = 20 $\mu$A')

plt.xlim(-0.5, 10)
plt.ylim(-0.5, 30)
plt.xlabel(r'$V_{\text{CE}}$ (V)')
plt.ylabel(r'$I_{\text{C}}$ (mA)')
plt.legend()

plt.savefig('outputchar.png', dpi=1200)  
plt.show()
plt.close()
