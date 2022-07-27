import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from uncertainties import ufloat
from scipy.optimize import curve_fit

def gauss(x, a, t):
    return a * np.exp(-4 * np.log(2) * (x/t)**2)
    # return a * np.exp(-4 * np.log(2) * (x/t1)**2) + b * np.exp(-4 * np.log(2) * (x/t2)**2)

#################### Spektrum ####################

l,I = np.genfromtxt("data/Puls_Spektrum.csv", unpack = "True")
plt.plot(l, I, 'r', linewidth=1, label=r'Spektrum')
plt.xlabel(r'$\lambda$ / nm')
plt.ylabel(r'I (arb. unit)')
plt.legend(loc='best')
plt.savefig('pictures/Spektrum_normal.pdf')
plt.show()
plt.clf()

l,I = np.genfromtxt("data/Puls_1550_30_Spektrum.csv", unpack = "True")
plt.plot(l, I, 'r', linewidth=1, label=r'Spektrum (30 nm-Filter)')
plt.xlabel(r'$\lambda$ / nm')
plt.ylabel(r'I (arb. unit)')
plt.legend(loc='best')
plt.savefig('pictures/Spektrum_30.pdf')
plt.show()
plt.clf()

l,I = np.genfromtxt("data/Puls_1550_12_Spektrum.csv", unpack = "True")
plt.plot(l, I, 'r', linewidth=1, label=r'Spektrum (12 nm-Filter)')
plt.xlabel(r'$\lambda$ / nm')
plt.ylabel(r'I (arb. unit)')
plt.legend(loc='best')
plt.savefig('pictures/Spektrum_12.pdf')
plt.show()
plt.clf()

#################### Normal ####################

x1,x2,x3,x4,x5,x6,x7,x8 = np.genfromtxt("data/Puls_Autokorrelation.csv", unpack = "True")
x2 = x2 - x2[162] #in mm
x2 = x2 / 1000 #in m
x2 = 2 * x2 / 299792458 #in s
x2 = x2 * 10**12 #in ps
x3 = x3 / x3[162]
plt.plot(x2[:153], x3[:153], '--r', linewidth=1)
plt.plot(x2[152:173], x3[152:173], 'r', linewidth=2.5, label=r'Autokorrelation')
plt.plot(x2[172:], x3[172:], '--r', linewidth=1)

x_plot = np.linspace(-0.75, 0.75, 1000)
params, covariance_matrix = curve_fit(gauss, x2[152:173], x3[152:173])
# params, covariance_matrix = curve_fit(gauss, x2, x3)
uncertainties = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, gauss(x_plot, *params), "b", linewidth=1, label=r'Gauss Fit')
# print('Parameter')
# print(params)
# print('Fehler')
# print(uncertainties)
tau_ac = ufloat(params[1], uncertainties[1])
# print('tau_AC')
# print(tau_ac)
tau_puls = tau_ac / np.sqrt(2) * 1000 #in fs
print('tau_Puls')
print(tau_puls)
print(tau_puls.n)
print(tau_puls.s)

plt.xlabel(r'$\tau$ / ps')
plt.ylabel(r'I (arb. unit)')
plt.xlim(-0.75,0.75)
# plt.ylim(0,0.6)
plt.legend(loc='best')
plt.savefig('pictures/Puls_normal.pdf')
# plt.show()
plt.clf()

#################### 30 Filter ####################

x1,x2,x3,x4,x5,x6,x7,x8 = np.genfromtxt("data/Puls_1550_30_Autokorrelation.csv", unpack = "True")
x2 = x2 - x2[172] #in mm
x2 = x2 / 1000 #in m
x2 = 2 * x2 / 299792458 #in s
x2 = x2 * 10**12 #in ps
x3 = x3 / x3[172]
plt.plot(x2[:150], x3[:150], '--r', linewidth=1)
plt.plot(x2[149:197], x3[149:197], 'r', linewidth=2.5, label=r'Autokorrelation (30 nm-Filter)')
plt.plot(x2[196:], x3[196:], '--r', linewidth=1)

x_plot = np.linspace(-1, 1, 1000)
params, covariance_matrix = curve_fit(gauss, x2[149:197], x3[149:197])
# params, covariance_matrix = curve_fit(gauss, x2, x3)
uncertainties = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, gauss(x_plot, *params), "b", linewidth=1, label=r'Gauss Fit')
# print('Parameter')
# print(params)
# print('Fehler')
# print(uncertainties)
tau_ac = ufloat(params[1], uncertainties[1])
# print('tau_AC')
# print(tau_ac)
tau_puls = tau_ac / np.sqrt(2) * 1000 #in fs
print('tau_Puls')
print(tau_puls)
print(tau_puls.n)
print(tau_puls.s)

plt.xlabel(r'$\tau$ / ps')
plt.ylabel(r'I (arb. unit)')
plt.xlim(-1,1)
# plt.ylim(0,0.6)
plt.legend(loc='best')
plt.savefig('pictures/Puls_30.pdf')
plt.show()
plt.clf()

#################### 12 Filter ####################

x1,x2,x3,x4,x5,x6,x7,x8 = np.genfromtxt("data/Puls_1550_12_Autokorrelation.csv", unpack = "True")
x2 = x2 - x2[166] #in mm
x2 = x2 / 1000 #in m
x2 = 2 * x2 / 299792458 #in s
x2 = x2 * 10**12 #in ps
x3 = x3 / x3[166]
plt.plot(x2[:142], x3[:142], '--r', linewidth=1)
plt.plot(x2[141:192], x3[141:192], 'r', linewidth=2.5, label=r'Autokorrelation (12 nm-Filter)')
plt.plot(x2[191:], x3[191:], '--r', linewidth=1)

x_plot = np.linspace(-1, 1, 1000)
params, covariance_matrix = curve_fit(gauss, x2[141:192], x3[141:192])
# params, covariance_matrix = curve_fit(gauss, x2, x3)
uncertainties = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, gauss(x_plot, *params), "b", linewidth=1, label=r'Gauss Fit')
# print('Parameter')
# print(params)
# print('Fehler')
# print(uncertainties)
tau_ac = ufloat(params[1], uncertainties[1])
# print('tau_AC')
# print(tau_ac)
tau_puls = tau_ac / np.sqrt(2) * 1000 #in fs
print('tau_Puls')
print(tau_puls)
print(tau_puls.n)
print(tau_puls.s)

plt.xlabel(r'$\tau$ / ps')
plt.ylabel(r'I (arb. unit)')
plt.xlim(-1,1)
# plt.ylim(0,0.6)
plt.legend(loc='best')
plt.savefig('pictures/Puls_12.pdf')
plt.show()
plt.clf()

#################### Silizium ####################

x1,x2,x3,x4,x5,x6,x7,x8 = np.genfromtxt("data/Puls_Si_Autokorrelation.csv", unpack = "True")
x2 = x2 - x2[349] #in mm
x2 = x2 / 1000 #in m
x2 = 2 * x2 / 299792458 #in s
x2 = x2 * 10**12 #in ps
x3 = x3 / x3[349]
plt.plot(x2[:283], x3[:283], '--r', linewidth=1)
plt.plot(x2[282:406], x3[282:406], 'r', linewidth=2.5, label=r'Autokorrelation (Si 12 mm)')
plt.plot(x2[405:], x3[405:], '--r', linewidth=1)

x_plot = np.linspace(-1, 1, 1000)
params, covariance_matrix = curve_fit(gauss, x2[282:406], x3[282:406])
# params, covariance_matrix = curve_fit(gauss, x2, x3)
uncertainties = np.sqrt(np.diag(covariance_matrix))
plt.plot(x_plot, gauss(x_plot, *params), "b", linewidth=1, label=r'Gauss Fit')
# print('Parameter')
# print(params)
# print('Fehler')
# print(uncertainties)
tau_ac = ufloat(params[1], uncertainties[1])
# print('tau_AC')
# print(tau_ac)
tau_puls = tau_ac / np.sqrt(2) * 1000 #in fs
print('tau_Puls')
print(tau_puls)
print(tau_puls.n)
print(tau_puls.s)

plt.xlabel(r'$\tau$ / ps')
plt.ylabel(r'I (arb. unit)')
# plt.xlim(-1,1)
# plt.ylim(0,0.6)
plt.legend(loc='best')
plt.savefig('pictures/Puls_Si.pdf')
plt.show()
plt.clf()

#################### Glass 13.51 ####################

x1,x2,x3,x4,x5,x6,x7,x8 = np.genfromtxt("data/Puls_SF11_Autokorrelation_1.csv", unpack = "True")
x2 = x2 - x2[261] #in mm
x2 = x2 / 1000 #in m
x2 = 2 * x2 / 299792458 #in s
x2 = x2 * 10**12 #in ps
x3 = x3 / x3[261]
plt.plot(x2[:251], x3[:251], '--r', linewidth=1)
plt.plot(x2[250:272], x3[250:272], 'r', linewidth=2.5, label=r'Autokorrelation (SF11 13,51 mm)')
plt.plot(x2[271:], x3[271:], '--r', linewidth=1)

x_plot = np.linspace(-1, 1, 1000)
params1, covariance_matrix1 = curve_fit(gauss, x2[250:272], x3[250:272])
# params, covariance_matrix = curve_fit(gauss, x2, x3)
uncertainties1 = np.sqrt(np.diag(covariance_matrix1))
plt.plot(x_plot, gauss(x_plot, *params1), "b", linewidth=1, label=r'Gauss Fit')
# print('Parameter')
# print(params)
# print('Fehler')
# print(uncertainties)
tau_ac = ufloat(params1[1], uncertainties1[1])
# print('tau_AC')
# print(tau_ac)
tau_puls1 = tau_ac / np.sqrt(2) * 1000 #in fs
print('tau_Puls1')
print(tau_puls1)
print(tau_puls1.n)
print(tau_puls1.s)

plt.xlabel(r'$\tau$ / ps')
plt.ylabel(r'I (arb. unit)')
plt.xlim(-1,1)
# plt.ylim(0,0.6)
plt.legend(loc='best')
plt.savefig('pictures/Puls_G1.pdf')
plt.show()
plt.clf()

#################### Glass 17.64 ####################

x1,x2,x3,x4,x5,x6,x7,x8 = np.genfromtxt("data/Puls_SF11_Autokorrelation_2.csv", unpack = "True")
x2 = x2 - x2[262] #in mm
x2 = x2 / 1000 #in m
x2 = 2 * x2 / 299792458 #in s
x2 = x2 * 10**12 #in ps
x3 = x3 / x3[262]
plt.plot(x2[:252], x3[:252], '--r', linewidth=1)
plt.plot(x2[251:273], x3[251:273], 'r', linewidth=2.5, label=r'Autokorrelation (SF11 17,64 mm)')
plt.plot(x2[272:], x3[272:], '--r', linewidth=1)

x_plot = np.linspace(-1, 1, 1000)
params2, covariance_matrix2 = curve_fit(gauss, x2[251:273], x3[251:273])
# params, covariance_matrix = curve_fit(gauss, x2, x3)
uncertainties2 = np.sqrt(np.diag(covariance_matrix2))
plt.plot(x_plot, gauss(x_plot, *params2), "b", linewidth=1, label=r'Gauss Fit')
# print('Parameter')
# print(params)
# print('Fehler')
# print(uncertainties)
tau_ac = ufloat(params2[1], uncertainties2[1])
# print('tau_AC')
# print(tau_ac)
tau_puls2 = tau_ac / np.sqrt(2) * 1000 #in fs
print('tau_Puls2')
print(tau_puls2)
print(tau_puls2.n)
print(tau_puls2.s)

plt.xlabel(r'$\tau$ / ps')
plt.ylabel(r'I (arb. unit)')
plt.xlim(-1,1)
# plt.ylim(0,0.6)
plt.legend(loc='best')
plt.savefig('pictures/Puls_G2.pdf')
plt.show()
plt.clf()

#################### Glass 23.85 ####################

x1,x2,x3,x4,x5,x6,x7,x8 = np.genfromtxt("data/Puls_SF11_Autokorrelation_3.csv", unpack = "True")
x2 = x2 - x2[259] #in mm
x2 = x2 / 1000 #in m
x2 = 2 * x2 / 299792458 #in s
x2 = x2 * 10**12 #in ps
x3 = x3 / x3[259]
plt.plot(x2[:249], x3[:249], '--r', linewidth=1)
plt.plot(x2[248:270], x3[248:270], 'r', linewidth=2.5, label=r'Autokorrelation (SF11 23,85 mm)')
plt.plot(x2[269:], x3[269:], '--r', linewidth=1)

x_plot = np.linspace(-1, 1, 1000)
params3, covariance_matrix3 = curve_fit(gauss, x2[248:270], x3[248:270])
# params, covariance_matrix = curve_fit(gauss, x2, x3)
uncertainties3 = np.sqrt(np.diag(covariance_matrix3))
plt.plot(x_plot, gauss(x_plot, *params3), "b", linewidth=1, label=r'Gauss Fit')
# print('Parameter')
# print(params)
# print('Fehler')
# print(uncertainties)
tau_ac = ufloat(params3[1], uncertainties3[1])
# print('tau_AC')
# print(tau_ac)
tau_puls3 = tau_ac / np.sqrt(2) * 1000 #in fs
print('tau_Puls3')
print(tau_puls3)
print(tau_puls3.n)
print(tau_puls3.s)

plt.xlabel(r'$\tau$ / ps')
plt.ylabel(r'I (arb. unit)')
plt.xlim(-1,1)
# plt.ylim(0,0.6)
plt.legend(loc='best')
plt.savefig('pictures/Puls_G3.pdf')
plt.show()
plt.clf()

#################### Glass 29.83 ####################

x1,x2,x3,x4,x5,x6,x7,x8 = np.genfromtxt("data/Puls_SF11_Autokorrelation_4.csv", unpack = "True")
x2 = x2 - x2[262] #in mm
x2 = x2 / 1000 #in m
x2 = 2 * x2 / 299792458 #in s
x2 = x2 * 10**12 #in ps
x3 = x3 / x3[262]
plt.plot(x2[:252], x3[:252], '--r', linewidth=1)
plt.plot(x2[251:274], x3[251:274], 'r', linewidth=2.5, label=r'Autokorrelation (SF11 29,83 mm)')
plt.plot(x2[273:], x3[273:], '--r', linewidth=1)

x_plot = np.linspace(-1, 1, 1000)
params4, covariance_matrix4 = curve_fit(gauss, x2[251:274], x3[251:274])
# params, covariance_matrix = curve_fit(gauss, x2, x3)
uncertainties4 = np.sqrt(np.diag(covariance_matrix4))
plt.plot(x_plot, gauss(x_plot, *params4), "b", linewidth=1, label=r'Gauss Fit')
# print('Parameter')
# print(params)
# print('Fehler')
# print(uncertainties)
tau_ac = ufloat(params4[1], uncertainties4[1])
# print('tau_AC')
# print(tau_ac)
tau_puls4 = tau_ac / np.sqrt(2) * 1000 #in fs
print('tau_Puls4')
print(tau_puls4)
print(tau_puls4.n)
print(tau_puls4.s)

plt.xlabel(r'$\tau$ / ps')
plt.ylabel(r'I (arb. unit)')
plt.xlim(-1,1)
# plt.ylim(0,0.6)
plt.legend(loc='best')
plt.savefig('pictures/Puls_G4.pdf')
plt.show()
plt.clf()

#################### Glass 53.68 ####################

x1,x2,x3,x4,x5,x6,x7,x8 = np.genfromtxt("data/Puls_SF11_Autokorrelation_5.csv", unpack = "True")
x2 = x2 - ((x2[264]+x2[265])/2) #in mm
x2 = x2 / 1000 #in m
x2 = 2 * x2 / 299792458 #in s
x2 = x2 * 10**12 #in ps
x3 = x3 / ((x3[264]+x3[265])/2)
plt.plot(x2[:254], x3[:254], '--r', linewidth=1)
plt.plot(x2[253:277], x3[253:277], 'r', linewidth=2.5, label=r'Autokorrelation (SF11 53,68 mm)')
plt.plot(x2[276:], x3[276:], '--r', linewidth=1)

x_plot = np.linspace(-1, 1, 1000)
params5, covariance_matrix5 = curve_fit(gauss, x2[251:274], x3[251:274])
# params, covariance_matrix = curve_fit(gauss, x2, x3)
uncertainties5 = np.sqrt(np.diag(covariance_matrix5))
plt.plot(x_plot, gauss(x_plot, *params5), "b", linewidth=1, label=r'Gauss Fit')
# print('Parameter')
# print(params)
# print('Fehler')
# print(uncertainties)
tau_ac = ufloat(params5[1], uncertainties5[1])
# print('tau_AC')
# print(tau_ac)
tau_puls5 = tau_ac / np.sqrt(2) * 1000 #in fs
print('tau_Puls5')
print(tau_puls5)
print(tau_puls5.n)
print(tau_puls5.s)

plt.xlabel(r'$\tau$ / ps')
plt.ylabel(r'I (arb. unit)')
plt.xlim(-1,1)
# plt.ylim(0,0.6)
plt.legend(loc='best')
plt.savefig('pictures/Puls_G5.pdf')
plt.show()
plt.clf()



plt.plot(x_plot, gauss(x_plot, *params1), linewidth=0.5, label=r'SF11 13,51 mm')
plt.plot(x_plot, gauss(x_plot, *params2), linewidth=0.5, label=r'SF11 17,64 mm')
plt.plot(x_plot, gauss(x_plot, *params3), linewidth=0.5, label=r'SF11 23,85 mm')
plt.plot(x_plot, gauss(x_plot, *params4), linewidth=0.5, label=r'SF11 29,83 mm')
plt.plot(x_plot, gauss(x_plot, *params5), linewidth=0.5, label=r'SF11 53,68 mm')
plt.xlabel(r'$\tau$ / ps')
plt.ylabel(r'I (arb. unit)')
plt.xlim(-0.15,0.15)
plt.legend(loc='best')
plt.savefig('pictures/Puls_Glass.pdf')
plt.show()
plt.clf()
