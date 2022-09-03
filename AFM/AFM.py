import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from uncertainties import ufloat
from scipy.optimize import curve_fit

# plt.rc('font', size=14, weight='normal')
# #mpl.rcParams['font.sans-serif'] = 'Arial'
# plt.rcParams['legend.fontsize'] = 14
# plt.rc('xtick', labelsize=14) 
# plt.rc('ytick', labelsize=14) 


#Formel für die Standardabweichung des Mittelwerts
def stanni(Zahlen, Mittelwert):
    i=0
    s=0
    n=len(Zahlen)
    while i<len(Zahlen):
        s = s + (Zahlen[i] - Mittelwert)**2
        i = i + 1
    return np.sqrt(s/(n*(n-1)))

def mean(x):
    return np.sum(x)/(len(x))


# x = [1.12, 1.07, 1.18]
# print('Mittelwert')
# print(mean(x))
# print('Fehler')
# print(stanni(x, mean(x)))

# i = 0
# sum = 0
# while i < 22951:
#     sum += 2*np.pi*(23+ufloat(0.001547,0.000056)*i)
#     i  += 1
# print('Spur in mm')
# print(sum.n)
# print(sum.s)

# Speicher=ufloat(5876630000,710000)/ufloat(0.684,0.013)*4
# print('Speicher')
# print('Bit')
# print(Speicher.n)
# print(Speicher.s)
# print('Byte')
# print((Speicher/8).n)
# print((Speicher/8).s)

x,y = np.genfromtxt("data/Edelstahl.csv", unpack = "True")
x_neu = x*20/75 #in um
d = x_neu[632]-x_neu[1082] #in um
dm = d/10**6 #in m
print(d)
k = 0.2 #in N/m
F = dm*k*10**9 #in nN
print('Adhäsionskraft (Edelstahl)')
print(F)
plt.plot(x_neu[:803], y[:803],'b.', markersize=1, label=r'Annäherungskurve (Edelstahl)')
plt.plot(x_neu[803:], y[803:],'r.', markersize=1, label=r'Rückzugskurve (Edelstahl)')
plt.xlabel(r'z-Auslenkung (Ursprung beliebig) / $\mu$m')
plt.ylabel(r'x-Ablenkung / V')
# # plt.ylim(0,80)
plt.legend(loc='best')
plt.savefig('pictures/Edelstahl.pdf')
plt.show()
plt.clf()

x,y = np.genfromtxt("data/Teflon.csv", unpack = "True")
x_neu = x*20/75 #in um
d = x_neu[438]-x_neu[1251] #in um
dm = d/10**6 #in m
print(d)
k = 0.2 #in N/m
F = dm*k*10**9 #in nN
print('Adhäsionskraft (Teflon)')
print(F)
plt.plot(x_neu[:804], y[:804],'b.' , markersize=1, label=r'Annäherungskurve (Teflon)')
plt.plot(x_neu[804:], y[804:],'r.' , markersize=1, label=r'Rückzugskurve (Teflon)')
plt.xlabel(r'z-Auslenkung (Ursprung beliebig) / $\mu$m')
plt.ylabel(r'x-Ablenkung / V')
# # plt.ylim(0,80)
plt.legend(loc='best')
plt.savefig('pictures/Teflon.pdf')
plt.show()
plt.clf()

x,y = np.genfromtxt("data/DLC.csv", unpack = "True")
x_neu = x*20/75 #in um
d = x_neu[501]-x_neu[1182] #in um
dm = d/10**6 #in m
print(d)
k = 0.2 #in N/m
F = dm*k*10**9 #in nN
print('Adhäsionskraft (DLC)')
print(F)
plt.plot(x_neu[:803], y[:803],'b.' , markersize=1, label=r'Annäherungskurve (DLC)')
plt.plot(x_neu[803:], y[803:],'r.' , markersize=1, label=r'Rückzugskurve (DLC)')
plt.xlabel(r'z-Auslenkung (Ursprung beliebig) / $\mu$m')
plt.ylabel(r'x-Ablenkung / V')
# # plt.ylim(0,80)
plt.legend(loc='best')
plt.savefig('pictures/DLC.pdf')
plt.show()
plt.clf()


ex,ey = np.genfromtxt("data/Edelstahl.csv", unpack = "True")
tx,ty = np.genfromtxt("data/Teflon.csv", unpack = "True")
ex_neu = ex*20/75 #in um
tx_neu = tx*20/75 #in um
ex_move = ex_neu - ex_neu[632]
ey_move = ey - ey[632]
tx_move = tx_neu - tx_neu[438]
ty_move = ty - ty[438]
plt.plot(ex_move[632:803], ey_move[632:803],'b.' , markersize=1, label=r'Annäherungskurve (Edelstahl)')
plt.plot(tx_move[438:804], ty_move[438:804],'r.' , markersize=1, label=r'Annäherungskurve (Teflon)')
plt.xlabel(r'z-Auslenkung (Ursprung beliebig) / $\mu$m')
plt.ylabel(r'x-Ablenkung / V')
# # plt.ylim(0,80)
plt.legend(loc='best')
plt.savefig('pictures/Ursprung.pdf')
plt.show()
plt.clf()


i = 0
while ex_move[i]<0.75:
    i  += 1
# print('Test')
# print('Edelstahl')
# print(i)
print(ex_move[i])
# print(ey_move[i])
z_E = ex_move[i] / 10**6 #in m
# print(z_E)

j = 0
while ty_move[j]<ey_move[i]:
    j  += 1
# print('Teflon')
# print(j-1)
print(tx_move[j-1])
# print(ty_move[j-1])
z_T = tx_move[j-1] / 10**6 #in m
# print(z_T)

k = 0.2 #in N/m
F = z_E * k
d = z_T - z_E #in m
v = 0.46
a = 10 * np.pi / 180 #in rad
print('Z Edelstahl')
print(z_E)
print('Z Teflon')
print(z_T)
print('F')
print(F)
print('d')
print(d)
# print('a')
# print(a)
E = F * (1 - v**2) * np.pi / (2 * np.tan(a) * d**2) #in Pa
print(E)
