import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit

plt.rc('font', size=14, weight='normal')
#mpl.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['legend.fontsize'] = 14
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 



#Formel für die Standardabweichung des Mittelwerts
def stanni(Zahlen, Mittelwert):
    i=0
    s=0
    n=len(Zahlen)
    while i<len(Zahlen):
        s = s + (Zahlen[i] - Mittelwert)**2
        i = i + 1
    return np.sqrt(s/(n*(n-1)))

def linear(x, m, b):
    return m*x+b

def hochdrei(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def mean(x):
    return np.sum(x)/(len(x))

def gauß(x, sigma, bg, x0, A):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))/(np.sqrt(2*np.pi) * sigma)



####### Lambda in nm und Intensität in arb. units 
 

# Anregung mit verschiedenen Wellenlängen 

lam_1_448, I_1_448, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_1_448nm.csv", unpack = "True") #Integrationszeit 500ms
lam_1_518, I_1_518, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_1_518nm.csv", unpack = "True") #Integrationszeit 6000ms
lam_1_636, I_1_636, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_1_636nm.csv", unpack = "True") #Integrationszeit 3000ms
P_1_448 = 3.00e-3
P_1_518 = 5.65e-3
P_1_636 = 6.63e-3

lam_2_448, I_2_448, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_2_448nm.csv", unpack = "True") #Integrationszeit 400ms
lam_2_518, I_2_518, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_2_518nm.csv", unpack = "True") #Integrationszeit 2000ms
lam_2_636, I_2_636, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_2_636nm.csv", unpack = "True") #Integrationszeit 1000ms
P_2_448 = 3.00e-3
P_2_518 = 5.65e-3
P_2_636 = 6.63e-3

lam_3_448, I_3_448, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_3_448nm.csv", unpack = "True") #Integrationszeit 200ms
lam_3_518, I_3_518, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_3_518nm.csv", unpack = "True") #Integrationszeit 1000ms
lam_3_636, I_3_636, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_3_636nm.csv", unpack = "True") #Integrationszeit 750ms
P_3_448 = 3.00e-3
P_3_518 = 5.65e-3
P_3_636 = 6.63e-3

lam_4_448, I_4_448, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_4_448nm.csv", unpack = "True") #Integrationszeit 500ms
lam_4_518, I_4_518, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_4_518nm.csv", unpack = "True") #Integrationszeit 2000ms
lam_4_636, I_4_636, B = np.genfromtxt("Spektrum(Lambda_Laser)/PL_4_636nm.csv", unpack = "True") #Integrationszeit 1000ms
P_4_448 = 3.00e-3
P_4_518 = 5.65e-3
P_4_636 = 6.63e-3



#Anregung mit 405nm
lam_1_P1, I_1_P1, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P1.csv", unpack = "True") #Integrationszeit 750ms
lam_1_P2, I_1_P2, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P2.csv", unpack = "True") #Integrationszeit 400ms
lam_1_P3, I_1_P3, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P3.csv", unpack = "True") #Integrationszeit 300ms
lam_1_P4, I_1_P4, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P4.csv", unpack = "True") #Integrationszeit 200ms
lam_1_P5, I_1_P5, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P5.csv", unpack = "True") #Integrationszeit 150ms
lam_1_P6, I_1_P6, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P6.csv", unpack = "True") #Integrationszeit 100ms
lam_1_P7, I_1_P7, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P7.csv", unpack = "True") #Integrationszeit 150ms
lam_1_P8, I_1_P8, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P8.csv", unpack = "True") #Integrationszeit 150ms
lam_1_P9, I_1_P9, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P9.csv", unpack = "True") #Integrationszeit 100ms
lam_1_P10, I_1_P10, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P10.csv", unpack = "True") #Integrationszeit 100ms
lam_1_P11, I_1_P11, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P11.csv", unpack = "True") #Integrationszeit 100ms
lam_1_P12, I_1_P12, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P12.csv", unpack = "True") #Integrationszeit 100ms
lam_1_P13, I_1_P13, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P13.csv", unpack = "True") #Integrationszeit 100ms
lam_1_P14, I_1_P14, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P14.csv", unpack = "True") #Integrationszeit 75ms
lam_1_P15, I_1_P15, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P15.csv", unpack = "True") #Integrationszeit 75ms
lam_1_P16, I_1_P16, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P16.csv", unpack = "True") #Integrationszeit 75ms
lam_1_P17, I_1_P17, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P17.csv", unpack = "True") #Integrationszeit 75ms
lam_1_P18, I_1_P18, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P18.csv", unpack = "True") #Integrationszeit 75ms
lam_1_P19, I_1_P19, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P19.csv", unpack = "True") #Integrationszeit 50ms
lam_1_P20, I_1_P20, B = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P20.csv", unpack = "True") #Integrationszeit 50ms

#Verschiedene Anregungsleistungen
P1 = 1.003e-3
P2 = 2.000e-3
P3 = 3.00e-3
P4 = 4.00e-3
P5 = 5.00e-3
P6 = 6.01e-3
P7 = 7.00e-3
P8 = 8.00e-3
P9 = 9.02e-3
P10 = 10.00e-3
P11 = 10.99e-3
P12 = 12.00e-3
P13 = 13.21e-3
P14 = 13.97e-3
P15 = 15.25e-3
P16 = 16.13e-3
P17 = 17.03e-3
P18 = 18.05e-3
P19 = 19.01e-3
P20 = 20.15e-3


#Anregung mit 405nm und P=1mW, um aus den Emissionsspektren die Kristallgröße zu bestimmen
lam_1_pol0, I_1_pol0, B = np.genfromtxt("Spektrum(Pol, 1mW)/PL_1_405nm_pol0.csv", unpack = "True") #Integrationszeit 750ms
lam_1_pol90, I_1_pol90, B = np.genfromtxt("Spektrum(Pol, 1mW)/PL_1_405nm_pol90.csv", unpack = "True") #Integrationszeit 750ms
P_1_pol = 1.04e-3

lam_2_pol0, I_2_pol0, B = np.genfromtxt("Spektrum(Pol, 1mW)/PL_2_405nm_pol0.csv", unpack = "True") #Integrationszeit 500ms
lam_2_pol90, I_2_pol90, B = np.genfromtxt("Spektrum(Pol, 1mW)/PL_2_405nm_pol90.csv", unpack = "True") #Integrationszeit 500ms
P_1_pol = 1.06e-3

lam_3_pol0, I_3_pol0, B = np.genfromtxt("Spektrum(Pol, 1mW)/PL_3_405nm_pol0.csv", unpack = "True") #Integrationszeit 300ms
lam_3_pol90, I_3_pol90, B = np.genfromtxt("Spektrum(Pol, 1mW)/PL_3_405nm_pol90.csv", unpack = "True") #Integrationszeit 300ms
P_1_pol = 1.05e-3

lam_4_pol0, I_4_pol0, B = np.genfromtxt("Spektrum(Pol, 1mW)/PL_4_405nm_pol0.csv", unpack = "True") #Integrationszeit 1000ms
lam_4_pol90, I_4_pol90, B = np.genfromtxt("Spektrum(Pol, 1mW)/PL_4_405nm_pol90.csv", unpack = "True") #Integrationszeit 1000ms
P_1_pol = 1.01e-3




# Emissionswellenlängen und Intensitäten aus Gauß-Anpassungen ermitteln
# Dabei normieren der Intensität auf die Integrationszeit

params_gauß_1_pol0, cov_gauß_1_pol0 = curve_fit(gauß, lam_1_pol0, I_1_pol0/(0.75))
params_gauß_1_pol90, cov_gauß_1_pol90 = curve_fit(gauß, lam_1_pol90, I_1_pol90/(0.75))

params_gauß_2_pol0, cov_gauß_2_pol0 = curve_fit(gauß, lam_2_pol0, I_2_pol0/(0.5))
params_gauß_2_pol90, cov_gauß_2_pol90 = curve_fit(gauß, lam_2_pol90, I_2_pol90/(0.5))

params_gauß_3_pol0, cov_gauß_3_pol0 = curve_fit(gauß, lam_3_pol0, I_3_pol0/(0.3))
params_gauß_3_pol90, cov_gauß_3_pol90 = curve_fit(gauß, lam_3_pol90, I_3_pol90/(0.3))

params_gauß_4_pol0, cov_gauß_4_pol0 = curve_fit(gauß, lam_4_pol0, I_4_pol0/(1))
params_gauß_4_pol90, cov_gauß_4_pol90 = curve_fit(gauß, lam_4_pol90, I_4_pol90/(1))




params_gauß_1_P1, cov_gauß_1_P1 = curve_fit(gauß, lam_1_P1, I_1_P1/(0.75))
params_gauß_1_P2, cov_gauß_1_P2 = curve_fit(gauß, lam_1_P2, I_1_P2/(0.4))
params_gauß_1_P3, cov_gauß_1_P3 = curve_fit(gauß, lam_1_P3, I_1_P3/(0.4))
params_gauß_1_P4, cov_gauß_1_P4 = curve_fit(gauß, lam_1_P4, I_1_P4/(0.2))
params_gauß_1_P5, cov_gauß_1_P5 = curve_fit(gauß, lam_1_P5, I_1_P5/(0.15))
params_gauß_1_P6, cov_gauß_1_P6 = curve_fit(gauß, lam_1_P6, I_1_P6/(0.1))
params_gauß_1_P7, cov_gauß_1_P7 = curve_fit(gauß, lam_1_P7, I_1_P7/(0.15))
params_gauß_1_P8, cov_gauß_1_P8 = curve_fit(gauß, lam_1_P8, I_1_P8/(0.15))
params_gauß_1_P9, cov_gauß_1_P9 = curve_fit(gauß, lam_1_P9, I_1_P9/(0.1))
params_gauß_1_P10, cov_gauß_1_P10 = curve_fit(gauß, lam_1_P10, I_1_P10/(0.1))
params_gauß_1_P11, cov_gauß_1_P11 = curve_fit(gauß, lam_1_P11, I_1_P11/(0.1))
params_gauß_1_P12, cov_gauß_1_P12 = curve_fit(gauß, lam_1_P12, I_1_P12/(0.1))
params_gauß_1_P13, cov_gauß_1_P13 = curve_fit(gauß, lam_1_P13, I_1_P13/(0.1))
params_gauß_1_P14, cov_gauß_1_P14 = curve_fit(gauß, lam_1_P14, I_1_P14/(0.075))
params_gauß_1_P15, cov_gauß_1_P15 = curve_fit(gauß, lam_1_P15, I_1_P15/(0.075))
params_gauß_1_P16, cov_gauß_1_P16 = curve_fit(gauß, lam_1_P16, I_1_P16/(0.075))
params_gauß_1_P17, cov_gauß_1_P17 = curve_fit(gauß, lam_1_P17, I_1_P17/(0.075))
params_gauß_1_P18, cov_gauß_1_P18 = curve_fit(gauß, lam_1_P18, I_1_P18/(0.075))
params_gauß_1_P19, cov_gauß_1_P19 = curve_fit(gauß, lam_1_P19, I_1_P19/(0.05))
params_gauß_1_P20, cov_gauß_1_P20 = curve_fit(gauß, lam_1_P20, I_1_P20/(0.05))




params_gauß_1_448, cov_gauß_1_448 = curve_fit(gauß, lam_1_448, I_1_448/(0.5))
params_gauß_1_518, cov_gauß_1_518 = curve_fit(gauß, lam_1_518, I_1_518/(6))
params_gauß_1_636, cov_gauß_1_636 = curve_fit(gauß, lam_1_636, I_1_636/(3))

params_gauß_2_448, cov_gauß_2_448 = curve_fit(gauß, lam_2_448, I_2_448/(0.4))
params_gauß_2_518, cov_gauß_2_518 = curve_fit(gauß, lam_2_518, I_2_518/(2))
params_gauß_2_636, cov_gauß_2_636 = curve_fit(gauß, lam_2_636, I_2_636/(1))

params_gauß_3_448, cov_gauß_3_448 = curve_fit(gauß, lam_3_448, I_3_448/(0.2))
params_gauß_3_518, cov_gauß_3_518 = curve_fit(gauß, lam_3_518, I_3_518/(1))
params_gauß_3_636, cov_gauß_3_636 = curve_fit(gauß, lam_3_636, I_3_636/(0.75))

params_gauß_4_448, cov_gauß_4_448 = curve_fit(gauß, lam_4_448, I_4_448/(0.5))
params_gauß_4_518, cov_gauß_4_518 = curve_fit(gauß, lam_4_518, I_4_518/(2))
params_gauß_4_636, cov_gauß_4_636 = curve_fit(gauß, lam_4_636, I_4_636/(1))
