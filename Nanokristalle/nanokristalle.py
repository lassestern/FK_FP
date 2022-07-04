import numpy as np
import scipy as scipy
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

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

def gauß(x, sigma, x0, A):
    return A*np.exp(-(x-x0)**2/(2*sigma**2))

def pq(p, q):
    return -p/2 + np.sqrt((p/2)**2 - q)




############################################ Grating-Efficiency des Spektrometers ############################################

lam_spek, nicht_wichtig, grat_eff = np.genfromtxt("gratingefficiency.txt", unpack = "True")

#Es wird die über beide Polarisationen gemittelte Effizienz genutzt, da das vom Nanokristall abgestrahlte Licht nicht polarisiert ist 

def pol6(x, g, a, b, c, d, e, f):
    return g*x**6 + a*x**5 + b*x**4 + c*x**3 + d*x**2 + e*x + f

grating_fit, grating_cov = curve_fit(pol6, lam_spek, grat_eff)
grat_fit = np.linspace(246*10**(-9), 1093*10**(-9))

plt.plot(lam_spek, grat_eff, "x", linewidth=0.2, label=r'Messpunkte')
plt.plot(grat_fit*10**9, pol6(grat_fit*10**(9), *grating_fit), linewidth=0.2, label=r'Interpolation')
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'I [will. Einh.]')
plt.ylim(0,80)
plt.legend(loc='best')
plt.savefig('grat_eff.pdf')
plt.clf()



############################################ Zunächst Auslesen aller Daten ############################################
####### Lambda in nm und Intensität in arb. units
 

# Anregung mit verschiedenen Wellenlängen 

lam_1_448, I_1_448 = np.genfromtxt("Spektrum(Lambda_Laser)/PL_1_448nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 500ms
lam_1_518, I_1_518 = np.genfromtxt("Spektrum(Lambda_Laser)/PL_1_518nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 6000ms
lam_1_636, I_1_636 = np.genfromtxt("Spektrum(Lambda_Laser)/PL_1_636nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 3000ms
I_1_448 = I_1_448/(pol6(lam_1_448, *grating_fit) * 500)
I_1_518 = I_1_518/(pol6(lam_1_518, *grating_fit) * 6000)
I_1_636 = I_1_636/(pol6(lam_1_636, *grating_fit) * 3000)
P_1_448 = 3.00e-3
P_1_518 = 5.65e-3
P_1_636 = 6.63e-3

lam_2_448, I_2_448= np.genfromtxt("Spektrum(Lambda_Laser)/PL_2_448nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 400ms
lam_2_518, I_2_518= np.genfromtxt("Spektrum(Lambda_Laser)/PL_2_518nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 2000ms
lam_2_636, I_2_636= np.genfromtxt("Spektrum(Lambda_Laser)/PL_2_636nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 1000ms
I_2_448 = I_2_448/(pol6(lam_2_448, *grating_fit) * 400)
I_2_518 = I_2_518/(pol6(lam_2_518, *grating_fit) * 2000)
I_2_636 = I_2_636/(pol6(lam_2_636, *grating_fit) * 1000)
P_2_448 = 3.00e-3
P_2_518 = 5.65e-3
P_2_636 = 6.63e-3

lam_3_448, I_3_448 = np.genfromtxt("Spektrum(Lambda_Laser)/PL_3_448nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 200ms
lam_3_518, I_3_518 = np.genfromtxt("Spektrum(Lambda_Laser)/PL_3_518nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 1000ms
lam_3_636, I_3_636 = np.genfromtxt("Spektrum(Lambda_Laser)/PL_3_636nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 750ms
I_3_448 = I_3_448/(pol6(lam_3_448, *grating_fit) * 200)
I_3_518 = I_3_518/(pol6(lam_3_518, *grating_fit) * 1000)
I_3_636 = I_3_636/(pol6(lam_3_636, *grating_fit) * 750)
P_3_448 = 3.00e-3
P_3_518 = 5.65e-3
P_3_636 = 6.63e-3

lam_4_448, I_4_448 = np.genfromtxt("Spektrum(Lambda_Laser)/PL_4_448nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 500ms
lam_4_518, I_4_518 = np.genfromtxt("Spektrum(Lambda_Laser)/PL_4_518nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 2000ms
lam_4_636, I_4_636 = np.genfromtxt("Spektrum(Lambda_Laser)/PL_4_636nm.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 1000ms
I_4_448 = I_4_448/(pol6(lam_4_448, *grating_fit) * 500)
I_4_518 = I_4_518/(pol6(lam_4_518, *grating_fit) * 2000)
I_4_636 = I_4_636/(pol6(lam_4_636, *grating_fit) * 1000)
P_4_448 = 3.00e-3
P_4_518 = 5.65e-3
P_4_636 = 6.63e-3



#######Anregung mit 405nm
lam_1_P1, I_1_P1 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P1.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 750ms
lam_1_P2, I_1_P2 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P2.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 400ms
lam_1_P3, I_1_P3 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P3.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 300ms
lam_1_P4, I_1_P4 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P4.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 200ms
lam_1_P5, I_1_P5 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P5.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 150ms
lam_1_P6, I_1_P6 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P6.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 100ms
lam_1_P7, I_1_P7 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P7.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 150ms
lam_1_P8, I_1_P8 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P8.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 150ms
lam_1_P9, I_1_P9 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P9.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 100ms
lam_1_P10, I_1_P10 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P10.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 100ms
lam_1_P11, I_1_P11 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P11.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 100ms
lam_1_P12, I_1_P12 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P12.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 100ms
lam_1_P13, I_1_P13 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P13.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 100ms
lam_1_P14, I_1_P14 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P14.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 75ms
lam_1_P15, I_1_P15 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P15.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 75ms
lam_1_P16, I_1_P16 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P16.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 75ms
lam_1_P17, I_1_P17 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P17.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 75ms
lam_1_P18, I_1_P18 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P18.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 75ms
lam_1_P19, I_1_P19 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P19.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 50ms
lam_1_P20, I_1_P20 = np.genfromtxt("Spektrum(Leistung)/PL_1_405nm_P20.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 50ms

I_1_P1 = I_1_P1/(pol6(lam_1_P1, *grating_fit) * 750)
I_1_P2 = I_1_P2/(pol6(lam_1_P2, *grating_fit) * 400)
I_1_P3 = I_1_P3/(pol6(lam_1_P3, *grating_fit) * 300)
I_1_P4 = I_1_P4/(pol6(lam_1_P4, *grating_fit) * 200)
I_1_P5 = I_1_P5/(pol6(lam_1_P5, *grating_fit) * 150)
I_1_P6 = I_1_P6/(pol6(lam_1_P6, *grating_fit) * 100)
I_1_P7 = I_1_P7/(pol6(lam_1_P7, *grating_fit) * 150)
I_1_P8 = I_1_P8/(pol6(lam_1_P8, *grating_fit) * 150)
I_1_P9 = I_1_P9/(pol6(lam_1_P9, *grating_fit) * 100)
I_1_P10 = I_1_P10/(pol6(lam_1_P10, *grating_fit) * 100)
I_1_P11 = I_1_P11/(pol6(lam_1_P11, *grating_fit) * 100)
I_1_P12 = I_1_P12/(pol6(lam_1_P12, *grating_fit) * 100)
I_1_P13 = I_1_P13/(pol6(lam_1_P13, *grating_fit) * 100)
I_1_P14 = I_1_P14/(pol6(lam_1_P14, *grating_fit) * 75)
I_1_P15 = I_1_P15/(pol6(lam_1_P15, *grating_fit) * 75)
I_1_P16 = I_1_P16/(pol6(lam_1_P16, *grating_fit) * 75)
I_1_P17 = I_1_P17/(pol6(lam_1_P17, *grating_fit) * 75)
I_1_P18 = I_1_P18/(pol6(lam_1_P18, *grating_fit) * 75)
I_1_P19 = I_1_P19/(pol6(lam_1_P19, *grating_fit) * 50)
I_1_P20 = I_1_P20/(pol6(lam_1_P20, *grating_fit) * 50)

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

lam_1_pol0, I_1_pol0 = np.genfromtxt("Spektrum(Pol, 1mW)/PL_1_405nm_pol0.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 750ms
lam_1_pol90, I_1_pol90 = np.genfromtxt("Spektrum(Pol, 1mW)/PL_1_405nm_pol90.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 750ms
P_1_pol = 1.04e-3
I_1_pol0 = I_1_pol0/(pol6(lam_1_pol0, *grating_fit) * 750)
I_1_pol90 = I_1_pol90/(pol6(lam_1_pol90, *grating_fit) *750)

lam_2_pol0, I_2_pol0 = np.genfromtxt("Spektrum(Pol, 1mW)/PL_2_405nm_pol0.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 500ms
lam_2_pol90, I_2_pol90 = np.genfromtxt("Spektrum(Pol, 1mW)/PL_2_405nm_pol90.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 500ms
P_2_pol = 1.06e-3
I_2_pol0 = I_2_pol0/(pol6(lam_2_pol0, *grating_fit) * 500)
I_2_pol90 = I_2_pol90/(pol6(lam_2_pol90, *grating_fit) * 500)

lam_3_pol0, I_3_pol0 = np.genfromtxt("Spektrum(Pol, 1mW)/PL_3_405nm_pol0.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 300ms
lam_3_pol90, I_3_pol90 = np.genfromtxt("Spektrum(Pol, 1mW)/PL_3_405nm_pol90.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 300ms
P_3_pol = 1.05e-3
I_3_pol0 = I_3_pol0/(pol6(lam_3_pol0, *grating_fit) * 300)
I_3_pol90 = I_3_pol90/(pol6(lam_3_pol90, *grating_fit) * 300)

lam_4_pol0, I_4_pol0 = np.genfromtxt("Spektrum(Pol, 1mW)/PL_4_405nm_pol0.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 1000ms
lam_4_pol90, I_4_pol90 = np.genfromtxt("Spektrum(Pol, 1mW)/PL_4_405nm_pol90.csv", comments = "#", skip_header=33, skip_footer=1, unpack = "True") #Integrationszeit 1000ms
P_4_pol = 1.01e-3
I_4_pol0 = I_4_pol0/(pol6(lam_4_pol0, *grating_fit) * 1000)
I_4_pol90 = I_4_pol90/(pol6(lam_4_pol90, *grating_fit) * 1000)



# Emissionswellenlängen und Intensitäten aus Gauß-Anpassungen ermitteln
# Dabei normieren der Intensität auf die Integrationszeit



############################################ Aufgabe (ii) a) & b) --> Bestimmen der Nanopartikeldurchmesser und des Polarisationsgrads der Emission ############################################ 


min = (0, 400, 0)
max = (40, 700, 0.000018)
boundaries = (min, max)

guess = ([12, 650, 0.00001])



params_gauß_1_pol0, cov_gauß_1_pol0 = curve_fit(gauß, lam_1_pol0[2000:], I_1_pol0[2000:]/(0.75), p0=guess, bounds=boundaries)
params_gauß_1_pol90, cov_gauß_1_pol90 = curve_fit(gauß, lam_1_pol90[2050:], I_1_pol90[2050:]/(0.75), p0=guess, bounds=boundaries)
# print(params_gauß_1_pol0)

min = (0, 400, 0)
max = (40, 700, 0.000023)
boundaries = (min, max)
guess = ([12, 570, 0.00002])
params_gauß_2_pol0, cov_gauß_2_pol0 = curve_fit(gauß, lam_2_pol0, I_2_pol0/(0.5), p0=guess, bounds=boundaries)
params_gauß_2_pol90, cov_gauß_2_pol90 = curve_fit(gauß, lam_2_pol90, I_2_pol90/(0.5), p0=guess, bounds=boundaries)
print(params_gauß_2_pol0)


min = (0, 400, 0)
max = (19, 700, 0.000035)
boundaries = (min, max)
guess = ([15, 580, 0.00003])
params_gauß_3_pol0, cov_gauß_3_pol0 = curve_fit(gauß, lam_3_pol0[1200:3000], I_3_pol0[1200:3000]/(0.3), p0=guess, bounds=boundaries)
params_gauß_3_pol90, cov_gauß_3_pol90 = curve_fit(gauß, lam_3_pol90[1200:3000], I_3_pol90[1200:3000]/(0.3), p0=guess, bounds=boundaries)


min = (0, 400, 0)
max = (40, 700, 0.000011)
boundaries = (min, max)
guess = ([30, 540, 0.00001])
params_gauß_4_pol0, cov_gauß_4_pol0 = curve_fit(gauß, lam_4_pol0, I_4_pol0/(1), p0=guess, bounds=boundaries)
params_gauß_4_pol90, cov_gauß_4_pol90 = curve_fit(gauß, lam_4_pol90, I_4_pol90/(1), p0=guess, bounds=boundaries)






lambda_fit = np.linspace(320, 747, 1000)*10**(-9)

#Kristallgröße bestimmen

#Materialeigenschaften des Kerns (CdSe)
E_G = 1.74*const.e
m_e = 0.13*const.m_e
Eg = 1.84*const.e
m_h = -0.45*const.m_e  
mu = m_e*m_h/(m_h+m_e)
epsilon_r = 9.15

E_R_1 = const.h * const.c/(params_gauß_1_pol0[1]*10**(-9))
E_R_2 = const.h * const.c/(params_gauß_2_pol0[1]*10**(-9))
E_R_3 = const.h * const.c/(params_gauß_3_pol0[1]*10**(-9))
E_R_4 = const.h * const.c/(params_gauß_4_pol0[1]*10**(-9))

def q(E_R):
    return -(np.pi*const.hbar)**2/2 *(1/m_e + 1/m_h)/(E_R- E_G + mu * const.e**4/(2*(4*np.pi * const.epsilon_0 * epsilon_r * const.hbar)**2)) 

def p(E_R):
    return (1.786 * const.e**2/(4*np.pi * const.epsilon_0 * epsilon_r))/(E_R- E_G + mu * const.e**4/(2*(4*np.pi * const.epsilon_0 * epsilon_r * const.hbar)**2)) 



print(f"""
Probe 1
Emission bei: {params_gauß_1_pol0[1]} nm
Polarisationsgrad: {((params_gauß_1_pol0[2] - params_gauß_1_pol90[2]))/(params_gauß_1_pol0[2] + params_gauß_1_pol90[2])}
Durchmesser: {pq(p(E_R_1), q(E_R_1))}

Probe 2
Emission bei: {params_gauß_2_pol0[1]} nm
Polarisationsgrad: {((params_gauß_2_pol0[2] - params_gauß_2_pol90[2]))/(params_gauß_2_pol0[2] + params_gauß_2_pol90[2])}
Durchmesser: {pq(p(E_R_2), q(E_R_2))}

Probe 3
Emission bei: {params_gauß_3_pol0[1]} nm
Polarisationsgrad: {((params_gauß_3_pol0[2] - params_gauß_3_pol90[2]))/(params_gauß_3_pol0[2] + params_gauß_3_pol90[2])}
Durchmesser: {pq(p(E_R_4), q(E_R_3))}

Probe 4
Emission bei: {params_gauß_4_pol0[1]} nm
Polarisationsgrad: {((params_gauß_4_pol0[2] - params_gauß_4_pol90[2]))/(params_gauß_4_pol0[2] + params_gauß_4_pol90[2])}
Durchmesser: {pq(p(E_R_4), q(E_R_4))}

""")




plt.plot(lam_1_pol0, I_1_pol0 * 100000, linewidth=0.2, label=r'Messung')
plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_pol0)* 100000, linewidth=0.2, label=r'Gauß Fit')
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'I [will. Einh.]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('Kristall_1.pdf')
plt.clf()


plt.plot(lam_2_pol0, I_2_pol0* 100000, linewidth=0.2, label=r'Messung')
plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_2_pol0)* 100000, linewidth=0.2, label=r'Gauß Fit')
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'I [will. Einh.]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('Kristall_2.pdf')
plt.clf()


plt.plot(lam_3_pol0, I_3_pol0* 100000, linewidth=0.2, label=r'Messung')
plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_3_pol0)* 100000, linewidth=0.2, label=r'Gauß Fit')
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'I [will. Einh.]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('Kristall_3.pdf')
plt.clf()


plt.plot(lam_4_pol0, I_4_pol0* 100000, linewidth=0.2, label=r'Messung')
plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_4_pol0)* 100000, linewidth=0.2, label=r'Gauß Fit')
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'I [will. Einh.]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('Kristall_4.pdf')
plt.clf()






############################################ Aufgabe (ii) c) --> Abhängigkeit der Emissionsenergie und -leistung von der Anregungsleistung ############################################ 

min = (0, 400, 0)
max = (40, 700, 0.00025)
boundaries = (min, max)
guess = ([12, 650, 0.00001])


params_gauß_1_P1, cov_gauß_1_P1 = curve_fit(gauß, lam_1_P1, I_1_P1, p0=guess, bounds=boundaries)
params_gauß_1_P2, cov_gauß_1_P2 = curve_fit(gauß, lam_1_P2, I_1_P2, p0=guess, bounds=boundaries)
params_gauß_1_P3, cov_gauß_1_P3 = curve_fit(gauß, lam_1_P3, I_1_P3, p0=guess, bounds=boundaries)
params_gauß_1_P4, cov_gauß_1_P4 = curve_fit(gauß, lam_1_P4, I_1_P4, p0=guess, bounds=boundaries)
params_gauß_1_P5, cov_gauß_1_P5 = curve_fit(gauß, lam_1_P5, I_1_P5, p0=guess, bounds=boundaries)
params_gauß_1_P6, cov_gauß_1_P6 = curve_fit(gauß, lam_1_P6, I_1_P6, p0=guess, bounds=boundaries)
params_gauß_1_P7, cov_gauß_1_P7 = curve_fit(gauß, lam_1_P7, I_1_P7, p0=guess, bounds=boundaries)
params_gauß_1_P8, cov_gauß_1_P8 = curve_fit(gauß, lam_1_P8, I_1_P8, p0=guess, bounds=boundaries)
params_gauß_1_P9, cov_gauß_1_P9 = curve_fit(gauß, lam_1_P9, I_1_P9, p0=guess, bounds=boundaries)
params_gauß_1_P10, cov_gauß_1_P10 = curve_fit(gauß, lam_1_P10, I_1_P10, p0=guess, bounds=boundaries)
params_gauß_1_P11, cov_gauß_1_P11 = curve_fit(gauß, lam_1_P11, I_1_P11, p0=guess, bounds=boundaries)
params_gauß_1_P12, cov_gauß_1_P12 = curve_fit(gauß, lam_1_P12, I_1_P12, p0=guess, bounds=boundaries)
params_gauß_1_P13, cov_gauß_1_P13 = curve_fit(gauß, lam_1_P13, I_1_P13, p0=guess, bounds=boundaries)
params_gauß_1_P14, cov_gauß_1_P14 = curve_fit(gauß, lam_1_P14, I_1_P14, p0=guess, bounds=boundaries)
params_gauß_1_P15, cov_gauß_1_P15 = curve_fit(gauß, lam_1_P15, I_1_P15, p0=guess, bounds=boundaries)
params_gauß_1_P16, cov_gauß_1_P16 = curve_fit(gauß, lam_1_P16, I_1_P16, p0=guess, bounds=boundaries)
params_gauß_1_P17, cov_gauß_1_P17 = curve_fit(gauß, lam_1_P17, I_1_P17, p0=guess, bounds=boundaries)
params_gauß_1_P18, cov_gauß_1_P18 = curve_fit(gauß, lam_1_P18, I_1_P18, p0=guess, bounds=boundaries)
params_gauß_1_P19, cov_gauß_1_P19 = curve_fit(gauß, lam_1_P19, I_1_P19, p0=guess, bounds=boundaries)
params_gauß_1_P20, cov_gauß_1_P20 = curve_fit(gauß, lam_1_P20, I_1_P20, p0=guess, bounds=boundaries)


I_P = np.array([params_gauß_1_P1[2], params_gauß_1_P2[2], params_gauß_1_P3[2], params_gauß_1_P4[2], params_gauß_1_P5[2], params_gauß_1_P6[2], params_gauß_1_P7[2], params_gauß_1_P8[2], params_gauß_1_P9[2], params_gauß_1_P10[2], params_gauß_1_P11[2], params_gauß_1_P12[2], params_gauß_1_P13[2], params_gauß_1_P14[2], params_gauß_1_P15[2], params_gauß_1_P16[2], params_gauß_1_P17[2], params_gauß_1_P18[2], params_gauß_1_P19[2], params_gauß_1_P20[2]])
P = np.array([1.009, 2.000, 3.00, 4.00, 5.00, 6.01, 7.00, 8.00, 9.02, 10.00, 10.99, 12.00, 13.21, 13.97, 15.25, 16.13, 17.03, 18.05, 19.01, 20.15])

plt.plot(P, I_P/I_P.max(), "x", linewidth=0.8, label=r'Messung')
#plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P1), linewidth=0.8, label=r'Gauß Fit')
plt.xlabel(r'P [mW]')
plt.ylabel(r'I/I$_{max}$ [will. Einh.]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('I_P.pdf')
plt.clf()



######################### Überprüfen der Fits #########################

# plt.plot(lam_1_P1, I_1_P1, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P1), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P1.pdf')
# plt.clf()


# plt.plot(lam_1_P2, I_1_P2, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P2), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P2.pdf')
# plt.clf()


# plt.plot(lam_1_P3, I_1_P3, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P3), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P3.pdf')
# plt.clf()


# plt.plot(lam_1_P4, I_1_P4, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P4), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P4.pdf')
# plt.clf()


# plt.plot(lam_1_P5, I_1_P5, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P5), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P5.pdf')
# plt.clf()


# plt.plot(lam_1_P6, I_1_P6, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P6), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P6.pdf')
# plt.clf()


# plt.plot(lam_1_P7, I_1_P7, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P7), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P7.pdf')
# plt.clf()


# plt.plot(lam_1_P8, I_1_P8, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P8), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P8.pdf')
# plt.clf()


# plt.plot(lam_1_P9, I_1_P9, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P9), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P9.pdf')
# plt.clf()


# plt.plot(lam_1_P10, I_1_P10, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P10), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P10.pdf')
# plt.clf()


# plt.plot(lam_1_P11, I_1_P11, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P11), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P11.pdf')
# plt.clf()


# plt.plot(lam_1_P12, I_1_P12, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P12), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P12.pdf')
# plt.clf()


# plt.plot(lam_1_P13, I_1_P13, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P13), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P13.pdf')
# plt.clf()


# plt.plot(lam_1_P14, I_1_P14, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P14), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P14.pdf')
# plt.clf()


# plt.plot(lam_1_P15, I_1_P15, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P15), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P15.pdf')
# plt.clf()


# plt.plot(lam_1_P16, I_1_P16, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P16), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P16.pdf')
# plt.clf()


# plt.plot(lam_1_P17, I_1_P17, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P17), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P17.pdf')
# plt.clf()


# plt.plot(lam_1_P18, I_1_P18, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P18), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P18.pdf')
# plt.clf()


# plt.plot(lam_1_P19, I_1_P19, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P19), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P19.pdf')
# plt.clf()


# plt.plot(lam_1_P20, I_1_P20, linewidth=0.8, label=r'Messung')
# plt.plot(lambda_fit*10**(9), gauß(lambda_fit*10**(9), *params_gauß_1_P20), linewidth=0.8, label=r'Gauß Fit')
# plt.xlabel(r'$\lambda$ [nm]')
# plt.ylabel(r'I [will. Einh.]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('P20.pdf')
# plt.clf()











############################################ Unterschiede der PL-Spektren bei Anregung mit verschiedenen Wellenlängen ############################################

# params_gauß_1_448, cov_gauß_1_448 = curve_fit(gauß, lam_1_448*10**(-9), I_1_448)
# params_gauß_1_518, cov_gauß_1_518 = curve_fit(gauß, lam_1_518*10**(-9), I_1_518)
# params_gauß_1_636, cov_gauß_1_636 = curve_fit(gauß, lam_1_636*10**(-9), I_1_636/(3))

# params_gauß_2_448, cov_gauß_2_448 = curve_fit(gauß, lam_2_448*10**(-9), I_2_448/(0.4))
# params_gauß_2_518, cov_gauß_2_518 = curve_fit(gauß, lam_2_518*10**(-9), I_2_518/(2))
# params_gauß_2_636, cov_gauß_2_636 = curve_fit(gauß, lam_2_636*10**(-9), I_2_636/(1))

# params_gauß_3_448, cov_gauß_3_448 = curve_fit(gauß, lam_3_448*10**(-9), I_3_448/(0.2))
# params_gauß_3_518, cov_gauß_3_518 = curve_fit(gauß, lam_3_518*10**(-9), I_3_518/(1))
# params_gauß_3_636, cov_gauß_3_636 = curve_fit(gauß, lam_3_636*10**(-9), I_3_636/(0.75))

# params_gauß_4_448, cov_gauß_4_448 = curve_fit(gauß, lam_4_448*10**(-9), I_4_448/(0.5))
# params_gauß_4_518, cov_gauß_4_518 = curve_fit(gauß, lam_4_518*10**(-9), I_4_518/(2))
# params_gauß_4_636, cov_gauß_4_636 = curve_fit(gauß, lam_4_636*10**(-9), I_4_636/(1))


plt.plot(lam_1_pol0, savgol_filter(I_1_pol0, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=405$nm')
plt.plot(lam_1_448, savgol_filter(I_1_448, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=448$nm')
plt.plot(lam_1_518, savgol_filter(I_1_518, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=518$nm')
plt.plot(lam_1_636, savgol_filter(I_1_636, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=636$nm')
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'I [will. Einh.]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('Probe_1_lambda.pdf')
plt.clf()


plt.plot(lam_2_pol0, savgol_filter(I_2_pol0, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=405$nm')
plt.plot(lam_2_448, savgol_filter(I_2_448, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=448$nm')
plt.plot(lam_2_518, savgol_filter(I_2_518, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=518$nm')
plt.plot(lam_2_636, savgol_filter(I_2_636, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=636$nm')
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'I [will. Einh.]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('Probe_2_lambda.pdf')
plt.clf()


plt.plot(lam_3_pol0, savgol_filter(I_3_pol0, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=405$nm')
plt.plot(lam_3_448, savgol_filter(I_3_448, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=448$nm')
plt.plot(lam_3_518, savgol_filter(I_3_518, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=518$nm')
plt.plot(lam_3_636, savgol_filter(I_3_636, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=636$nm')
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'I [will. Einh.]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('Probe_3_lambda.pdf')
plt.clf()


plt.plot(lam_4_pol0, savgol_filter(I_4_pol0, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=405$nm')
plt.plot(lam_4_448, savgol_filter(I_4_448, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=448$nm')
plt.plot(lam_4_518, savgol_filter(I_4_518, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=518$nm')
plt.plot(lam_4_636, savgol_filter(I_4_636, 51, 3), linewidth=0.4, label=r'$\lambda_{Laser}=636$nm')
plt.xlabel(r'$\lambda$ [nm]')
plt.ylabel(r'I [will. Einh.]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('Probe_4_lambda.pdf')
plt.clf()