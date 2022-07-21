import numpy as np
import io
import scipy as scipy
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.signal import find_peaks 
import uncertainties.unumpy as unp 
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import pandas as pd

plt.rc('font', size=14, weight='normal')
#mpl.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['legend.fontsize'] = 14
plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14) 
plt.rcParams['figure.figsize'] = (14,6)



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

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


#Laserleistungen in mW aus Intensitäten in mA berechnen
def I_to_P(P):
    return 0.598 * P - 26.3

P_50 = 0.029 * 50**2 - 2.138 * 50 + 39.23
P_100 = 0.598 * 100 - 26.3
P_150 = 0.598 * 150 - 26.3
P_200 = 0.598 * 200 - 26.3
P_250 = 0.598 * 250 - 26.3


#Konversionsfaktoren zwischen Spannung und Position aus Programm

slope_x_50 = unp.uarray(464.65e-3, 208.65e-6)
slope_x_100 = unp.uarray(415.54e-3, 546.18e-6)
slope_x_150 = unp.uarray(385.12e-3, 153.33e-6)
slope_x_200 = unp.uarray(406.17e-3, 1.48e-3)
slope_x_250 = unp.uarray(384.26e-3, 416.79e-6)

slope_y_50 = unp.uarray(295.34e-3, 116.88e-6)
slope_y_100 = unp.uarray(282.78e-3, 66.24e-6)
slope_y_150 = unp.uarray(223.82e-3, 299.93e-6)
slope_y_200 = unp.uarray(244.29e-3, 340.64e-6)
slope_y_250 = unp.uarray(279.17e-3, 84.52e-6)


print(f"""
Laserleistung[mW]   Konversionsfaktor_x[V/micrometer]   Konversionsfaktor_y[V/micrometer]
{P_50}  ||| {slope_x_50}   |||  {slope_y_50}
{P_100} ||| {slope_x_100}  |||  {slope_y_100}
{P_150} ||| {slope_x_150}  |||  {slope_y_150}
{P_200} ||| {slope_x_200}  |||  {slope_y_200}
{P_250} ||| {slope_x_250}  |||  {slope_y_250}
""")

P = ([P_50, P_100, P_150, P_200, P_250])
slope_x = unp.uarray([noms(slope_x_50), noms(slope_x_100), noms(slope_x_150), noms(slope_x_200), noms(slope_x_250)], [stds(slope_x_50), stds(slope_x_100), stds(slope_x_150), stds(slope_x_200), stds(slope_x_250)])
slope_y = unp.uarray([noms(slope_y_50), noms(slope_y_100), noms(slope_y_150), noms(slope_y_200), noms(slope_y_250)], [stds(slope_y_50), stds(slope_y_100), stds(slope_y_150), stds(slope_y_200), stds(slope_y_250)])
# slope_y = unp.uarray([slope_y_50, slope_y_100, slope_y_150, slope_y_200, slope_y_250])

slope_x_mean = np.mean(noms(slope_x))
slope_y_mean = np.mean(noms(slope_y))

print(f"""
Konversionsfaktoren
x: {slope_x_mean}
y: {slope_y_mean}
""")


# plt.plot(P, noms(slope_x), "x", label=r'x-Konversionsfaktor')
# plt.plot(P, noms(slope_y), "x", label=r'y-Konversionsfaktor')
# plt.xlabel(r'P [mW]')
# plt.ylabel(r'K [V/$\mu$m]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('Konversion.pdf')
# plt.clf()



#Diodensummensignal mit Fit-Funktion auswerten und mit theoretischer Kurve vergleichen

def I_z(z, z0):
    return np.sqrt(1+(z/z0)**2) * np.sin(np.arctan(z/z0))

lam = 975*10**(-9)
n=1.33
w_0 = 1.22*lam/n * np.sqrt((n/1.25)**2 - 1)
z_0 = np.pi*w_0**2/lam *10**(9)

z, U = np.genfromtxt("Diodensumme_von_z.csv", delimiter=",", unpack=True)

guess = 0.01
I_z_params, I_z_cov = curve_fit(I_z, z[2:11]*10**(-3), U[2:11]/U.max(), p0=guess)
z_fit = np.linspace(2, 10, 1000) *10**(3)
print(I_z_params)


# plt.plot(z*10, U/U.max(), "x--", label=r'Messung')
# # plt.plot(z_fit, I_z(z_fit, z_0), "-", label=r'y-Theorie')
# # plt.plot(z_fit, I_z(z_fit, *I_z_params), "-", label=r'y-Theorie')
# plt.ylabel(r'I$_z$/I($Z$) [a.U.]')
# plt.xlabel(r'z [$\mu$m]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('Diodensumme.pdf')
# plt.clf()






#Fallensteifigkeiten bestimmen

def PSD(f, A, f_0):
    return A/(f**2 + f_0**2)

def k(f):
    return 3 * np.pi * 2.06*10**(-6) * 0.891e-3 * 2 * np.pi * f

def var(x):
    return np.var(x - np.mean(x))

def sin(t, A, w, b, t0):
    return A*np.sin(w*(t-t0)) + b



# # a) ohne ext. Krafteinwirkung

with open("OP_Sternikov/k_noForce_150mA.FDdat", 'rb') as f:
    clean_lines = (line.replace(b',',b'') for line in f)
    f_150, PSD_x_150, PSD_y_150, egal = np.genfromtxt(clean_lines, unpack=True)

PSD_x_150_params, PSD_x_150_cov = curve_fit(PSD, f_150[5:50000], PSD_x_150[5:50000])
PSD_y_150_params, PSD_y_150_cov = curve_fit(PSD, f_150[5:50000], PSD_y_150[5:50000])
print(PSD_x_150_params)


with open("OP_Sternikov/k_noForce_200mA.FDdat", 'rb') as f:
    clean_lines = (line.replace(b',',b'') for line in f)
    f_200, PSD_x_200, PSD_y_200, egal = np.genfromtxt(clean_lines, unpack=True)

PSD_x_200_params, PSD_x_200_cov = curve_fit(PSD, f_200[5:50000], PSD_x_200[5:50000])
PSD_y_200_params, PSD_y_200_cov = curve_fit(PSD, f_200[5:50000], PSD_y_200[5:50000])


with open("OP_Sternikov/k_noForce_250mA.FDdat", 'rb') as f:
    clean_lines = (line.replace(b',',b'') for line in f)
    f_250, PSD_x_250, PSD_y_250, egal = np.genfromtxt(clean_lines, unpack=True)

PSD_x_250_params, PSD_x_250_cov = curve_fit(PSD, f_250[5:50000], PSD_x_250[5:50000])
PSD_y_250_params, PSD_y_250_cov = curve_fit(PSD, f_250[5:50000], PSD_y_250[5:50000])


with open("OP_Sternikov/k_noForce_300mA.FDdat", 'rb') as f:
    clean_lines = (line.replace(b',',b'') for line in f)
    f_300, PSD_x_300, PSD_y_300, egal = np.genfromtxt(clean_lines, unpack=True)

PSD_x_300_params, PSD_x_300_cov = curve_fit(PSD, f_300[5:50000], PSD_x_300[5:50000])
PSD_y_300_params, PSD_y_300_cov = curve_fit(PSD, f_300[5:50000], PSD_y_300[5:50000])


with open("OP_Sternikov/k_noForce_350mA.FDdat", 'rb') as f:
    clean_lines = (line.replace(b',',b'') for line in f)
    f_350, PSD_x_350, PSD_y_350, egal = np.genfromtxt(clean_lines, unpack=True)

PSD_x_350_params, PSD_x_350_cov = curve_fit(PSD, f_350[5:50000], PSD_x_350[5:50000])
PSD_y_350_params, PSD_y_350_cov = curve_fit(PSD, f_350[5:50000], PSD_y_350[5:50000])



k_x_150 = k(PSD_x_150_params[1])
k_y_150 = k(PSD_y_150_params[1])

k_x_200 = k(PSD_x_200_params[1])
k_y_200 = k(PSD_y_200_params[1])

k_x_250 = k(PSD_x_250_params[1])
k_y_250 = k(PSD_y_250_params[1])

k_x_300 = k(PSD_x_300_params[1])
k_y_300 = k(PSD_y_300_params[1])

k_x_350 = k(PSD_x_350_params[1])
k_y_350 = k(PSD_y_350_params[1])

k_x = np.array([k_x_150, k_x_200, k_x_250, k_x_300, k_x_350])
k_y = np.array([k_y_150, k_y_200, k_y_250, k_y_300, k_y_350])

P_k = np.array([150, 200, 250, 300, 350])
P_k = I_to_P(P_k)

k_x_fit, cov = curve_fit(linear, P_k[1:4], k_x[1:4])
k_y_fit, cov = curve_fit(linear, P_k[1:4], k_y[1:4])


print(f"""
k_x_fit: m={k_x_fit[0]}     b={k_x_fit[1]}
k_y_fit: m={k_y_fit[0]}     b={k_y_fit[1]}
""")

plt.plot(P_k, k_x*10**(6), "x", label=r'$k_x$')
plt.plot(P_k, linear(P_k, *k_x_fit)*10**(6), "--", label=r'$k_x$-Anpassung')
plt.plot(P_k, k_y*10**(6), "x", label=r'$k_y$')
plt.plot(P_k, linear(P_k, *k_y_fit)*10**(6), "--", label=r'$k_y$-Anpassung')
plt.xlabel(r'P [mW]')
plt.ylabel(r'k [N/m$\cdot 10^{-6}$]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('k_noForce.pdf')
plt.clf()



t_150, x_150, y_150, egal = np.genfromtxt("OP_Sternikov/k_noForce_150mA.TDdat", unpack=True)
t_200, x_200, y_200, egal = np.genfromtxt("OP_Sternikov/k_noForce_200mA.TDdat", unpack=True)
t_250, x_250, y_250, egal = np.genfromtxt("OP_Sternikov/k_noForce_250mA.TDdat", unpack=True)
t_300, x_300, y_300, egal = np.genfromtxt("OP_Sternikov/k_noForce_300mA.TDdat", unpack=True)
t_350, x_350, y_350, egal = np.genfromtxt("OP_Sternikov/k_noForce_350mA.TDdat", unpack=True)


k_B_x_150 = k_x_150 * var(x_150*10**(-6)/slope_x_mean)/(25+273.15)
k_B_y_150 = k_y_150 * var(y_150*10**(-6)/slope_y_mean)/(25+273.15)
k_B_x_200 = k_x_200 * var(x_200*10**(-6)/slope_x_mean)/(25+273.15)
k_B_y_200 = k_y_200 * var(y_200*10**(-6)/slope_y_mean)/(25+273.15)
k_B_x_250 = k_x_250 * var(x_250*10**(-6)/slope_x_mean)/(25+273.15)
k_B_y_250 = k_y_250 * var(y_250*10**(-6)/slope_y_mean)/(25+273.15)
k_B_x_300 = k_x_300 * var(x_300*10**(-6)/slope_x_mean)/(25+273.15)
k_B_y_300 = k_y_300 * var(y_300*10**(-6)/slope_y_mean)/(25+273.15)
k_B_x_350 = k_x_350 * var(x_350*10**(-6)/slope_x_mean)/(25+273.15)
k_B_y_350 = k_y_350 * var(y_350*10**(-6)/slope_y_mean)/(25+273.15)



print(f"""
Ohne Krafteinwirkung

Laserleistung[mW]          k_x                   k_y                       Boltzmann_x         Boltzmann_y                      Abweichung x                        Abweichung y
{I_to_P(150)} |||        {k_x_150}                 {k_y_150}                     {k_B_x_150}         |||  {k_B_y_150}       {(k_B_x_150-const.k)/const.k}       {(k_B_y_150-const.k)/const.k}
{I_to_P(200)} |||        {k_x_200}                 {k_y_200}                     {k_B_x_200}         |||  {k_B_y_200}       {(k_B_x_200-const.k)/const.k}       {(k_B_y_200-const.k)/const.k}
{I_to_P(250)} |||        {k_x_250}                 {k_y_250}                     {k_B_x_250}         |||  {k_B_y_250}       {(k_B_x_250-const.k)/const.k}       {(k_B_y_250-const.k)/const.k}
{I_to_P(300)} |||        {k_x_300}                 {k_y_300}                     {k_B_x_300}         |||  {k_B_y_300}       {(k_B_x_300-const.k)/const.k}       {(k_B_y_300-const.k)/const.k}
{I_to_P(350)} |||        {k_x_350}                 {k_y_350}                     {k_B_x_350}         |||  {k_B_y_350}       {(k_B_x_350-const.k)/const.k}       {(k_B_y_350-const.k)/const.k}
""")




plt.rcParams["figure.figsize"] = (14,6)

plt.plot(f_150, PSD_x_150, "-", label=r'Messung')
plt.plot(f_150, PSD(f_150, *PSD_x_150_params), "-", label=r'Anpassung')
plt.axvspan(f_150[5], f_150[53000], alpha=0.3, color="grey", label=r'Anpassungsbereich')
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r'PSD [$\mu$m$\sqrt{s}$]')
plt.xlabel(r'f [Hz]')
plt.legend(loc='best')
plt.savefig('freq_y.pdf')
plt.clf()

plt.plot(f_150, PSD_y_150, "-", label=r'Messung')
plt.plot(f_150, PSD(f_150, *PSD_y_150_params), "-", label=r'Anpassung')
plt.axvspan(f_150[5], f_150[53000], alpha=0.3, color="grey", label=r'Anpassungsbereich')
plt.xscale("log")
plt.yscale("log")
plt.ylabel(r'PSD [$\mu$m$\sqrt{s}$]')
plt.xlabel(r'f [Hz]')
plt.legend(loc='best')
plt.savefig('freq_x.pdf')
plt.clf()

# plt.plot(t, savgol_filter(y, 3001, 3), "-", linewidth=0.8, label=r'y')
# plt.plot(t, savgol_filter(x, 99, 6), "-", linewidth=0.8, label=r'x')
plt.plot(t_150, savgol_filter(smooth(x_150/slope_x_mean, 5000), 51, 3), "-")
plt.ylabel(r'x [$\mu$m]')
plt.xlabel(r't [s]')
# plt.legend(loc='best')
plt.savefig('force_x.pdf')
plt.clf()

# plt.plot(t, savgol_filter(y, 3001, 3), "-", linewidth=0.8, label=r'y')
# plt.plot(t, savgol_filter(x, 99, 6), "-", linewidth=0.8, label=r'x')
plt.plot(t_150, savgol_filter(smooth(y_150/slope_y_mean, 5000), 51, 3), "-")
plt.ylabel(r'x [$\mu$m]')
plt.xlabel(r't [s]')
# plt.legend(loc='best')
plt.savefig('force_y.pdf')
plt.clf()


# b) mit ext. Kraft in x-Richtung

t_F_150, x_F_150, y_F_150, egal = np.genfromtxt("OP_Sternikov/k_xForce_150mA.TDdat", unpack=True)
t_F_200, x_F_200, y_F_200, egal = np.genfromtxt("OP_Sternikov/k_xForce_200mA.TDdat", unpack=True)
t_F_250, x_F_250, y_F_250, egal = np.genfromtxt("OP_Sternikov/k_xForce_250mA.TDdat", unpack=True)
t_F_300, x_F_300, y_F_300, egal = np.genfromtxt("OP_Sternikov/k_xForce_300mA.TDdat", unpack=True)
t_F_350, x_F_350, y_F_350, egal = np.genfromtxt("OP_Sternikov/k_xForce_350mA.TDdat", unpack=True)


k_F_x_150 = 2.64e-6
k_F_y_150 = 3.12e-6

k_F_x_200 = 3.11e-6
k_F_y_200 = 5.12e-6

k_F_x_250 = 2.06e-6
k_F_y_250 = 2.47e-6

k_F_x_300 = 1.71e-6
k_F_y_300 = 4.18e-6

k_F_x_350 = 1.47e-6
k_F_y_350 = 3.60e-6

k_F_x = np.array([k_F_x_150, k_F_x_200, k_F_x_250, k_F_x_300, k_F_x_350])
k_F_y = np.array([k_F_y_150, k_F_y_200, k_F_y_250, k_F_y_300, k_F_y_350])


k_B_F_x_150 = k_F_x_150 * var(x_F_150*10**(-6)/slope_x_mean)/(25+273.15)
k_B_F_y_150 = k_F_y_150 * var(y_F_150*10**(-6)/slope_y_mean)/(25+273.15)
k_B_F_x_200 = k_F_x_200 * var(x_F_200*10**(-6)/slope_x_mean)/(25+273.15)
k_B_F_y_200 = k_F_y_200 * var(y_F_200*10**(-6)/slope_y_mean)/(25+273.15)
k_B_F_x_250 = k_F_x_250 * var(x_F_250*10**(-6)/slope_x_mean)/(25+273.15)
k_B_F_y_250 = k_F_y_250 * var(y_F_250*10**(-6)/slope_y_mean)/(25+273.15)
k_B_F_x_300 = k_F_x_300 * var(x_F_300*10**(-6)/slope_x_mean)/(25+273.15)
k_B_F_y_300 = k_F_y_300 * var(y_F_300*10**(-6)/slope_y_mean)/(25+273.15)
k_B_F_x_350 = k_F_x_350 * var(x_F_350*10**(-6)/slope_x_mean)/(25+273.15)
k_B_F_y_350 = k_F_y_350 * var(y_F_350*10**(-6)/slope_y_mean)/(25+273.15)



print(f"""
Mit ext. Krafteinwirkung in x-Richtung

Laserleistung[mW]          k_x                   k_y                       Boltzmann_x         Boltzmann_y                      Abweichung x                        Abweichung y
{I_to_P(150)} |||        {k_F_x_150}                 {k_F_y_150}                     {k_B_F_x_150}         |||  {k_B_F_y_150}       {(k_B_F_x_150-const.k)/const.k}       {(k_B_F_y_150-const.k)/const.k}
{I_to_P(200)} |||        {k_F_x_200}                 {k_F_y_200}                     {k_B_F_x_200}         |||  {k_B_F_y_200}       {(k_B_F_x_200-const.k)/const.k}       {(k_B_F_y_200-const.k)/const.k}
{I_to_P(250)} |||        {k_F_x_250}                 {k_F_y_250}                     {k_B_F_x_250}         |||  {k_B_F_y_250}       {(k_B_F_x_250-const.k)/const.k}       {(k_B_F_y_250-const.k)/const.k}
{I_to_P(300)} |||        {k_F_x_300}                 {k_F_y_300}                     {k_B_F_x_300}         |||  {k_B_F_y_300}       {(k_B_F_x_300-const.k)/const.k}       {(k_B_F_y_300-const.k)/const.k}
{I_to_P(350)} |||        {k_F_x_350}                 {k_F_y_350}                     {k_B_F_x_350}         |||  {k_B_F_y_350}       {(k_B_F_x_350-const.k)/const.k}       {(k_B_F_y_350-const.k)/const.k}
""")



# print(f"""
# Mit ext. Krafteinwirkung in x-Richtung

# Laserleistung[mW]   Boltzmann_x         Boltzmann_y
# 150 |||         {k_B_F_x_150}         |||  {k_B_F_y_150}
# 200 |||         {k_B_F_x_200}         |||  {k_B_F_y_200}
# 250 |||         {k_B_F_x_250}         |||  {k_B_F_y_250}
# 300 |||         {k_B_F_x_300}         |||  {k_B_F_y_300}
# 350 |||         {k_B_F_x_350}         |||  {k_B_F_y_350}
# """)



# plt.plot(P_k, k_F_x*10**(6), "x--", label=r'$k_x$')
# plt.plot(P_k, k_F_y*10**(6), "x--", label=r'$k_y$')
# plt.xlabel(r'P [mW]')
# plt.ylabel(r'k [N/m$\cdot 10^{-6}$]')
# #plt.xlim(-30,30)
# #plt.ylim(-30,30)
# plt.legend(loc='best')
# plt.savefig('k_xForce.pdf')
# plt.clf()


# with open("OP_Sternikov/k_xForce_150mA.FDdat", 'rb') as f:
#     clean_lines = (line.replace(b',',b'') for line in f)
#     f_F_150, PSD_x_F_150, PSD_y_F_150, egal = np.genfromtxt(clean_lines, unpack=True)

# with open("OP_Sternikov/k_xForce_200mA.FDdat", 'rb') as f:
#     clean_lines = (line.replace(b',',b'') for line in f)
#     f_F_200, PSD_x_F_200, PSD_y_F_200, egal = np.genfromtxt(clean_lines, unpack=True)

# with open("OP_Sternikov/k_xForce_250mA.FDdat", 'rb') as f:
#     clean_lines = (line.replace(b',',b'') for line in f)
#     f_F_250, PSD_x_F_250, PSD_y_F_250, egal = np.genfromtxt(clean_lines, unpack=True)

# with open("OP_Sternikov/k_xForce_300mA.FDdat", 'rb') as f:
#     clean_lines = (line.replace(b',',b'') for line in f)
#     f_F_300, PSD_x_F_300, PSD_y_F_300, egal = np.genfromtxt(clean_lines, unpack=True)

# with open("OP_Sternikov/k_xForce_350mA.FDdat", 'rb') as f:
#     clean_lines = (line.replace(b',',b'') for line in f)
#     f_F_350, PSD_x_F_350, PSD_y_F_350, egal = np.genfromtxt(clean_lines, unpack=True)

# t_F_150, x_F_150, y_F_150, egal = np.genfromtxt("OP_Sternikov/k_xForce_150mA.TDdat", unpack=True)
# t_F_200, x_F_200, y_F_150, egal = np.genfromtxt("OP_Sternikov/k_xForce_200mA.TDdat", unpack=True)
# t_F_250, x_F_250, y_F_150, egal = np.genfromtxt("OP_Sternikov/k_xForce_250mA.TDdat", unpack=True)
# t_F_300, x_F_300, y_F_150, egal = np.genfromtxt("OP_Sternikov/k_xForce_300mA.TDdat", unpack=True)
# t_F_350, x_F_350, y_F_150, egal = np.genfromtxt("OP_Sternikov/k_xForce_350mA.TDdat", unpack=True)

# t_F_sin_150_params, t_F_sin_150_cov = curve_fit(sin, t_F_150, savgol_filter(smooth(x_F_150, 5000), 51, 3))




# plt.plot(t_F_150, savgol_filter(smooth(x_F_150, 5000), 51, 3), "-", label=r'x')
# plt.plot(t_F_150, sin(t_F_150, *t_F_sin_150_params), "-", label=r'fit')
# plt.ylabel(r'x [$\mu$m]')
# plt.xlabel(r't [s]')
# plt.grid()
# plt.legend(loc='best')
# plt.savefig('oszi.pdf')
# plt.clf()




# c) mit ext. Kraft in x-Richtung + Vortex-Retarder


t_vortex_150, x_vortex_150, y_vortex_150, egal = np.genfromtxt("OP_Sternikov/k_vortex_150mA.TDdat", unpack=True)
t_vortex_200, x_vortex_200, y_vortex_200, egal = np.genfromtxt("OP_Sternikov/k_vortex_200mA.TDdat", unpack=True)
t_vortex_250, x_vortex_250, y_vortex_250, egal = np.genfromtxt("OP_Sternikov/k_vortex_250mA.TDdat", unpack=True)
t_vortex_300, x_vortex_300, y_vortex_300, egal = np.genfromtxt("OP_Sternikov/k_vortex_300mA.TDdat", unpack=True)
t_vortex_350, x_vortex_350, y_vortex_350, egal = np.genfromtxt("OP_Sternikov/k_vortex_350mA.TDdat", unpack=True)



k_vortex_x_150 = 4.23e-7
k_vortex_y_150 = 1.97e-6

k_vortex_x_200 = 2.47e-6
k_vortex_y_200 = 1.08e-5

k_vortex_x_250 = 2.40e-6
k_vortex_y_250 = 3.61e-6

k_vortex_x_300 = 1.07e-6
k_vortex_y_300 = 2.39e-5

k_vortex_x_350 = 4.78e-6
k_vortex_y_350 = 7.20e-5

k_vortex_x = np.array([k_vortex_x_150, k_vortex_x_200, k_vortex_x_250, k_vortex_x_300, k_vortex_x_350])
k_vortex_y = np.array([k_vortex_y_150, k_vortex_y_200, k_vortex_y_250, k_vortex_y_300, k_vortex_y_350])



k_B_vortex_x_150 = k_vortex_x_150 * var(x_vortex_150*10**(-6)/slope_x_mean)/(25+273.15)
k_B_vortex_y_150 = k_vortex_y_150 * var(y_vortex_150*10**(-6)/slope_y_mean)/(25+273.15)
k_B_vortex_x_200 = k_vortex_x_200 * var(x_vortex_200*10**(-6)/slope_x_mean)/(25+273.15)
k_B_vortex_y_200 = k_vortex_y_200 * var(y_vortex_200*10**(-6)/slope_y_mean)/(25+273.15)
k_B_vortex_x_250 = k_vortex_x_250 * var(x_vortex_250*10**(-6)/slope_x_mean)/(25+273.15)
k_B_vortex_y_250 = k_vortex_y_250 * var(y_vortex_250*10**(-6)/slope_y_mean)/(25+273.15)
k_B_vortex_x_300 = k_vortex_x_300 * var(x_vortex_300*10**(-6)/slope_x_mean)/(25+273.15)
k_B_vortex_y_300 = k_vortex_y_300 * var(y_vortex_300*10**(-6)/slope_y_mean)/(25+273.15)
k_B_vortex_x_350 = k_vortex_x_350 * var(x_vortex_350*10**(-6)/slope_x_mean)/(25+273.15)
k_B_vortex_y_350 = k_vortex_y_350 * var(y_vortex_350*10**(-6)/slope_y_mean)/(25+273.15)







print(f"""
Mit ext. Krafteinwirkung in x-Richtung und Vortex-Retarder

Laserleistung[mW]          k_x                             k_y                       Boltzmann_x                Boltzmann_y                      Abweichung x                        Abweichung y
{I_to_P(150)} |||        {k_vortex_x_150}                 {k_vortex_y_150}                     {k_B_vortex_x_150}         |||  {k_B_vortex_y_150}      |||  {(k_B_vortex_x_150-const.k)/const.k}|||  {(k_B_vortex_y_150-const.k)/const.k}
{I_to_P(200)} |||        {k_vortex_x_200}                 {k_vortex_y_200}                     {k_B_vortex_x_200}         |||  {k_B_vortex_y_200}      |||  {(k_B_vortex_x_200-const.k)/const.k}|||  {(k_B_vortex_y_200-const.k)/const.k}
{I_to_P(250)} |||        {k_vortex_x_250}                 {k_vortex_y_250}                     {k_B_vortex_x_250}         |||  {k_B_vortex_y_250}      |||  {(k_B_vortex_x_250-const.k)/const.k}|||  {(k_B_vortex_y_250-const.k)/const.k}
{I_to_P(300)} |||        {k_vortex_x_300}                 {k_vortex_y_300}                     {k_B_vortex_x_300}         |||  {k_B_vortex_y_300}      |||  {(k_B_vortex_x_300-const.k)/const.k}|||  {(k_B_vortex_y_300-const.k)/const.k}
{I_to_P(350)} |||        {k_vortex_x_350}                 {k_vortex_y_350}                     {k_B_vortex_x_350}         |||  {k_B_vortex_y_350}      |||  {(k_B_vortex_x_350-const.k)/const.k}|||  {(k_B_vortex_y_350-const.k)/const.k}
""")



# print(f"""
# Mit ext. Krafteinwirkung in x-Richtung und Vortex-Retarder

# Laserleistung[mW]   Boltzmann_x         Boltzmann_y
# 150 |||         {k_B_vortex_x_150}         |||  {k_B_vortex_y_150}
# 200 |||         {k_B_vortex_x_200}         |||  {k_B_vortex_y_200}
# 250 |||         {k_B_vortex_x_250}         |||  {k_B_vortex_y_250}
# 300 |||         {k_B_vortex_x_300}         |||  {k_B_vortex_y_300}
# 350 |||         {k_B_vortex_x_350}         |||  {k_B_vortex_y_350}
# """)




plt.plot(P_k, k_vortex_x*10**(6), "x--", label=r'$k_x$')
plt.plot(P_k, k_vortex_y*10**(6), "x--", label=r'$k_y$')
plt.xlabel(r'P [mW]')
plt.ylabel(r'k [N/m$\cdot 10^{-6}$]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('k_vortex.pdf')
plt.clf()


with open("OP_Sternikov/k_vortex_150mA.FDdat", 'rb') as f:
    clean_lines = (line.replace(b',',b'') for line in f)
    f_vortex_150, PSD_x_vortex_150, PSD_y_vortex_150, egal = np.genfromtxt(clean_lines, unpack=True)

with open("OP_Sternikov/k_vortex_200mA.FDdat", 'rb') as f:
    clean_lines = (line.replace(b',',b'') for line in f)
    f_vortex_200, PSD_x_vortex_200, PSD_y_vortex_200, egal = np.genfromtxt(clean_lines, unpack=True)

with open("OP_Sternikov/k_vortex_250mA.FDdat", 'rb') as f:
    clean_lines = (line.replace(b',',b'') for line in f)
    f_vortex_250, PSD_x_vortex_250, PSD_y_vortex_250, egal = np.genfromtxt(clean_lines, unpack=True)

with open("OP_Sternikov/k_vortex_300mA.FDdat", 'rb') as f:
    clean_lines = (line.replace(b',',b'') for line in f)
    f_vortex_300, PSD_x_vortex_300, PSD_y_vortex_300, egal = np.genfromtxt(clean_lines, unpack=True)

with open("OP_Sternikov/k_vortex_350mA.FDdat", 'rb') as f:
    clean_lines = (line.replace(b',',b'') for line in f)
    f_vortex_350, PSD_x_vortex_350, PSD_y_vortex_350, egal = np.genfromtxt(clean_lines, unpack=True)






#Untersuchung der Vesikel in einer Zwiebel


# a) Vesikelgröße
px_size = 2.06e-6 /65

ves_size = (21+19+28+21)*px_size/4


# b) Vesikelgeschwindigkeit

I_v_x, I_v_y, I_v_sum = np.genfromtxt("OP_Sternikov/Speed.dat", unpack=True)
t_v = np.linspace(0, 7, 70000)

ves_speed = 2*ves_size/1

plt.plot(t_v, savgol_filter(I_v_sum/I_v_sum.max(), 51, 3), "-")
plt.axvspan(1.12, 2.12, alpha=0.3, color="grey", label=r"$\Delta$t=1s")
plt.xlabel(r't [s]')
plt.ylabel(r'I [a.U.]')
#plt.xlim(-30,30)
#plt.ylim(-30,30)
plt.legend(loc='best')
plt.savefig('v_vesikel.pdf')
plt.clf()


print(f"""
Mittl. Vesikelgröße {ves_size*10**(6)} micrometer
Vesikelgeschwindigkeit {ves_speed*10**(6)} micrometer/s
{21*px_size*10**(6)}
{19*px_size*10**(6)}
{21*px_size*10**(6)}
{28*px_size*10**(6)}
Stopp bei P={I_to_P(210)}
Stopp bei k={(k_y_fit[0]+k_x_fit[0])/2 * I_to_P(210) + (k_y_fit[1]+k_x_fit[1])/2}
""")





