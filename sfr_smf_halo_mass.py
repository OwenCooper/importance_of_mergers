# Star formation rate vs stellar mass for the 100 cMpc / h FABLE simulation heat maps (x4) + SMF (y-axis) vs halo mass (x-axis) for z=0,1,2,4 heat maps (x4).
# log scale colour map
# ssh -X dc-coop5@login7.cosma.dur.ac.uk
# module load python/3.10.12
# cd /cosma7/data/

import h5py
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

def model(x, a, b):
	# power law
	# takes log10(SMF) & returns log10(SFR)
	return(np.log10(a * ((10 ** x) / (10 ** 9)) ** b))

def modelBroken(x, a):
	# broken power law
	# takes log10(haloMass) and returns log10(SMF)
	return(np.log10(2 * (10 ** x) * a / (((10 ** x) / (10 ** d)) ** (- b) + ((10 ** x) / (10 ** d)) ** c)))

# data is in f1 = h5py.File("/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_i/fof_subhalo_tab_i.k.hdf5","r") for i=000-135 and k=0-31
# variables required:
# h0 = f1['Header'].attrs['HubbleParam'] (same for all snapshots)
# stellarMass = f1['Subhalo']['SubhaloMassType'][:,4] * 10 ** 10 /h0 (wind-phase cells are not included in type 4)
# starFormationRate = f1['Subhalo']['SubhaloSFR'] (for all gas cells)
# haloMass = f1['Subhalo']['SubhaloMass'] * (10 ** 10 / h0)
# g1 = h5py.File("/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/snapdir_i/snap_i.k.hdf5","r") for i=000-135 and k=0-31
# sft = g1['PartType4']['GFM_StellarFormationTime'] (>0 for stars and <0 for gas)

# assume every stellar particle is ~1e7 solar masses so only plot galaxies which have stellar mass > 2e8 solar masses
massResolutionLimit = 2*10**8
requiredRedshifts = [0,1,2,4]

# need to multiply stellar mass by 10^10 / Hubble parameter to convert stellar mass to M_sun units
h0 = h5py.File("/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_000/fof_subhalo_tab_000.0.hdf5", 'r')['Header'].attrs['HubbleParam'] # = H0 / 100


# SECTION 1:  Obtain redshifts and corresponding universe ages
def red(i):
	if i < 10:
		iNew = "00" + str(i)
	elif i < 100:
		iNew = "0" + str(i)
	else:
		iNew = str(i)
	return(h5py.File("/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_"+ iNew + "/fof_subhalo_tab_" + iNew + ".0.hdf5","r")['Header'].attrs['Redshift'])

if __name__ == '__main__':
	with Pool(20) as p:
		zLst = p.map(red, range(0,136))

# find iLst = list of snapshot numbers which correspond to required redshifts
iLst = []
for i in range(0,len(requiredRedshifts)):
	lst = []
	for j in range(0,len(zLst)):
		lst.append(abs(zLst[j] - requiredRedshifts[i]))
	for j in range(0,len(zLst)):
		if lst[j] == min(lst):
			iLst.append(j)


# SECTION 2:  Read data
def getData(n):
	i = iLst[n]
	if i < 10:
		iNew = "00" + str(i)
	elif i < 100:
		iNew = "0" + str(i)
	else:
		iNew = str(i)
	stellarMass = np.array([])
	sfr = np.array([])
	haloMass = np.array([])
	for j in range(0,32):
		jNew = str(j)
		f1 = h5py.File("/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_" + iNew + "/fof_subhalo_tab_" + iNew + "." + jNew + ".hdf5","r")
		if len(f1['Subhalo']) > 0:
			# checks data exists
			stellarMass = np.append(stellarMass, np.array(f1['Subhalo']['SubhaloMassType'][:,4]) * 10 ** 10 /h0)
			sfr = np.append(sfr, np.array(f1['Subhalo']['SubhaloSFR']))
			haloMass = np.append(haloMass, np.array(f1['Subhalo']['SubhaloMass']) * 10 ** 10 / h0)
	# now append only subhalos for which mass > 2e8 solar masses
	ind = np.where(np.array(stellarMass) > massResolutionLimit)
	return([stellarMass[ind], sfr[ind], haloMass[ind]])

if __name__ == '__main__':
	with Pool(5) as p:
		dataLst = p.map(getData, range(0,len(iLst)))


# SECTION 3:  Plot graphs
for i in range(0,len(requiredRedshifts)):
	cmapCust = plt.cm.colors.ListedColormap(['white', 'blue'])
	# SFT vs stellar mass
	z = str(requiredRedshifts[i]) # this is the redshift the graph is approximately at
	print("Exact redshift = " + str(zLst[iLst[i]])) # exact redshift of graph for records
	plt.loglog()
	plt.xlabel("Stellar mass [$\mathrm{M_\u2609}$]")
	plt.ylabel("Star formation rate [$\mathrm{M_\u2609 yr^{-1}}$]")
	x_space = np.logspace(np.log10(massResolutionLimit), 12, 100)
	y_space = np.logspace(-3, 2.5, 100)
	h = plt.hist2d(dataLst[i][0],dataLst[i][1], bins=(x_space, y_space), cmap=plt.cm.jet, norm=LogNorm(vmin = 1, vmax = 1000))
	colorbar = plt.colorbar(h[3])
	colorbar.set_label('Number of galaxies per pixel')
	# statistics
	xs = np.log10(np.array(dataLst[i][0]))
	ys = np.log10(np.array(dataLst[i][1]))
	xsNew = xs[(np.isfinite(xs)) & (np.isfinite(ys)) & (xs > np.log10(massResolutionLimit))]
	ysNew = ys[(np.isfinite(xs)) & (np.isfinite(ys)) & (xs > np.log10(massResolutionLimit))]
	xs = xsNew
	ys = ysNew
	popt, pcov = curve_fit(model, xs, ys, p0 = [10, 1], bounds = ([0, 0], [np.inf, np.inf]), method='trf', maxfev=5000)
	a, b = popt
	aError = np.sqrt(pcov[0, 0])
	bError = np.sqrt(pcov[1, 1])
	print("SFR vs SMF at z = " + z + ": Constant a = " + str(a) + " +- " + str(aError))
	print("SFR vs SMF at z = " + z + ": Constant b = " + str(b) + " +- " + str(bError))
	plt.plot(x_space, 10 ** model(np.log10(x_space), a, b), color = 'black',  linestyle='--', linewidth=2.5)
	plt.show()
	plt.savefig('/cosma7/data/dp012/dc-coop5/figureSFRvsSMFatz=' + z + '.png')
	plt.close()
	# SMF vs halo mass
	plt.loglog()
	plt.xlabel("Halo mass [$\mathrm{M_\u2609}$]")
	plt.ylabel("Stellar mass [$\mathrm{M_\u2609}$]")
	x_space = np.logspace(9, 14, 100)
	y_space = np.logspace(np.log10(massResolutionLimit), 12.5, 100)
	h = plt.hist2d(dataLst[i][2], dataLst[i][0], bins=(x_space, y_space), cmap=plt.cm.jet, norm = LogNorm(vmin = 1, vmax = 1000))
	colorbar = plt.colorbar(h[3])
	colorbar.set_label('Number of galaxies per pixel')
	# statistics
	xs = np.log10(np.array(dataLst[i][2]))
	ys = np.log10(np.array(dataLst[i][0]))
	xsNew = xs[(np.isfinite(xs)) & (np.isfinite(ys)) & (ys > np.log10(massResolutionLimit))]
	ysNew = ys[(np.isfinite(xs)) & (np.isfinite(ys)) & (ys > np.log10(massResolutionLimit))]
	xs = xsNew
	ys = ysNew
	# consideration of broken power law above and below halo mass ~ 10^12 solar masses
	popt, pcov = curve_fit(model, xs[(xs > 11) & (xs < 11.5)], ys[(xs > 11) & (xs < 11.5)], method='trf', maxfev=5000)
	b = popt[1] - 1
	bError = np.sqrt(pcov[1, 1])
	popt, pcov = curve_fit(model, xs[(xs > 12.5)], ys[(xs > 12.5)], method='trf', maxfev=5000)
	c = 1 - popt[1]
	cError = np.sqrt(pcov[1, 1])
	d = 11.88 # assumed by inspection of graph + literature value at z=0
	popt, pcov = curve_fit(modelBroken, xs, ys, p0 = [10**np.mean(ys - xs)], bounds = ([0], [np.inf]), method='trf', maxfev=10000)
	a = popt[0]
	aError = np.sqrt(pcov[0, 0])
	print("SMF vs halo mass at z = " + z + ": Constant a = " + str(a) + " +- " + str(aError))
	print("SMF vs halo mass at z = " + z + ": Constant b = " + str(b) + " +- " + str(bError))
	print("SMF vs halo mass at z = " + z + ": Constant c = " + str(c) + " +- " + str(cError))
	print("SMF vs halo mass at z = " + z + ": Constant d (assumed) = " + str(d))
	plt.plot(x_space, 10**modelBroken(np.log10(x_space), a), color = 'black',  linestyle='--', linewidth=2.5)
	plt.show()
	plt.savefig('/cosma7/data/dp012/dc-coop5/figureSMFvsHaloMassatz=' + z + '.png')
	plt.close()

