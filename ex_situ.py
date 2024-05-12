# Ex situ stellar mass fraction of a random sample of galaxies as a function of time (Cosma7 Python code for 100 cMpc / h FABLE simulation)
# ssh -X dc-coop5@login7.cosma.dur.ac.uk
# module load python/3.10.12
# cd /cosma7/data/

import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
from multiprocessing import Pool
from astropy.cosmology import LambdaCDM
from random import sample
import math
import statistics
galSam = 20 # number of galaxies sampled
minMass = 10**9 # in solar masses
h0 = h5py.File("/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_000/fof_subhalo_tab_000.0.hdf5", 'r')['Header'].attrs['HubbleParam'] # = H0 / 100
f0 = h5py.File("dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_135/fof_subhalo_tab_135.0.hdf5", 'r')['Header'].attrs # accesses z~0 snapshot
cosmo = LambdaCDM(H0=f0['HubbleParam']*100, Om0=f0['Omega0'], Ode0=f0['OmegaLambda'])
# cosmo.age(z).value will now give the age of the snapshot in Gyr since the Big Bang as a number


# SECTION 1:  Read merger tree
tree = np.array(h5py.File("/cosma8/data/dp012/dc-jian5/merger_trees/FableFidFull/tree.hdf5", 'r')['Tree'])
# tree contains list of lists (subhalos) which contain numbers: [SubhaloID(0), SubhaloIDRaw(1), LastProgenitorID(2), MainLeafProgenitorID(3), RootDescendantID(4), TreeID(5), SnapNum(6), FirstProgenitorID(7), NextProgenitorID(8), DescendantID(9), FirstSubhaloInFOFGroupID(10), NextSubhaloInFOFGroupID(11), NumParticles(12), Mass(13), MassHistory(14), SubfindID(15)]


# SECTION 2:  List redshifts
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


# SECTION 3:  Get list of stars
def getStars(mainLine):
	data = []
	for snap in mainLine:
		# snap = [SubfindID, SnapNum]
		i = snap[1]
		if i < 10:
			iNew = "00" + str(i)
		if i < 100:
			iNew = "0" + str(i)
		else:
			iNew = str(i)
		stellarLen = np.array([])
		mass = np.array([])
		sft = np.array([])
		initMass = np.array([])
		haloLen = np.array([])
		haloNum = np.array([])
		for j in range(0,32):
			jNew = str(j)
			f1 = h5py.File("/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_" + iNew + "/fof_subhalo_tab_" + iNew + "." + jNew + ".hdf5","r")
			g1 = h5py.File("/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/snapdir_" + iNew + "/snap_" + iNew + "." + jNew + ".hdf5","r")['PartType4']
			if len(f1) > 0:	# checks data exists
				stellarLen = np.append(stellarLen, np.array(f1['Subhalo']['SubhaloLenType'][:,4]))
				haloLen = np.append(haloLen, np.array(f1['Group']['GroupLenType'][:,4]))
				haloNum = np.append(haloNum, np.array(f1['Subhalo']['SubhaloGrNr']))
			mass = np.append(mass, np.array(g1['Masses']) * 10 ** 10 /h0)
			sft = np.append(sft, np.array(g1['GFM_StellarFormationTime']))
			initMass = np.append(initMass, np.array(g1['GFM_InitialMass']))
		index = np.where(haloNum < int(haloNum[snap[0]]))
		fuzz = np.sum(haloLen[0:int(haloNum[snap[0]])]) - np.sum(stellarLen[index]) # number of non-subhalo-bound stars
		if snap[0] !=0:
			indexStart = int(np.sum(np.array(stellarLen)[0:snap[0]]) + fuzz)
		else:
			indexStart = 0
		indexEnd = int(indexStart + stellarLen[snap[0]])
		ind = np.where(np.array(sft) > 0)[0]
		ind = ind[(ind >= indexStart) & (ind < indexEnd)]
		data.append([mass[ind], 1 / sft[ind] - 1, initMass[ind] * 10 ** 10 + sft[ind]])
	return(data)


# SECTION 4:  Choose a random sample of galaxies at the largest available SnapNum
ind135 = sample(np.where((tree['SnapNum'] == 135) & (tree['Mass'] * 10**10 / h0 > minMass))[0].tolist(), galSam) # samples list of merger tree indexes of galaxies (with masses > minMass) at latest available snapshot (SnapNum=135) in merger tree
masses = tree['Mass'][ind135] * 10**10 / h0 # this is the list of masses [M_sun] at the last snapshot for use in colouring the lines in the figure
print(ind135)


# SECTION 5:  Obtain the main leaf progenitor line from the merger tree
def mainCode(i):
	index = ind135[i]
	index = np.where(tree['SubhaloID'] == tree['MainLeafProgenitorID'][index]) # goes to main leaf progenitor
	mainLine = [] # to store [SubfindID, SnapNum]
	mainLine.append([tree['SubfindID'][index].tolist()[0], tree['SnapNum'][index].tolist()[0]])
	while tree['DescendantID'][index] != -1:
		index = np.where(tree['SubhaloID'] == tree['DescendantID'][index]) # goes to descendant
		mainLine.append([tree['SubfindID'][index].tolist()[0], tree['SnapNum'][index].tolist()[0]])
	data = getStars(mainLine)
	exLst = np.array([]) # initialise ex situ stellar ID list = nothing
	ratio = [np.nan] # place holder for first snapshot (arbitarily zero otherwise)
	for i in range(1,len(data)):
		lst = np.array(data[i][2]) # list of IDs
		oldLst = np.array(data[i-1][2]) # list of old IDs
		exLst = np.intersect1d(exLst, lst) # remove stars no longer present in exLst
		newLst = np.setdiff1d(lst, oldLst) # find stellar IDs present in lst but not oldLst
		index1 = np.where(np.in1d(newLst, np.array(data[i][2]))) # new stars
		index2 = np.where(data[i][1] > zLst[mainLine[i][1]])
		index = np.intersect1d(index1, index2) # new ex situ stars
		exLst = np.append(exLst, lst[index]) # update ex situ star list
		index = np.where(np.in1d(data[i][2], exLst)) # all ex situ stars present
		exMass = np.sum(data[i][0][index]) # ex situ mass present
		totMass = np.sum(data[i][0]) # total mass
		ratio.append(exMass / totMass) # ex situ stellar mass fraction
	timePlot = []
	for i in range(len(mainLine)):
		timePlot.append(cosmo.age(zLst[mainLine[i][1]]).value)
	return([ratio, timePlot])

if __name__ == '__main__':
	with Pool(20) as p:
		results = p.map(mainCode, range(galSam))


# SECTION 6:  Plot ratio of ex situ vs total stellar mass fraction as a function of time for the sample of galaxies
# results = [[ratios, times], ...]
ax = plt.gca()
ax.set_facecolor('lightgrey')
plt.xlabel("Age of universe [Gyr]")
plt.xlim(0,14)
plt.ylabel("Ex situ stellar mass fraction")
plt.ylim(0,1)
# plot mean and shade between +- 1 standard deviation of mean
timeLst = []
for i in results:
	for j in i[1]:
		if j not in timeLst:
			timeLst.append(j)

timeLst = sorted(timeLst)
yMean = []
yStd = []
for t in timeLst:
	yTemp = []
	for i in results:
		for j in range(len(i[0])):
			if i[1][j] == t and math.isnan(i[0][j]) == False:
				yTemp.append(i[0][j])
	yMean.append(np.mean(yTemp))
	yStd.append(np.std(yTemp))

timeLst = np.array(timeLst)
yMean = np.array(yMean)
yStd = np.array(yStd)
plt.fill_between(timeLst, yMean - yStd, yMean + yStd, color='white', alpha=1, label='Standard deviation')
cmap = cm.jet
norm = colors.LogNorm(vmin=10**9, vmax=masses.max())

for j in range(len(results)):
	i = results[j]
	data = []
	for n in range(len(i[0])):
		data.append([i[1][n], i[0][n]])
	x, y = zip(*data)
	plt.plot(x, y, color=cmap(norm(masses[j])))

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, label='Galaxy mass at z=0 [$\mathrm{M_\u2609}$]')
plt.plot(timeLst, yMean, 'k-', linewidth=2.5, label='Mean', linestyle='dotted')
plt.legend()
legend = ax.legend(facecolor='darkgrey')
plt.show()
plt.savefig("/cosma7/data/dp012/dc-coop5/figureEXSITUvsTIME.png")
plt.close()


# SECTION 7:  Statistics
from scipy.stats import norm # import here as norm was previously used for another object
print("---STATISTICS AT LATE TIMES---")
finalRatios = []
for i in range(galSam):
	if results[i][0][-1] < 1:
		finalRatios.append(results[i][0][-1])

galSam = len(finalRatios) # analysis here excludes nan final ratios
print("SAMPLE SIZE = " + str(galSam) + " RANDOMLY CHOSEN GALAXIES")
print("LATEST TIME = " + str(results[0][1][-1]) + " Gyr")
print("MEAN EX SITU STELLAR MASS FRACTION  = " + str(statistics.mean(finalRatios)))
stdError = statistics.stdev(finalRatios) / np.sqrt(galSam - 1) # standard error of the mean for an unknown mean (calculation of the mean from the data removes one degree of freedom)
print("EX SITU STELLAR MASS FRACTION STANDARD ERROR OF THE MEAN = " + str(stdError))
z = (statistics.mean(finalRatios) - 0.5) / stdError
print("Z = " + str(z))
p = norm.cdf(z)
print("ONE-TAILED NORMAL DISTRIBUTION STATISTICAL TEST FOR MEAN < 0.5:  p = " + str(p * 100) + " %")
