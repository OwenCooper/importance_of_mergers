# Merger rate vs pair fraction code (Cosma7 Python code for 100 cMpc / h FABLE simulation)
# ssh -X dc-coop5@login7.cosma.dur.ac.uk
# module load python/3.10.12
# cd /cosma7/data/

import h5py
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from astropy.cosmology import LambdaCDM
from random import randint

f0 = h5py.File("dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_135/fof_subhalo_tab_135.0.hdf5", 'r')['Header'].attrs # accesses z~0 snapshot
h0 = f0['HubbleParam']
cosmo = LambdaCDM(H0=f0['HubbleParam']*100, Om0=f0['Omega0'], Ode0=f0['OmegaLambda'])
# cosmo.age(z) will now give the age of the snapshot in Gyr since the Big Bang
# cosmo.age(z).value gives the age as a plain number


# SECTION 1:  Read merger tree
# Read merger tree
tree = np.array(h5py.File("/cosma8/data/dp012/dc-jian5/merger_trees/FableFidFull/tree.hdf5", 'r')['Tree'])
# tree contains list of lists (subhalos) which contain numbers: [SubhaloID(0), SubhaloIDRaw(1), LastProgenitorID(2), MainLeafProgenitorID(3), RootDescendantID(4), TreeID(5), SnapNum(6), FirstProgenitorID(7), NextProgenitorID(8), DescendantID(9), FirstSubhaloInFOFGroupID(10), NextSubhaloInFOFGroupID(11), NumParticles(12), Mass(13), MassHistory(14), SubfindID(15)]


# SECTION 2:  Create list of time / Gyr values for all snapshots
def getTimes(i):
	# i = snapshot number = 0,...,135
	if i < 10:
		# 00x
		k = "00" + str(i)
	elif i < 100:
 		# 0xy
		k = "0" + str(i)
	else:
		# xyz
		k = str(i)
	z = h5py.File("dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_" + k + "/fof_subhalo_tab_" + k + ".0.hdf5", 'r')['Header'].attrs['Redshift']
	age = cosmo.age(z).value # / Gyr
	return(age)

if __name__ == '__main__':
	with Pool(20) as p:
		ageLst = p.map(getTimes, range(0,136))

print(ageLst)


# SECTION 3:  Calculate the pair fraction
# pair fraction = number of pairs / number of galaxies in sample
# this section divides 100 cMpc / h co-moving cube up into smaller boxes and randomly chooses a box to sample, where each sample box non-co-moving volume is approximately equal to the initial box size
# conditions for pairs:
rsep = 50 # kpc h^-1 maximum separation
vsep = 500 # km s^-1 maximum velocity difference
# label pairs as (i, j) where count over i>j (avoids self-/double-counting)
def pairFrac(folder):
	if folder < 10:
		# 00x
		k = "00" + str(folder)
	elif folder < 100:
 		# 0xy
		k = "0" + str(folder)
	else:
		# xyz
		k = str(folder)
	pos = np.empty((0,3))
	vel = np.empty((0,3))
	tempCount = 0
	for file in range(0,32):
		if len(h5py.File("dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_" + k + "/fof_subhalo_tab_" + k + "." + str(file) + ".hdf5", 'r')['Subhalo']) > 0:
			# this loop checks the data exists (throws errors otherwise)
			f1 = h5py.File("dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_" + k + "/fof_subhalo_tab_" + k + "." + str(file) + ".hdf5", 'r')
			SubhaloPos = np.array(f1['Subhalo']['SubhaloPos'])
			SubhaloVel = np.array(f1['Subhalo']['SubhaloVel'])
			Time = f1['Header'].attrs['Time'] # scale factor of snapshot
			boxLen = f1['Header'].attrs['BoxSize'] * Time
			pos = np.vstack((pos, SubhaloPos * Time)) # ckpc/h (co-moving) -> kpc/h ( * a) - the "c" stands for co-moving
			vel = np.vstack((vel, SubhaloVel))
	if len(pos) > 0:
		boxSplit = int(Time / h5py.File("dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_000/fof_subhalo_tab_000.0.hdf5", 'r')['Header'].attrs['Time']) # for pair counting, this divides the co-moving box up into boxSplit^3 little boxes
		box = np.random.randint(0, boxSplit - 1, 3)
		boxLen = f1['Header'].attrs['BoxSize'] * Time
		boxRange = np.array([box, box + 1]) * boxLen / boxSplit
		mask = np.all((pos > boxRange[0]) & (pos < boxRange[1]), axis = 1)
		pos = pos[mask]
		vel = vel[mask]
		pairCount = 0
	if len(pos) == 0:
		return("No galaxies present in snapshot")
	elif len(pos) == 1:
		# one galaxy present in snapshot so pair fraction = 0 arbitarily
		return(0)
	else:
		gals = len(pos)
		for i in range(gals):
			for j in range(i+1, gals):
				dx = abs(pos[i][0] - pos[j][0])
				dy = abs(pos[i][1] - pos[j][1])
				dz = abs(pos[i][2] - pos[j][2])
				dvx = vel[i][0] - vel[j][0]
				dvy = vel[i][1] - vel[j][1]
				dvz = vel[i][2] - vel[j][2]
				dx = dx if dx < boxLen else dx - boxLen
				dy = dy if dy < boxLen else dy - boxLen
				dz = dz if dz < boxLen else dz - boxLen
				dist = np.sqrt(dx**2 + dy**2 + dz**2)
				speedDiff = np.sqrt(dvx**2 + dvy**2 + dvz**2)
				if dist < rsep and speedDiff < vsep:
					pairCount += 1
		print(str(folder) + " / 135: pair fraction = " + str(pairCount / gals))
		return(pairCount / gals)

pairFraction = []
for i in range(5): # number of times pair fraction code is sampled
	if __name__ == '__main__':
		with Pool(20) as p:
			pairFractionAdd = p.map(pairFrac, range(0,136))
	pairFraction.append(pairFractionAdd)

print(pairFraction)

pairFraction = list(map(list, zip(*pairFraction))) # transposes 
pairFractionMean = []
pairFractionStd = []
for i in range(len(pairFraction)):
	lst = []
	for j in range(len(pairFraction[i])):
		if not type(pairFraction[i][j]) is str:
			lst.append(pairFraction[i][j])
	if len(lst) > 0:
		pairFractionMean.append(np.mean(lst))
		pairFractionStd.append(np.std(lst))
	else:
		pairFractionMean.append("No data")
		pairFractionStd.append("No data")

pairFraction = pairFractionMean


# SECTION 4:  Plot pair fraction vs time
ageLstNew = []
pairFractionNew = []
pairFractionStdNew = []
for i in range(0,136):
	if not type(pairFraction[i]) is str:
		pairFractionNew.append(pairFraction[i])
		ageLstNew.append(ageLst[i])
		pairFractionStdNew.append(pairFractionStd[i])

index = np.argmax(np.array(pairFractionNew))
maxPair = [pairFractionNew[index], pairFractionStdNew[index], ageLstNew[index], (ageLstNew[index + 1] - ageLstNew[index - 1]) / 2] # gives [max pair fraction(0), +- its error(1), time at which this maximum occurs(2), +- its error(3)]

plt.xlabel("Age of universe [Gyr]")
plt.ylabel("Pair fraction")
plt.xlim([0,14])
plt.errorbar(ageLstNew, pairFractionNew, pairFractionStdNew, linestyle='None', marker='x', ecolor='red', color='blue', capsize=3)
plt.plot()
plt.show()
plt.savefig("/cosma7/data/dp012/dc-coop5/figurePAIRFRACTIONvsTIME.png")
plt.close()


# SECTION 5:  Calculate merger count
# assume mergers occur pairwise so if at SnapNum N, there are n galaxies that have the same DescendantID then this corresponds to n-1 mergers happening between SnapNum N and N+1
mergerCount = [] # mergerCount is number of mergers / number of galaxies in snapshot
for snap in range(0,136):
	ind = np.where((tree['SnapNum'] == snap) & (tree['DescendantID'] != -1)) # selects correct snapshot and removes terminating subhalos
	if len(np.where(tree['SnapNum'] == snap)[0]) == 0:
		mergerCount.append("No galaxies")
	else:
		ind = np.where((tree['SnapNum'] == snap) & (tree['DescendantID'] != -1)) # selects correct snapshot and removes terminating subhalos
		arr = tree['DescendantID'][ind]
		u, c = np.unique(arr, return_counts=True) # u is values and c is the number of times they appear
		count = np.sum(c-1) # if n objects merge, there are (n-1) pairwise mergers - n=1 case (no merger) corresponds to n-1=0
		mergerCount.append(count / len(np.where(tree['SnapNum'] == snap)[0])) # merger count divided by number of galaxies in snapshot

print(mergerCount)


# SECTION 6:  Calculate merger rate (= number of mergers per unit time per galaxy)
# use results from a and c to calculate merger rate
dAgeLst = []
for i in range(0,135):
	dAgeLst.append(ageLst[i+1] - ageLst[i])

dAgeLst.append("N/A")
mergerRate = []
for i in range(0,136):
	if not type(mergerCount[i]) is str and not type(dAgeLst[i]) is str:
		mergerRate.append(mergerCount[i] / dAgeLst[i])
	else:
		mergerRate.append("N/A")

print(mergerRate)


# SECTION 7:  Plot merger rate vs time
ageLstNew = []
mergerRateNew = []
dAgeLstNew = []
for i in range(0,136):
	if not type(mergerRate[i]) is str:
		mergerRateNew.append(mergerRate[i])
		ageLstNew.append(ageLst[i])
		dAgeLstNew.append(dAgeLst[i])

index = np.argmax(np.array(mergerRateNew))
maxMerger = [mergerRateNew[index], 0, ageLstNew[index], (ageLstNew[index + 1] - ageLstNew[index - 1]) / 2] # gives [max merger rate(0), +- its error(1), time at which this maximum occurs(2), +- its error(3)]

fig, ax1 = plt.subplots()
curve1, = ax1.plot(ageLstNew, mergerRateNew, 'b', label='Merger rate per galaxy')
ax1.set_xlabel("Age of universe [Gyr]")
ax1.set_ylabel("Merger rate per galaxy [$\mathrm{Gyr^{-1}}$]")
plt.xlim([0,14])
ax2 = ax1.twinx()
curve2, = ax2.plot(ageLstNew, dAgeLstNew, 'r--', label='Time intervals')
ax2.set_ylabel('Time interval [Gyr]')
plt.legend(handles = [curve1, curve2])
plt.tight_layout()
plt.show()
plt.savefig("/cosma7/data/dp012/dc-coop5/figureMERGERRATEvsTIME.png")
plt.close()


# SECTION 8:  Plot merger rate as a function of pair fraction
# remove error messages:
pairFractionNew = []
pairFractionStdNew = []
mergerRateNew = []
ageLstNew = []
for i in range(0,136):
	if not type(pairFraction[i]) is str and not type(mergerRate[i]) is str:
		pairFractionNew.append(pairFraction[i])
		pairFractionStdNew.append(pairFractionStd[i])
		mergerRateNew.append(mergerRate[i])
		ageLstNew.append(ageLst[i])

plt.scatter(pairFractionNew,mergerRateNew, color='blue',marker='x')
plt.xlabel("Galaxy pair fraction")
plt.ylabel("Merger rate per galaxy [$\mathrm{Gyr^{-1}}$]")
plt.errorbar(pairFractionNew, mergerRateNew, xerr = pairFractionStdNew, fmt='x', ecolor='red', color='blue', capsize=3)
plt.plot()
plt.show()
plt.savefig("/cosma7/data/dp012/dc-coop5/figureMERGERRATEvsPAIRFRACTION.png")
plt.close()


# SECTION 9:  Plot characteristic merger time as a function of time
plt.xlabel("Age of universe [Gyr]")
plt.ylabel("Characteristic merger time [Gyr]")
plt.xlim([0,14])
# analysis of mean and standard deviation of data between 13 & 14 Gyr (previously calculated standard deviations ignored due to presence of zero standard deviations which leads to infinite weightings)
index = np.where((np.array(ageLstNew) > 13) & (np.array(ageLstNew) < 14))
mean = np.mean(np.array(pairFractionNew)[index] / np.array(mergerRateNew)[index])
std = np.std(np.array(pairFractionNew)[index] / np.array(mergerRateNew)[index])
plt.errorbar(np.array(ageLstNew), np.array(pairFractionNew) / np.array(mergerRateNew), np.array(pairFractionStdNew) * np.array(pairFractionNew) / np.array(mergerRateNew), fmt='x', ecolor='red', color='blue', capsize=3)
plt.plot()
plt.show()
plt.savefig("/cosma7/data/dp012/dc-coop5/figureMERGERTIMEvsTIME.png")
plt.close()


# SECTION 10:  Key results
print("---KEY RESULTS---")
print("PEAK PAIR FRACTION = " + str(maxPair[0]) + " +- " + str(maxPair[1]) + " AT T = " + str(maxPair[2]) + " +- " + str(maxPair[3]) + " Gyr")
print("PEAK MERGER RATE = " + str(maxMerger[0]) + " +- " + str(maxMerger[1]) + "Gyr^-1 AT T = " + str(maxMerger[2]) + " +- " + str(maxMerger[3]) + " Gyr")
print("BETWEEN 13-14 Gyr, MEAN CHARACTERISTIC MERGER TIME = " + str(mean) + " +- " + str(std / np.sqrt(len(index[0]) - 1)) + " Gyr") # converting standard deviation to standard error of the mean with -1 due to lost degree of freedom
