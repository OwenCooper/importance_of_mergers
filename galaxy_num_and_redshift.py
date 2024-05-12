# General 100 cMpc / h FABLE simulation properties:  Number of galaxies vs time and redshift vs time code (Cosma7 Python code for 100 cMpc / h FABLE simulation)
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


# SECTION 3:  Calculate number of galaxies
galCount = []
for snap in range(0,136):
	galCount.append(len(np.where(tree['SnapNum'] == snap)[0]))

print(galCount)


# SECTION 4:  Plot number of galaxies vs time
plt.xlabel("Age of universe [Gyr]")
plt.ylabel("Number of galaxies in universe")
plt.xlim([0,14])
plt.yscale('log')
plt.plot(ageLst, galCount)
plt.show()
plt.savefig("/cosma7/data/dp012/dc-coop5/figureGALAXYNUMvsTIME.png")
plt.close()


# SECTION 5:  Create list of redshift values for all snapshots
def getReds(i):
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
	return(z)

if __name__ == '__main__':
	with Pool(20) as p:
		zLst = p.map(getReds, range(0,136))

print(zLst)


# SECTION 6:  Plot number of galaxies vs time
plt.xlabel("Age of universe [Gyr]")
plt.ylabel("Redshift")
plt.xlim([0,14])
plt.plot(ageLst, zLst)
plt.show()
plt.savefig("/cosma7/data/dp012/dc-coop5/figureREDSHIFTvsTIME.png")
plt.close()


# SECTION 7:  Key results
print("---KEY RESULTS---")
print("MAXIMUM NUMBER OF GALAXIES = " + str(np.max(galCount)))
ind = np.where(galCount == np.max(galCount))[0][0]
print("TIME AT THIS MAXIMUM = " + str(ageLst[ind]) + " +- " + str((ageLst[ind + 1] - ageLst[ind - 1]) / 2) + " Gyr")

