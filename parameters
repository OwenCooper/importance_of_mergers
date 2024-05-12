# Extraction of key parameters for the 100 cMpc / h FABLE simulation
# ssh -X dc-coop5@login7.cosma.dur.ac.uk
# module load python/3.10.12
# cd /cosma7/data/

import h5py
f1 = h5py.File("/cosma7/data/dp012/dc-bigw2/FABLE-NewICs/FABLE-newICs-100/groups_135/fof_subhalo_tab_135.0.hdf5","r")
# f1['Header'].attrs contains ['BoxSize', 'FlagDoubleprecision', 'HubbleParam', 'Ngroups_ThisFile', 'Ngroups_Total', 'Nids_ThisFile', 'Nids_Total', 'Nsubgroups_ThisFile', 'Nsubgroups_Total', 'NumFiles', 'Omega0', 'OmegaLambda', 'Redshift', 'Time']

print("---KEY PARAMETERS AT Z=0---")
print("1) Co-moving box size = " + str(f1['Header'].attrs['BoxSize']) + " ckpc / h")
print("2) Hubble parameter, H = " + str(f1['Header'].attrs['HubbleParam'] * 100) + " km / s / Mpc")
print("3) Omega0 = " + str(f1['Header'].attrs['Omega0']))
print("4) OmegaLambda = " + str(f1['Header'].attrs['OmegaLambda']))
