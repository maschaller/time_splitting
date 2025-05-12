#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:59:16 2025

@author: manuel
"""


import matplotlib.pyplot as plt


import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# # plt.rcParams.update(matplotlib.rcParamsDefault)
#plt.rcParams.update({'font.size': 22})
# # plt.rcParams.update({
# #     "text.usetex": True,
# #     "font.family": "Helvetica"
# # })


# plt.rcParams['xtick.major.pad']='8'
# plt.rcParams['ytick.major.pad']='8'
# plt.rcParams['axes.labelpad']='8'


with open('../data/PH ODEmu_1.pkl', 'rb') as f: 
     error1,normxtot1,normltot1, normutot1 = pickle.load(f)
     
with open('../data/PH ODEmu_5.pkl', 'rb') as f: 
     error2,normxtot2,normltot2, normutot2 = pickle.load(f)

with open('../data/PH ODEmu_10.pkl', 'rb') as f: 
     error3,normxtot3,normltot3, normutot3 = pickle.load(f)


# error over PR-Iterations
fig_soln, ax_soln = plt.subplots(1,3)#constrained_layout=True)
fig_soln.tight_layout(pad=1.3)

fig_soln.set_size_inches(6, 1.3)
fig_soln.set_dpi(500)

dum = 0.01*np.ones((len(error2[:,1]),1));


ax_soln[0].plot(error1[:, 0]/normxtot1, label='$\mu=1$')
ax_soln[0].plot(error2[:, 0]/normxtot2, label='$\mu=5$')
ax_soln[0].plot(error3[:, 0]/normxtot3, label='$\mu=10$')
ax_soln[0].set_title("rel. err. state")


ax_soln[1].plot(error1[:, 1]/normltot1, label='$\mu=1$')
ax_soln[1].plot(error2[:, 1]/normltot2, label='$\mu=5$')
ax_soln[1].plot(error3[:, 1]/normltot3, label='$\mu=10$')
ax_soln[1].set_title("rel. err. adjoint")


ax_soln[2].plot(error1[:, 2]/normutot1, label='$\mu=1$')
ax_soln[2].plot(error2[:, 2]/normutot2, label='$\mu=5$')
ax_soln[2].plot(error3[:, 2]/normutot3, label='$\mu=10$')
ax_soln[2].set_title("rel. err. control")

ax_soln[1].legend(loc="upper center", bbox_to_anchor=(0.53, 1.58), ncol=4)

for i in range(3):
    ax_soln[i].set_xticks([0,250,500])
    ax_soln[i].set_xlabel('Iterations')
    ax_soln[i].set_yscale('log')
    ax_soln[i].plot(dum, 'k--', linewidth=1.0)
    ax_soln[i].set_yticks([1e-4,1e-3,1e-2,1e-1])




plt.savefig('figs/phs.png', bbox_inches='tight', dpi=500)

plt.show()
