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

with open('../data/Adv_Diff3_coarsemu_1.pkl', 'rb') as f: 
      error2c,normxtot2c,normltot2c, normutot2c = pickle.load(f)

with open('../data/Adv_Diff3_coarsemu_10.pkl', 'rb') as f: 
      error3c,normxtot3c,normltot3c, normutot3c = pickle.load(f)

with open('../data/Adv_Diff3_coarsemu_20.pkl', 'rb') as f: 
      error4c,normxtot4c,normltot4c, normutot4c = pickle.load(f)
     
with open('../data/Adv_Diff3mu_1.pkl', 'rb') as f: 
      error2,normxtot2,normltot2, normutot2 = pickle.load(f)

with open('../data/Adv_Diff3mu_10.pkl', 'rb') as f: 
      error3,normxtot3,normltot3, normutot3= pickle.load(f)

with open('../data/Adv_Diff3mu_20.pkl', 'rb') as f: 
      error4,normxtot4,normltot4, normutot4= pickle.load(f)
      
with open('../data/Adv_Diff3_finemu_1.pkl', 'rb') as f: 
      error2f,normxtot2f,normltot2f, normutot2f = pickle.load(f)

with open('../data/Adv_Diff3_finemu_10.pkl', 'rb') as f: 
      error3f,normxtot3f,normltot3f, normutot3f = pickle.load(f)

with open('../data/Adv_Diff3_finemu_20.pkl', 'rb') as f: 
      error4f,normxtot4f,normltot4f, normutot4f = pickle.load(f)

      
# error over PR-Iterations
fig_soln, ax_soln = plt.subplots(1,3)#constrained_layout=True)
fig_soln.tight_layout(pad=1.3)

fig_soln.set_size_inches(6, 1.3)
fig_soln.set_dpi(500)

dum = 0.01*np.ones((len(error2[:,1]),1));


#ax_soln[0].plot(error1[:, 0]/normxtot1, label='$\mu=0.1$')
ax_soln[0].plot(error2c[:, 2]/normutot2c, label='343 DOFs')
ax_soln[0].plot(error2[:, 2]/normutot2, label='1331 DOFs')
ax_soln[0].plot(error2f[:, 2]/normutot2f, label='2744 DOFs')
ax_soln[0].set_title("$\mu=1$")

ax_soln[1].plot(error3c[:, 2]/normutot3c, label='343 DOFs')
ax_soln[1].plot(error3[:, 2]/normutot3, label='1331 DOFs')
ax_soln[1].plot(error3f[:, 2]/normutot3f,label='2744 DOFs')
ax_soln[1].set_title("$\mu=10$")


#ax_soln[2].plot(error1[:, 2]/normutot1, label='$\mu=0.1$')
ax_soln[2].plot(error4c[:, 2]/normutot4c, label='343 DOFs')
ax_soln[2].plot(error4[:, 2]/normxtot4,  label='1331 DOFs')
ax_soln[2].plot(error4f[:, 2]/normutot4f, label='2744 DOFs')
ax_soln[2].set_title("$\mu=20$")

ax_soln[1].legend(loc="upper center", bbox_to_anchor=(0.53, 1.55), ncol=5)

for i in range(3):
    ax_soln[i].set_xticks([0,100,200,300])
    ax_soln[i].set_xlabel('Iterations')
    ax_soln[i].set_yscale('log')
    ax_soln[i].plot(dum, 'k--', linewidth=1.0)
    ax_soln[i].set_yticks([1e-5,1e-3,1e-2,1e-0])



plt.savefig('figs/heat_grids.png', bbox_inches='tight', dpi=500)

plt.show()
