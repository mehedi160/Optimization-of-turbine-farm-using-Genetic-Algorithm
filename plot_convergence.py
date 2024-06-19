import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.rcParams.update(
    {'font.family': 'serif', 'font.serif': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 15})

eff1s = np.genfromtxt('eta_1_5_9_16.dat')
eff1m = np.genfromtxt('eta_1_5_12_12.dat')
eff1l = np.genfromtxt('eta_1_5_16_9.dat')

eff2s = np.genfromtxt('eta_2_5_9_16.dat')
eff2m = np.genfromtxt('eta_2_5_12_12.dat')
eff2l = np.genfromtxt('eta_2_5_16_9.dat')


fig1, ax = plt.subplots(figsize=(7, 5))
ax.set_position([0.15, 0.12, 0.8, 0.8])
#ax.plot(range(len(eff1s[:, 0])), eff1s[:, 0], ls='-', linewidth=3, label=r'$9 \times 16$')
#ax.plot(range(len(eff1m[:, 0])), eff1m[:, 0], ls='-', linewidth=3, label=r'$12 \times 12$')
#ax.plot(range(len(eff1l[:, 0])), eff1l[:, 0], ls='-', linewidth=3, label=r'$16 \times 9$')
ax.plot(range(len(eff2s[:, 0])), eff2s[:, 0], ls='-', linewidth=3, label=r'$9 \times 16$')
ax.plot(range(len(eff2m[:, 0])), eff2m[:, 0], ls='-', linewidth=3, label=r'$12 \times 12$')
ax.plot(range(len(eff2l[:, 0])), eff2l[:, 0], ls='-', linewidth=3, label=r'$16 \times 9$')
ax.grid(True, ls='--', linewidth=1, alpha=0.2)
ax.set_xlabel('Generation')
ax.set_ylabel(r'Farm efficiency, $\eta_{\rm{farm}}$')
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(length=7, width=2, direction='in')
ax.tick_params(bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.minorticks_on()
ax.tick_params(which='minor', bottom=True, top=True, left=True, right=True)
ax.tick_params(which='minor', length=4, width=1.5, direction='in')
# plt.ylim(0.5, 1.0)
leg = ax.legend(fancybox=False, frameon=True, framealpha=1, loc=4)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(2)
#fig1.savefig('case1_1_5_conv.png', dpi=300)
fig1.savefig('case1_2_5_conv.png', dpi=300)
plt.show()
