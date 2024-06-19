import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.ticker import FormatStrFormatter

plt.rcParams.update(
    {'font.family': 'serif', 'font.serif': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 24})

eff1s = np.genfromtxt('best_eta_1_5_9_16.dat')
eff2s = np.genfromtxt('best_eta_2_5_9_16.dat')


eff1m = np.genfromtxt('best_eta_1_5_12_12.dat')
eff2m = np.genfromtxt('best_eta_2_5_12_12.dat')


eff1l = np.genfromtxt('best_eta_1_5_16_9.dat')
eff2l = np.genfromtxt('best_eta_2_5_16_9.dat')



'''case1 = [eff1s, eff1m, eff1l]
case2 = [eff2s, eff2m, eff2l]
case3 = [eff3s, eff3m, eff3l]'''

case1 = [eff1s, eff1m, eff1l]
case2 = [eff2s, eff2m, eff2l]


fig, ax = plt.subplots(figsize=(7, 7))
ax.set_position([0.16, 0.17, 0.8, 0.8])
ax.grid(axis='y', ls='--', linewidth=1, alpha=0.5)
boxprops = dict(linestyle='--', linewidth=2, color='black')
flierprops = dict(marker='o', markersize=5,
                  linestyle='none')
medianprops = dict(linestyle='-', linewidth=2, color='black')
meanpointprops = dict(marker='D', markeredgecolor='black',
                      markerfacecolor='firebrick')
whiskerprops = dict(linestyle='-', linewidth=2)
meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
capprops = dict(linestyle='-', linewidth=2, color='Black')
bpl = plt.boxplot(case1, positions=np.array(range(len(case1))) * 4.0 - 0.5, sym='', widths=0.3, boxprops=boxprops,
                  vert=True, flierprops=flierprops, medianprops=medianprops,
                  patch_artist=True, whiskerprops=whiskerprops, capprops=capprops)
bpr = plt.boxplot(case2, positions=np.array(range(len(case2))) * 4.0 + 0.0, sym='', widths=0.3, boxprops=boxprops,
                  vert=True, flierprops=flierprops, medianprops=medianprops,
                  patch_artist=True, whiskerprops=whiskerprops, capprops=capprops)
'''bpm = plt.boxplot(case3, positions=np.array(range(len(case3))) * 4.0 + 0.5, sym='', widths=0.3, boxprops=boxprops,
                  vert=True, flierprops=flierprops, medianprops=medianprops,
                  patch_artist=True, whiskerprops=whiskerprops, capprops=capprops)'''


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['caps'], color='black')
    plt.setp(bp['medians'], color='black')


set_box_color(bpl, 'tab:blue')
set_box_color(bpr, 'tab:orange')
#set_box_color(bpm, 'tab:green')
plt.plot([], c='tab:blue', label='Flow condition 1')
plt.plot([], c='tab:orange', label='Flow condition 2')
#plt.plot([], c='tab:green', label=r'$16\times 16$ cell')
ticks = [r'$9:16$', r'$12:12$', r'$16:9$']
plt.xticks(np.linspace(0.0, 8, 3), ticks, fontsize=20)
plt.xlim(-2, 10)
#plt.ylim(0.775, 1.0)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(length=7, width=2, direction='in')
ax.tick_params(bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.minorticks_on()
ax.tick_params(which='minor', bottom=False, top=False, left=True, right=True)
ax.tick_params(which='minor', length=4, width=1.5, direction='in')
leg = ax.legend(fancybox=False, frameon=True, framealpha=1, loc='lower left')
leg.get_lines()[0].set_linewidth(10)
leg.get_lines()[1].set_linewidth(10)
#leg.get_lines()[2].set_linewidth(10)
leg.get_frame().set_edgecolor('black')
leg.get_frame().set_linewidth(2)
ax.set_xlabel('Farm aspect ratio')
ax.set_ylabel(r'Farm efficiency, $\eta_{\rm{farm}}$')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#fig.savefig('case_1_farm_ar.png', dpi=300)
plt.show()


'''fig, ax = plt.subplots(figsize=(17, 7))
ax.set_position([0.08, 0.17, 0.7, 0.7])
ax.grid(axis='y')
ax.boxplot([eff1l, eff2l, eff3l])
plt.show()'''


