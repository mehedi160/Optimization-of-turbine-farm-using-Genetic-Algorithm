import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.interpolate import griddata
from matplotlib import transforms
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
from matplotlib import cm

plt.rcParams.update(
    {'font.family': 'serif', 'font.serif': 'Times New Roman', 'mathtext.fontset': 'stix', 'font.size': 10})
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
plt.rc('axes', labelsize=15)


def get_image(path):
    return OffsetImage(plt.imread(path, format="png"), zoom=0.05 / 2.3)


rows_ = np.array([9, 12, 16])
cols_ = np.array([16, 12, 9])

vel_ = np.array([1.5, 2.5])

x0_ = np.array([17.12, 7.829999999999373])

e_ = np.array([[0.127899, 0.49482], [0.0948, 0.2613]])

pop_ = np.array([['best_layouts_1_5_9_16.dat', 'best_layouts_1_5_12_12.dat', 'best_layouts_1_5_16_9.dat'],
                 ['best_layouts_2_5_9_16.dat', 'best_layouts_2_5_12_12.dat', 'best_layouts_2_5_16_9.dat']])

vel_fig = np.array([['case1_1_5_aspect_ratio_9_16_vel.png', 'case1_1_5_aspect_ratio_12_12_vel.png',
                     'case1_1_5_aspect_ratio_16_9_vel.png'],
                    ['case1_2_5_aspect_ratio_9_16_vel.png', 'case1_2_5_aspect_ratio_12_12_vel.png',
                     'case1_2_5_aspect_ratio_16_9_vel.png']])

cell_fig = np.array([['case1_1_5_aspect_ratio_9_16_cell.png', 'case1_1_5_aspect_ratio_12_12_cell.png',
                      'case1_1_5_aspect_ratio_16_9_cell.png'],
                     ['case1_2_5_aspect_ratio_9_16_cell.png', 'case1_2_5_aspect_ratio_12_12_cell.png',
                      'case1_2_5_aspect_ratio_16_9_cell.png']])

power_fig = np.array([['case1_1_5_aspect_ratio_9_16_power.png', 'case1_1_5_aspect_ratio_12_12_power.png',
                       'case1_1_5_aspect_ratio_16_9_power.png'],
                      ['case1_2_5_aspect_ratio_9_16_power.png', 'case1_2_5_aspect_ratio_12_12_power.png',
                       'case1_2_5_aspect_ratio_16_9_power.png']])
lll = np.array([0.0, 0.4])

for ii in range(3):
    for jj in range(2):
        rows = rows_[ii]
        cols = cols_[ii]
        vel = vel_[jj]
        x0 = x0_[jj]
        e = e_[jj, :]
        pop = np.genfromtxt(pop_[jj, ii])
        N = 25
        layout = pop.reshape(rows, cols)
        ind = pop
        D = 5.4
        cell_width = D * 3.0  # unit : m
        cell_height = D * 2.5
        cell_width_half = 0.5 * cell_width
        R = D / 2

        D1 = D  # unit m
        D2 = D

        Cp = 4 * (e[0] * (1 - e[0]) ** 2 + (e[1] - 2 * e[0]) * (1 - e[1]) ** 2)
        D_x0 = np.sqrt((1 - e[0]) / (1 - 2 * e[0]) * D1 ** 2 - 2 * (2 * e[0] - e[1]) * (1 - e[1]) / (
                (1 - 2 * e[0]) * (1 - 2 * e[1] + 2 * e[0])) * D2 ** 2)
        d2 = D2 * np.sqrt((1 - e[1]) / (1 - 2 * e[1] + 2 * e[0]))
        rho = 998.2

        R1 = D1 / 2

        entrainment_const = (D_x0 - D1) / (2 * x0)

        ind_indices = 0
        ind_pos = np.zeros(N)
        for bb in range(rows * cols):
            if ind[bb] == 1:
                ind_pos[ind_indices] = bb
                ind_indices += 1
        xy_position = np.zeros((2, N), dtype=np.float32)  # x y position
        cr_position = np.zeros((2, N), dtype=np.int32)  # column row position
        r_i = np.floor(ind_pos / cols)
        c_i = np.floor(ind_pos - r_i * cols)
        cr_position[0, :] = c_i
        cr_position[1, :] = r_i
        xy_position[0, :] = c_i * cell_width + cell_width_half
        xy_position[1, :] = r_i * cell_height + 0.5 * cell_height


        def wake_calculate(trans_xy_position, N, vel, x0, R1, D1, D2, entrainment_const, e, D_x0, d2):
            # print(-trans_xy_position)
            sorted_index = np.argsort(-trans_xy_position[1, :])  # y value descending
            wake_deficiency = np.zeros(N, dtype=np.float32)
            # print(1-wake_deficiency)
            wake_deficiency[sorted_index[0]] = 0
            for i in range(1, N):
                for j in range(i):
                    xdis = np.absolute(trans_xy_position[0, sorted_index[i]] - trans_xy_position[0, sorted_index[j]])
                    ydis = np.absolute(trans_xy_position[1, sorted_index[i]] - trans_xy_position[1, sorted_index[j]])
                    d = cal_deficiency(dx=xdis, dy=ydis, r=R1, D1=D1, D2=D2, ec=entrainment_const, e=e, D_x0=D_x0,
                                       x0=x0, d2=d2,
                                       velocity=vel)
                    wake_deficiency[sorted_index[i]] += d ** 2

                wake_deficiency[sorted_index[i]] = np.sqrt(wake_deficiency[sorted_index[i]])

            return wake_deficiency


        # ec : entrainment_const = alpha in rad
        def cal_deficiency(dx, dy, r, D1, D2, ec, e, D_x0, x0, d2, velocity):
            if dy == 0:
                return 0
            R = r + ec * dy  # R = wake radius; r = turbine radius
            inter_area = cal_interaction_area(dx=dx, dy=dy, r=r, R=R)
            if dy < x0:
                Uij = velocity * (
                        ((1 - e[1]) * D2 ** 2 + (D_x0 ** 2 - d2 ** 2) * (1 - 2 * e[0])) / (D1 + 2 * ec * dy) ** 2)
            else:
                Uij = (velocity * (1 + 2 * e[0] - 2 * e[1]) * d2 ** 2 +
                       (D_x0 ** 2 - d2 ** 2) * velocity * (1 - 2 * e[0]) +
                       velocity * ((D1 + 2 * ec * dy) ** 2 - (D1 + 2 * ec * x0) ** 2)) / (
                              D1 + 2 * ec * dy) ** 2
            d = np.sqrt(inter_area / (np.pi * r ** 2)) * (velocity - Uij)
            return d


        def cal_interaction_area(dx, dy, r, R):
            if dx >= r + R:
                return 0
            elif dx >= np.sqrt(R ** 2 - r ** 2):
                alpha = np.arccos((R ** 2 + dx ** 2 - r ** 2) / (2 * R * dx))
                beta = np.arccos((r ** 2 + dx ** 2 - R ** 2) / (2 * r * dx))
                A1 = alpha * R ** 2
                A2 = beta * r ** 2
                A3 = R * dx * np.sin(alpha)
                return A1 + A2 - A3
            elif dx >= R - r:
                alpha = np.arccos((R ** 2 + dx ** 2 - r ** 2) / (2 * R * dx))
                beta = np.pi - np.arccos((r ** 2 + dx ** 2 - R ** 2) / (2 * r * dx))
                A1 = alpha * R ** 2
                A2 = beta * r ** 2
                A3 = R * dx * np.sin(alpha)
                return np.pi * r ** 2 - (A2 + A3 - A1)
            else:
                return np.pi * r ** 2


        def vel_field(U, D, x0, e, ec, d2, ch, rows, vel):
            D1 = D
            D2 = D
            dy0 = np.linspace(0.1, x0, 100)
            Uij0 = np.zeros(np.shape(dy0))
            for i in range(dy0.size):
                Uij0[i] = U * (((1 - e[1]) * D2 ** 2 + (D_x0 ** 2 - d2 ** 2) * (1 - 2 * e[0])) / (
                        D1 + 2 * ec * dy0[i]) ** 2)

            dy1 = np.linspace((x0 + .1), ch * rows, 200)
            Uij1 = np.zeros(np.shape(dy1))
            for i in range(dy1.size):
                Uij1[i] = (U * (1 + 2 * e[0] - 2 * e[1]) * d2 ** 2 +
                           (D_x0 ** 2 - d2 ** 2) * U * (1 - 2 * e[0]) +
                           U * ((D1 + 2 * ec * dy1[i]) ** 2 - (D1 + 2 * ec * x0) ** 2)) / (
                                  D1 + 2 * ec * dy1[i]) ** 2

            xD = np.concatenate([dy0, dy1]) / D
            Uij = np.concatenate([Uij0, Uij1]) / vel
            return xD, Uij


        speed_deficiency = wake_calculate(xy_position, N, vel, x0, R1, D1, D2, entrainment_const, e, D_x0, d2)
        actual_velocity = vel - speed_deficiency

        xD = np.zeros([np.size(actual_velocity), 300])
        Uij = np.zeros([np.size(actual_velocity), 300])
        for gg in range(np.size(actual_velocity)):
            xD[gg, :], Uij[gg, :] = vel_field(actual_velocity[gg], D, x0, e, entrainment_const, d2, cell_height, rows,
                                              vel)

        ############################################ VELOCITY FIELD PLOT #######################################################
        '''fig1, ax = plt.subplots(figsize=(5, 5))
        ax.set_position([0.15, 0.12, 0.7, 0.8])
        ax.plot(xy_position[0, :] / D, xy_position[1, :] / D, color='none')
        for x0, y0 in zip(xy_position[0, :] / D, xy_position[1, :] / D):
            ab = AnnotationBbox(get_image('turbine.png'), (x0, y0), frameon=False)
            ax.add_artist(ab)
        # ax.grid(True, ls='--', linewidth=1, alpha=0.2)
        ax.set_xlabel('$x/D$')
        ax.set_ylabel('$y/D$')
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(2)
        ax.tick_params(length=7, width=2, direction='in')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.minorticks_on()
        ax.tick_params(which='minor', bottom=True, top=True, left=True, right=True)
        ax.tick_params(which='minor', length=4, width=1.5, direction='in')
        plt.xlim(0.0, cell_width * cols / D)
        plt.ylim(0.0, cell_height * rows / D)
        level = np.linspace(lll[jj], 1.0, 21 + 1)
        ax.contourf(np.ones([300, 500]), levels=level, vmin=0.0, vmax=1.0, cmap='hot')
        x_c = np.linspace(0, cols * cell_width / D, 30)
        for kk in reversed(range(np.size(actual_velocity))):
            y_c1 = np.linspace(xy_position[1, kk] / D, xy_position[1, kk] / D - x0 / D, 100)
            y_c2 = np.linspace(xy_position[1, kk] / D - x0 / D - 0.1 / D,
                               xy_position[1, kk] / D - cell_height * rows / D, 200)
            y_c = np.concatenate([y_c1, y_c2])
            xi, yi = np.meshgrid(x_c, y_c)
            con = np.rot90(np.tile(Uij[kk, :], (30, 1)))
            CS = ax.contourf(xi, yi, np.flipud(con), levels=level, cmap='hot')
            clip_x = np.array([xy_position[0, kk] / D + 0.5, xy_position[0, kk] / D - 0.5,
                               xy_position[0, kk] / D - 0.5 - entrainment_const * (xy_position[1, kk] - 0) / D,
                               xy_position[0, kk] / D + 0.5 + entrainment_const * (xy_position[1, kk] - 0) / D,
                               xy_position[0, kk] / D + 0.5])
            clip_y = np.array([xy_position[1, kk] / D, xy_position[1, kk] / D, 0, 0, xy_position[1, kk] / D])
            clippath = Path(np.c_[clip_x, clip_y])
            patch = PathPatch(clippath, facecolor='none', edgecolor='none')
            ax.add_patch(patch)
            for c in CS.collections:
                c.set_clip_path(patch)

        divider = make_axes_locatable(ax)
        kwargs = {'format': '%.2f'}
        cax = divider.append_axes("right", size="6%", pad=0.15)
        plt.colorbar(CS, cax=cax, **kwargs).ax.tick_params(size=0)
        ax.set_aspect('equal')
        fig1.savefig(vel_fig[jj, ii], dpi=300)
        plt.show()

        ################################################# CELL PLOT ############################################################
        nbr = np.arange(1, rows * cols + 1, 1)
        cell_num = nbr.reshape(rows, cols)
        xlabels = np.arange(1, cols + 1, 1)
        ylabels = np.arange(1, rows + 1, 1)
        fig2, ax = plt.subplots(figsize=(6, 6))
        ax.set_position([0.12, 0.12, 0.8, 0.8])
        CS = ax.pcolor(layout, cmap='binary', edgecolor='black', linestyle='-', lw=1, vmax=3.0)
        ax.set_aspect('equal')
        ax.grid(True, ls='-', linewidth=1, color='black', alpha=1)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)
        ax.tick_params(length=4, width=1, direction='in')
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        for i in range(np.shape(layout)[0]):
            for j in range(np.shape(layout)[1]):
                text = ax.text(j + 0.5, i + 0.5, cell_num[i, j],
                               ha="center", va="center", color="black")

        ax.xaxis.set(ticks=np.arange(0.5, cols), ticklabels=xlabels)
        ax.yaxis.set(ticks=np.arange(0.5, rows), ticklabels=ylabels)
        ax.grid(False)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        fig2.savefig(cell_fig[jj, ii], dpi=300)
        plt.show()'''

        ################################################## POWER DISTRIBUTION PLOT #############################################
        fig3 = plt.figure(figsize=(6, 6))
        ax = fig3.add_subplot(111, projection=Axes3D.name)
        ax.set_position([0.005, 0.01, 0.9, 1.0])
        xlabels = np.arange(1, cols, 1)
        ylabels = np.arange(1, rows, 1)

        def make_bar(ax, x0=0, y0=0, width=0.5, height=1, cmap="jet",
                     norm=matplotlib.colors.Normalize(vmin=0, vmax=1), **kwargs):
            # Make data
            u = np.linspace(0, 2 * np.pi, 4 + 1) + np.pi / 4.
            v_ = np.linspace(np.pi / 4., 3. / 4 * np.pi, 100)
            v = np.linspace(0, np.pi, len(v_) + 2)
            v[0] = 0
            v[-1] = np.pi
            v[1:-1] = v_
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))

            xthr = np.sin(np.pi / 4.) ** 2
            zthr = np.sin(np.pi / 4.)
            x[x > xthr] = xthr
            x[x < -xthr] = -xthr
            y[y > xthr] = xthr
            y[y < -xthr] = -xthr
            z[z > zthr] = zthr
            z[z < -zthr] = -zthr

            x *= 1. / xthr * width
            y *= 1. / xthr * width
            z += zthr
            z *= height / (2. * zthr)
            # translate
            x += x0
            y += y0
            # plot
            ax.plot_surface(x, y, z, cmap=cmap, norm=norm, **kwargs, edgecolor='none')


        def make_bars(ax, x, y, height, width=1.0):
            widths = np.array(width) * np.ones_like(x)
            x = np.array(x).flatten()
            y = np.array(y).flatten()

            h = np.array(height).flatten()
            w = np.array(widths).flatten()
            norm = matplotlib.colors.Normalize(vmin=0, vmax=h.max())
            for i in range(len(x.flatten())):
                make_bar(ax, x0=x[i], y0=y[i], width=w[i], height=h[i], norm=norm)


        X, Y = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
        ind_indices = 0
        vel_lay = np.zeros(rows * cols)
        for ll in range(rows * cols):
            if ind[ll] == 1:
                vel_lay[ll] = actual_velocity[ind_indices]
                ind_indices += 1
        velocity_layout = vel_lay.reshape(rows, cols)
        Z = velocity_layout ** 3 / vel ** 3
        make_bars(ax, X, Y, Z, width=0.2)


        ax.tick_params(length=4, width=1, direction='in')
        ax.tick_params(bottom=False, top=False, left=False, right=False)
        ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_zlabel(r'$P_i / P_{rated}$')
        ax.xaxis.set(ticks=np.arange(1, cols), ticklabels=xlabels)
        ax.yaxis.set(ticks=np.arange(1, rows), ticklabels=ylabels)

        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        # Set new tick positions and labels, skipping every other one
        ax.set_xticks(xticks[::2])
        ax.set_yticks(yticks[::2])

        # Optionally, customize the tick labels to match the new tick positions
        ax.set_xticklabels(xticks[::2])
        ax.set_yticklabels(yticks[::2])

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.grid(False)
        fig3.savefig(power_fig[jj, ii], dpi=300)
        plt.show()



