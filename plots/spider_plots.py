import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


data = [['OpenSet Risk', 'Accessibility', 'Replication', 'Speed', 'Computational Demand', 'Maintainability', 'Community', 'Reliability', 'Practicability'],
        ('Basecase', [
            [0, 1, 1, 0.69, 0, 1, 0.4, 0.00, 0.90],
            [1, 1, 0.6, 0.00, 0, 0.3, 0.13, 0.00, 0.80],
            [0, 1, 0.6, 0.43, 0, 1, 0.59, 0.00, 0.10],
            [0, 1, 1, 0.69, 0, 1, 0.4, 0.00, 0.90],
            [0, 1, 0.6, 0.43, 0, 1, 0.59, 0.00, 0.10],
            [0, 1, 1, 0.69, 0, 1, 0.4, 0.00, 0.90],
            [1, 1, 0.6, 0.00, 0, 0.3, 0.13, 0.00, 0.80],
            [0, 1, 0.6, 0.43, 0, 1, 0.59, 0.00, 0.10],
            [0, 1, 1, 0.69, 0, 1, 0.4, 0.00, 0.90],
            [0, 1, 1, 0.69, 0, 1, 0.4, 0.00, 0.90],
            [0, 1, 0.6, 0.43, 0, 1, 0.59, 0.00, 0.10],
            [0, 1, 0.7, 1, 0, 0.8, 1, 0.00, 0.60]])]

labels_names = ["SoftMax", "OpenMax", "Energy OOD", "DOC","COOL","RNA","OpenNet","BNN","WNN","DCN","CR","ARSVR"]

N = len(data[0])
theta = radar_factory(N, frame='polygon')

spoke_labels = data.pop(0)
title, case_data = data[0]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='radar'))
fig.subplots_adjust(top=0.85, bottom=0.05)

ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
ax.set_title(title,  position=(0.5, 1.1), ha='center')

labels = iter(labels_names)

for d in case_data:
    line = ax.plot(theta, d)
    ax.fill(theta, d,  alpha=0.25, label=labels.__next__())
ax.set_varlabels(spoke_labels)
ax.legend(loc=4)
plt.savefig("./plot_all_losses.pdf", dpi=700)
plt.show()
#
#
# import numpy as np
# from bokeh.plotting import figure, show, output_file
# from bokeh.models import ColumnDataSource, LabelSet
#
# num_vars = 9
#
# centre = 0.5
#
# theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
# # rotate theta such that the first axis is at the top
# theta += np.pi/2
#
# def unit_poly_verts(theta, centre ):
#     """Return vertices of polygon for subplot axes.
#     This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
#     """
#     x0, y0, r = [centre ] * 3
#     verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
#     return verts
#
# def radar_patch(r, theta, centre ):
#     """ Returns the x and y coordinates corresponding to the magnitudes of
#     each variable displayed in the radar plot
#     """
#     # offset from centre of circle
#     offset = 0.01
#     yt = (r*centre + offset) * np.sin(theta) + centre
#     xt = (r*centre + offset) * np.cos(theta) + centre
#     return xt, yt
#
# verts = unit_poly_verts(theta, centre)
# x = [v[0] for v in verts]
# y = [v[1] for v in verts]
#
# p = figure(title="Baseline - Radar plot")
# text = ['OpenSet Risk', 'Accessibility', 'Replication', 'Speed', 'Computational Demand', 'Maintainability', 'Community', 'Reliability', 'Practicability']
# source = ColumnDataSource({'x':x + [centre ],'y':y + [1],'text':text})
#
# p.line(x="x", y="y", source=source)
#
# labels = LabelSet(x="x",y="y",text="text",source=source)
#
# p.add_layout(labels)
#
# # example factor:
# f1 = np.array([0, 1, 1, 0.69, 0, 1, 0.4, 0.00, 0.90])
# f2 = np.array([1, 1, 0.6, 0.00, 0, 0.3, 0.13, 0.00, 0.80])
# f3 = np.array([0, 1, 0.6, 0.43, 0, 1, 0.59, 0.00, 0.10])
# f4 = np.array([0, 1, 1, 0.69, 0, 1, 0.4, 0.00, 0.90])
# f5 = np.array([0, 1, 0.6, 0.43, 0, 1, 0.59, 0.00, 0.10])
# f6 = np.array([0, 1, 1, 0.69, 0, 1, 0.4, 0.00, 0.90])
# f7 = np.array([1, 1, 0.6, 0.00, 0, 0.3, 0.13, 0.00, 0.80])
# f8 = np.array([0, 1, 0.6, 0.43, 0, 1, 0.59, 0.00, 0.10])
# f9 = np.array([0, 1, 1, 0.69, 0, 1, 0.4, 0.00, 0.90])
# f10 = np.array([1, 1, 0.6, 0.00, 0, 0.3, 0.13, 0.00, 0.80])
# f11 = np.array([0, 1, 0.6, 0.43, 0, 1, 0.59, 0.00, 0.10])
# f12 = np.array([0, 1, 1, 0.69, 0, 1, 0.4, 0.00, 0.90])
# #xt = np.array(x)
# flist = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12]
# colors = ['blue','green','red', 'orange','purple','grey','deepskyblue','brown','darkslategray','black','rosybrown','mediumspringgreen']
# #source for colors : https://docs.bokeh.org/en/latest/docs/reference/colors.html?highlight=colors
# for i in range(len(flist)):
#     xt, yt = radar_patch(flist[i], theta, centre)
#     p.patch(x=xt, y=yt, fill_alpha=0.15, fill_color=colors[i])
# show(p)