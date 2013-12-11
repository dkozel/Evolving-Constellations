import numpy as np
import matplotlib
# matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import animation
import sys

u = 20

if len(sys.argv) == 1:
    print "You can also give filename as a command line argument"
    filename = raw_input("Enter Filename: ")
    fps = 5
else:
    filename = sys.argv[1]
    fps = sys.argv[2]

indivs = np.genfromtxt(filename, dtype=float, comments='#', delimiter=', ')
print indivs
arity_x2 = len(indivs[0])

imag_values = indivs[:,range(1,arity_x2,2)]
real_values = indivs[:,range(0,arity_x2,2)]
print imag_values

# Normalize the points such that the average norm is 1. This is done in GNURadio as well, so it is a realistic depiction. 

for i in range(0,len(imag_values)):
	normfactor = 0
	for x in range(0,arity_x2/2):
		normfactor += np.linalg.norm([imag_values[i,x],real_values[i,x]])
	normfactor = normfactor/(arity_x2/2)
	imag_values[i] = imag_values[i]/normfactor
	real_values[i] = real_values[i]/normfactor

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
ax = plt.axes(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
points, = ax.plot([], [], 'bo')
bestpoints, = ax.plot([], [], 'bo')

#ax.figure()

i2 = len(indivs[:,0])/u
xb = real_values[u*i2,:]
yb = imag_values[u*i2,:]

ax.plot(xb,yb, 'bo')


# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html


plt.show()
