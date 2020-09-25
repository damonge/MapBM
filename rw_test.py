import os
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from mapbm.mapfmt import MapFmtHEALPixFull, MapFmtHEALPixPartial, MapFmtNpz, \
    MapFmtHDF5, MapFmtFITSIOImg, MapFmtFITSIOTab
from matplotlib import cm
import warnings
warnings.filterwarnings("ignore")


if not os.path.isfile('rw_times.npz'):
    msk = hp.read_map("data/mask_apodized.fits.gz", verbose=False)

    times = {}
    for c in [MapFmtHDF5,
              MapFmtFITSIOImg,
              MapFmtFITSIOTab,
              MapFmtNpz,
              MapFmtHEALPixPartial,
              MapFmtHEALPixFull]:
        times[c.name] = c.time_IO(msk)

    np.savez('rw_times.npz', **times)
times = np.load('rw_times.npz')
formats = list(times.keys())
timing = np.zeros([2, len(formats)])
for i, f in enumerate(formats):
    timing[0, i] = times[f][0]
    timing[1, i] = times[f][1]

tmax = np.amax(timing, axis=-1)
tmin = np.amin(timing, axis=-1)*0
tfracs = (timing - tmin[:, None])/(tmax-tmin)[:, None]

plt.imshow(tfracs.T, cmap=cm.bwr, aspect='auto', alpha=0.7, origin='lower')
plt.gca().set_yticks([0, 1, 2, 3, 4, 5])
plt.gca().set_yticklabels(formats, fontsize=14)
plt.gca().set_xticks([0, 1])
plt.gca().set_xticklabels(['Write', 'Read'], fontsize=14)
for i, t in enumerate(timing.T):
    plt.text(-0.1, i, '%.2lf s' % t[0])
    plt.text(0.9, i, '%.2lf s' % t[1])
plt.savefig('rw_times.png', bbox_inches='tight')
plt.show()
