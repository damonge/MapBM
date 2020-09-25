import healpy as hp
import numpy as np
import h5py
from fitsio import FITS
import os


class MapFmtBase(object):
    name = 'Base'
    ext = 'dum'

    def __init__(self, mask):
        self.npix = len(mask)
        self.nside = hp.npix2nside(self.npix)
        self.mask = mask
        self.goodpix = np.where(mask > 0)[0]
        self.badpix = np.where(mask <= 0)[0]
        self.npix_good = len(self.goodpix)

    def _make_maps(self, nmaps, full_sky=False):
        maps = np.random.randn(nmaps, self.npix_good)

        if full_sky:
            maps_out = np.zeros([nmaps, self.npix])
            maps_out[:, self.goodpix] = maps
            maps_out[:, self.badpix] = hp.UNSEEN
        else:
            maps_out = maps
        return maps_out

    def read_maps(self, fname):
        raise NotImplementedError("Don't use base class")

    def write_maps(self, maps, fname):
        raise NotImplementedError("Don't use base class")

    @classmethod
    def time_IO(cls, mask, n_maps=48, n_iter=10, verbose=True, cleanup=True):
        import time
        mf = cls(mask)
        m = mf.make_maps(n_maps)
        fname = "testmaps_" + mf.name + '.' + mf.ext

        if verbose:
            print('Timing ' + mf.name + ':')
        start = time.time()
        for i in range(n_iter):
            mf.write_maps(m, fname)
        stop = time.time()
        t_write = (stop - start) / n_iter
        if verbose:
            print("Write time: %lf s" % t_write)

        start = time.time()
        for i in range(n_iter):
            d = mf.read_maps(fname)
            np.sum(d)
        stop = time.time()
        t_read = (stop - start) / n_iter

        if verbose:
            print("Read time: %lf s" % t_read)
            print(" ")

        if cleanup:
            os.system("rm -rf " + fname)
        return t_write, t_read


class MapFmtHEALPixFull(MapFmtBase):
    name = 'Hpx-full'
    ext = 'fits'

    def make_maps(self, nmaps):
        return self._make_maps(nmaps, full_sky=True)

    def write_maps(self, maps, fname):
        hp.write_map(fname, maps, overwrite=True, dtype=np.float64)

    def read_maps(self, fname):
        return hp.read_map(fname, field=None, verbose=False)


class MapFmtHEALPixPartial(MapFmtBase):
    name = 'Hpx-partial'
    ext = 'fits'

    def make_maps(self, nmaps):
        return self._make_maps(nmaps, full_sky=True)

    def write_maps(self, maps, fname):
        hp.write_map(fname, maps, overwrite=True,
                     partial=True, dtype=np.float64)

    def read_maps(self, fname):
        return hp.read_map(fname, field=None,
                           verbose=False, partial=True)


class MapFmtNpz(MapFmtBase):
    name = 'Npz'
    ext = 'npz'

    def make_maps(self, nmaps):
        return self._make_maps(nmaps, full_sky=False)

    def write_maps(self, maps, fname):
        np.savez(fname, pixels=self.goodpix, maps=maps)

    def read_maps(self, fname):
        d = np.load(fname)
        p = d['pixels']  # noqa
        m = d['maps']
        return m


class MapFmtHDF5(MapFmtBase):
    name = 'HDF5'
    ext = 'hdf5'

    def make_maps(self, nmaps):
        return self._make_maps(nmaps, full_sky=False)

    def write_maps(self, maps, fname):
        with h5py.File(fname, 'w') as f:
            f.create_dataset('pixels', data=self.goodpix)
            f.create_dataset('maps', data=maps)

    def read_maps(self, fname):
        f = h5py.File(fname, 'r')
        p = f.get('pixels').value  # noqa
        m = f.get('maps').value
        f.close()
        return m


class MapFmtFITSIOImg(MapFmtBase):
    name = 'fitsio-Img'
    ext = 'fits'

    def make_maps(self, nmaps):
        return self._make_maps(nmaps, full_sky=False)

    def write_maps(self, maps, fname):
        f = FITS(fname, 'rw', clobber=True)
        f.write(self.goodpix)
        f.write(maps)
        f.close()

    def read_maps(self, fname):
        f = FITS(fname, 'r')
        p = f[0].read()  # noqa
        m = f[1].read()
        f.close()
        return m


class MapFmtFITSIOTab(MapFmtBase):
    name = 'fitsio-Tab'
    ext = 'fits'

    def make_maps(self, nmaps):
        return self._make_maps(nmaps, full_sky=False)

    def write_maps(self, maps, fname):
        f = FITS(fname, 'rw', clobber=True)
        f.write([self.goodpix], names=['pixels'])
        names = ['map_%d' % i for i in range(len(maps))]
        f.write(list(maps), names=names)
        f.close()

    def read_maps(self, fname):
        f = FITS(fname, 'r')
        p = f[1].read()  # noqa
        m = f[2].read()
        m = np.array([m[n] for n in f[2].get_colnames()])
        f.close()
        return m
