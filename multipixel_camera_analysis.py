import numpy as np
import multiprocessing
import h5py
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import scipy
import functools
from BeadDataFile import *

def start_process():
    '''Function ran when starting new multithreading processes'''
    print('Starting', multiprocessing.current_process().name)
    
def lin_detrend(vals):
    '''Performs linear detrending on a 1D array/list of values'''
    slope, intercept, _, _, _ = scipy.stats.linregress(np.array(range(0, len(vals))), vals)

    # Calculate fitted values
    fitted_values = slope * np.array(range(0, len(vals))) + intercept

    # Subtract fitted values from original data
    residuals = vals - fitted_values
    return residuals

def findclosestoset(series, setseries):
    '''Given a series and a list of target values, returns a 1D array of indices corresponding to the eleemnts in the series closest to each target value. Usually used to find target frequencies in FFT results.'''
    return np.array([np.argmin(np.abs(series-setser)) for setser in setseries])

#The following are helper functions to pull specific data from h5 datasets. Note that these do not work if the file is already open.

def getsamplingrate(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        samplingrate = f['auxdata']['samplingrate'][()]
    return samplingrate

def getsubsetmap(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        subsetmap = f['auxdata']['subsetmap'][:]
    return subsetmap

def getnormscale(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        image0 = f['auxdata']['normalizationscale'][()]
    return image0

def getimage0(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        image0 = f['cameradata']['arrays'][0]
    return image0



def subsetframe(sourcedata, frame, normalize, normalfactor):
    '''Multithreaded function used in multisubset'''
    if normalize:
        return np.uint8(np.round(normalfactor*sourcedata[frame]/np.sum(sourcedata[frame])))
    else:
        return sourcedata[frame]
    
def multisubset(sourcefilename, targetfilename, frame, datalength = np.inf, normalize=True):
    '''Given a source h5 file, and an array of boolean values corresponding to the image shape,
    creates a new h5 file at a given file path only containing the pixels corresponding to true
    values in the boolean array. Can also be used to normalize each frame by sum.
    
    Note: loads entire file into RAM. Avoid datasets >48000 frames.
    
    Inputs:
        sourcefilename (string): h5 filepath to get camera data from
        targetfilename (string): h5 filepath to save the new dataset to
        frame (array[bool]): boolean array representing the pixels to subset
        datalength (int): maximum number of frames to keep (default infinity)
        normalize (bool): normalize each frame by sum if true (default true)'''
    
    im0 = getimage0(sourcefilename)
    
    sourcefile = h5py.File(sourcefilename, 'r')
    targetfile = h5py.File(targetfilename, 'w')
    datalength = min(len(sourcefile['cameradata']['arrays']), datalength)
    
    #This is a crude metric for scaling that sets the normalized value of the brightest pixel in the
    #first image to 240. This is used to ensure the normalized data can be saved as uint8 without
    #losing significant information. This might be part of what's causing the map volitility.
    
    #To do: find alternatives
    if normalize:
        scale = 240/np.max(im0/np.sum(im0))
    else:
        scale = 1
        
    #Used to pass the initial values into the multiprocessing function at each frame
    subsetframe_specific = functools.partial(subsetframe, frame=frame, normalize=normalize, normalfactor=scale)
    
    returnval=[]
    pool = multiprocessing.Pool(processes=6,
                initializer=start_process, maxtasksperchild=50)
    
    #Try-Except loop ensures everything closes properly if code fails
    try:
        #Initialize file
        targetfile.create_group("auxdata")
        targetfile.create_dataset("auxdata/subsetmap", data=frame)
        targetfile.create_dataset("auxdata/samplingrate", data=sourcefile['auxdata']['samplingrate'][()])
        targetfile.create_dataset("auxdata/normalizationscale", data=scale)
        im0_test = sourcefile['cameradata']['arrays'][0][frame]
        targetfile.create_group("cameradata")
        targetfile.create_dataset("cameradata/arrays", data=pool.starmap(subsetframe_specific, zip(sourcefile['cameradata']['arrays'][:datalength])))
    except:  
        pool.close()
        sourcefile.close()
        targetfile.close()
        raise
    pool.close()
    sourcefile.close()
    targetfile.close()

def expand_fromsubset(vector, subsetmap):
    '''Given a 1D array of values, and a boolean array corresponding to a subset map, returns a
    map of the same size as the original image, where true elements are populated with the values
    and false elements are set to 0'''
    outp = np.zeros_like(subsetmap,dtype=vector.dtype)
    outp[subsetmap] = vector
    return outp

def h5_fft(h5filepath, datalength=np.inf, sectionlength=8000):
    '''Given a filepath to an h5 file with camera data, takes the fft of each
    individual pixel. It does this by splitting into smaller sections to ease
    processing strain; default is 10 seconds by convention.
    
    Inputs:
        h5filepath (string): h5 filepath to get camera data from
        datalength (int): maximum number of frames to keep (default infinity)
        sectionlength (int): number of frames per section (default 8000)
        
    Output:
        tuple[array] with the fft frequencies and the fft data'''
    with h5py.File(h5filepath, 'r') as f:
        datalength=min(len(f['cameradata']['arrays']), datalength)
        nsections = len(f['cameradata']['arrays'])//sectionlength
        imageshape = f['cameradata']['arrays'][0].shape
        
        #index represents the xy coordinate of the pixel with greatest magnitude in the first image.
        #it is arbitrarily chosen to normalize the phase
        index = np.unravel_index(np.argmax(f['cameradata']['arrays'][0]), imageshape)
        index_slice = tuple([slice(None)]) + tuple(index)
        
        #calculate fft, then divide by phase of index pixel. 
        #Dividing by nsections allows us to get the average with a simple sum.
        fft_transf=np.fft.rfft(f['cameradata']['arrays'][:sectionlength], axis=0)/nsections
        bigpixel_phase = fft_transf[index_slice]/np.abs(fft_transf[index_slice])
        fft_transf /= bigpixel_phase[:,np.newaxis, np.newaxis]
        
        #splits the data into sections, takes the fft of each, and averages.
        for i in range(1,nsections):
            temp = np.fft.rfft(f['cameradata']['arrays'][i*sectionlength:(i+1)*sectionlength], axis=0)/nsections
            bigpixel_phase = temp[index_slice]/np.abs(temp[index_slice])
            fft_transf += temp/bigpixel_phase[:,np.newaxis,np.newaxis]
            
        #calculates and returns numpy fft frequency conventions for sampling rate and section length
        samplingrate_transf = np.round(f['auxdata']['samplingrate'][()])
        freqs_transf = np.fft.rfftfreq(sectionlength, 1/samplingrate_transf)
    return freqs_transf, fft_transf

def isolate_frequency(target_frequency, full_fft, sectionlength=8000):
    '''Uses h5_fft (or an existing fft data result) to return each pixels fft data
    at a given frequency.
    Inputs:
        target_frequency (int): frequency to return
        full_fft (tuple/array OR str): either the results from a previous full_fft run or a filepath to perform the fft on
        sectionlength (int): number of frames per section (default 8000)'''
    if type(full_fft) is str:
        full_fft = h5_fft(full_fft, sectionlength=8000)
    target_index = findclosestoset(full_fft[0], [target_frequency])[0]
    return full_fft[1][target_index]

def singlefreq_fourier(h5filepath, frequency, hz=True, datalength=np.inf):
    '''(DEPRECATED) Test to speed up calculating one frequency at a time instead of
    subsetting from the full fft. No significant speedups found. Likely areas for efficieny
    improvement if this becomes a part of future analysis.'''
    if hz: frequency = frequency*2*np.pi
    with h5py.File(h5filepath, 'r') as f:
        samplingrate = f['auxdata']['samplingrate'][()]
        deltatime = 1/samplingrate
        counter = np.zeros(f['cameradata']['arrays'][0].shape, dtype='complex128')
        datalength = min(datalength, len(f['cameradata']['arrays']))
        
        #Probably don't need to load entire file into memory
        for n, val in enumerate(f['cameradata']['arrays'][:datalength]):
            counter += val*np.exp(-1j*deltatime*frequency*n)
    return counter

def custom_cmap():
    '''Wrapper for a custom color map that matches the seismic color map, but has low and high values as the same color. Intended for plotting frequency maps.'''
    colors = [(0, 'blue'), (0.25, 'black'), (0.5, 'red'), (0.75, 'white'), (1, 'blue')]
    return LinearSegmentedColormap.from_list("custom_colormap", colors)

def make_angleplot(fig, ax, arr, title, mask=None):
    '''Helper function for plotting the fft phase for a camera frame at a given frequency.    
    Inputs:
        fig, ax: existing matplotlib fig, ax
        arr (array): phases to plot
        title (string): title of graph
        mask (array[float]): alpha mask to overlay over image'''
    if mask is not None:
        color_map_overlay = ax.imshow(arr, cmap=custom_cmap(), alpha=mask/np.max(mask))
    else:
        color_map_overlay = ax.imshow(arr, cmap=custom_cmap(), alpha=1)
    cbar = fig.colorbar(color_map_overlay, cmap=custom_cmap, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.ax.set_yticklabels([r'$-\pi$', r'$\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
    cbar.ax.set_ylabel("complex phase")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

def save_angleplot(arr, title, filename, mask=None):
    '''Makes a graph as in make_angleplot and saves it to a given filepath
    Inputs:
        arr (array): phases to plot
        title (string): title of graph
        filename (string): filepath to save image to
        mask (array[float]): alpha mask to overlay over image'''
    fig, ax = plt.subplots()
    make_angleplot(fig, ax, arr, title, mask)
    plt.savefig(filename)
    plt.clf()
    
def make_scatterplot(fig, ax, values, title, xvals=None, ylim=None, **plotargs):
    '''Helper function to contain some plot arguments when making a scatterplot.
    Would not recommend using outside of other functions in this file'''
    if xvals is not None:
        indices = findclosestoset(values[1],xvals)
    else:
        indices = slice(None)
    ax.scatter(values[1][indices], values[0][indices], **plotargs)

    ax.tick_params(axis='both',which='both', bottom=False, left=False)

    ax.set_xscale('log')
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlim(1, 800)
    ax.grid(which='major', linestyle='--', linewidth='0.5', color='black', alpha=0.5)
    ax.set_title(title)
    
def save_scatterplot(values, title, filename, xvals=None, ylim=None, **plotargs):
    '''Saves scatterplot as made above. Would not recommend using outside of other functions in this file.'''
    fig, ax = plt.subplots()
    make_scatterplot(fig, ax, values, title, xvals=xvals, ylim=ylim, **plotargs)
    plt.savefig(filename)
    plt.clf()
    
def phase_to_mask(phasegrid, std=10):
    '''Given an array of phases, uses a gaussian blur and binarizes to turn into a mask with
    opposite phases having values of +-1.'''
    blurred = np.zeros(phasegrid.shape)
    
    #This creates a gaussian blurred frame representing absolute phase difference from 0
    phasegrid = np.abs(phasegrid)
    blurred = scipy.ndimage.gaussian_filter(phasegrid, std)
    
    #Binarizes the image
    blurred[np.where(blurred<np.pi/2)]=-1
    blurred[np.where(blurred>np.pi/2)]=1
    return blurred

def psd(series, samplingrate, sqrt = True, detrend='linear'):
    '''Calculates the psd of a series. 
    See (https://grattalab.com/elog/optlev/2023/10/25/conventions-for-spectra-spectral-densities-fft-normalization-etc/) for conventions.'''
    if detrend not in ['linear', 'mean', 'none', None]:
        raise ValueError("Valid options for detrend; linear, mean, none")
    if detrend == 'linear': series = lin_detrend(series)
    if detrend == 'mean': series = series-np.mean(series)
    
    spectrum = np.fft.rfft(series)
    freqs = np.fft.rfftfreq(len(series), d=1/samplingrate)
    p = 2/(samplingrate*len(series))*np.abs(spectrum)**2
    
    if sqrt: return np.sqrt(p), freqs
    else: return p, freqs

def windowed_psd(series, samplingrate, sqrt = True, detrend='linear', winsize=8000):
    '''Calculates the psd of a series, split into sections with a tukey window applied. 
    See (https://grattalab.com/elog/optlev/2023/10/25/conventions-for-spectra-spectral-densities-fft-normalization-etc/) for conventions.'''
    if detrend not in ['linear', 'mean', 'none', None]:
        raise ValueError("Valid options for detrend; linear, mean, none")
    if detrend == 'linear': series = lin_detrend(series)
    if detrend == 'mean': series = series-np.mean(series)
    
    #Getting values from window to properly scale psd
    sections = [series[i:i+winsize] for i in range(0, len(series), winsize)]
    win = scipy.signal.windows.tukey(winsize, 0.05)
    S_1 = np.sum(win)
    S_2 = np.sum(win**2)
    
    freqs = np.fft.rfftfreq(winsize, d=1/samplingrate)
    
    spectrum = np.array(np.fft.rfft(sections[0]*win))
    avgpsd = 2/(samplingrate*S_2)*np.abs(spectrum)**2
    avgpsd /= len(sections)
    
    for sec in sections[1:]:
        spectrum = np.array(np.fft.rfft(sec*win))
        p = 2/(samplingrate*S_2)*np.abs(spectrum)**2
        avgpsd += p/len(sections)
    
    if sqrt: return np.sqrt(avgpsd), freqs
    else: return avgpsd, freqs

def windowed_fft(series, samplingrate, detrend='linear', winsize=8000):
    '''Calculates the fft of a series, split into sections with a tukey window applied. 
    See (https://grattalab.com/elog/optlev/2023/10/25/conventions-for-spectra-spectral-densities-fft-normalization-etc/) for conventions.'''
    if detrend not in ['linear', 'mean', 'none', None]:
        raise ValueError("Valid options for detrend; linear, mean, none")
    if detrend == 'linear': series = lin_detrend(series)
    if detrend == 'mean': series = series-np.mean(series)
    
    sections = [series[i:i+winsize] for i in range(0, len(series), winsize)]
    win = scipy.signal.windows.tukey(winsize, 0.05)
    S_1 = np.sum(win)
    S_2 = np.sum(win**2)
    spec = []
    
    freqs = np.fft.rfftfreq(winsize, d=1/samplingrate)
    
    spectrum = np.array(np.fft.rfft(sections[0]*win))
    avgpsd = np.sqrt(2)/(S_1)*np.abs(spectrum)
    avgpsd /= len(sections)
    
    for sec in sections[1:]:
        spectrum = np.array(np.fft.rfft(sec*win))
        p = np.sqrt(2)/(S_1)*np.abs(spectrum)
        avgpsd += p/len(sections)
    return avgpsd, freqs
    
#Note: the following code is structured the way it is because it uses class hierarchy to pass consistent values to starmap.
#This is not the most effective approach, and will be changed.

#Dims represents the dimensions of the array passed to the object; raw camera data is 2D, but subsetted camera data is 1D
#The "mask" can also be multiple masks stacked along the last axis; in that case, all functions here will return an array with
#the sums taken from each seperate mask.


def findSectionSumsMasked(frame, mask, dims):
    '''Parallel part of parallelSumsMasked'''
    return np.squeeze(np.apply_over_axes(np.sum, (np.expand_dims(frame, -1)/np.sum(frame))*mask, range(dims)))

def parallelSumsMasked_h5(mask, h5filepath, datalength=np.inf, dims=2):
    '''Given a filepath to an h5 file with camera data, returns the sum of each image
    weighted by the maps defined in the parallelsummer object.'''
    if len(mask.shape) < dims or len(mask.shape) > dims+1: raise ValueError("Dimension and mask mismatch")
    if len(mask.shape) == dims: mask = np.expand_dims(mask,-1)
    nums = []

    sectionsum_specific = functools.partial(findSectionSumsMasked, mask=mask, dims=dims)
    
    f = h5py.File(h5filepath, 'r')
    datalength=min(len(f['cameradata']['arrays']), datalength)
    returnval=[]

    # ### parallel processing ###
    pool = multiprocessing.Pool(processes=6, \
            initializer=start_process, maxtasksperchild=50)
    returnval = pool.starmap(sectionsum_specific, zip(f['cameradata']['arrays'][:datalength]))
    pool.close()
    f.close()

    return returnval
        
def generate_masks(xfile, yfile, xfrequency, yfrequency, blurred=True):
    '''Given an x file, a y file, and a frequency for each, converts the
    phase at each pixel at each frequency to a map for weighted sums
    
    Output:
        The x and y maps as a numpy array stacked along the last axis'''
    
    xmap = np.angle(isolate_frequency(xfrequency, xfile))
    ymap = np.angle(isolate_frequency(yfrequency, yfile))
    if blurred:
        xmap = phase_to_mask(xmap)
        ymap = phase_to_mask(ymap)
    return np.dstack((xmap, ymap))

def manual_leftinv(matrix):
    '''Calculates left inverse using matrix multiplication (not SVD)'''
    return np.matmul(np.linalg.inv(np.matmul(matrix.T, matrix)), matrix.T)

def generate_quad_masks(shape):
    '''Given a 2x2 array shape, return a mask that mimics the qpd'''
    xquadmap = np.ones(shape)
    xquadmap[:, :shape[1]//2] = -1
    yquadmap = np.ones(shape)
    yquadmap[:shape[0]//2, :] = -1
    return np.dstack((xquadmap,yquadmap))

def generate_diagonal_masks(xfile, yfile, xfrequency, yfrequency, real=True, xnormalized = True, ynormalized = True):
    '''Given an x file, a y file, and a frequency for each, generates maps from the diagonalization procedure
    shown here https://grattalab.com/elog/optlev/2024/04/25/applying-diagonalization-maps-to-11-27-camera-data/
    Inputs:
        xfile (string): filepath of h5 file with camera data for x data
        yfile (string): filepath of h5 file with camera data for y data
        xfrequency (int): frequency to diagonalize the x map at
        yfrequency (int): frequency to diagonalize the y map at
        real (bool): only keep the real part of the frequency response to generate the maps (default True)
        xnormalized, ynormalized (bool): set to true if the x and y files were normalized during subsetting (default true)'''
    shape = getimage0(xfile).shape
    if shape != getimage0(yfile).shape:
        raise ValueError("X File and Y file aren't the same shape")
        
    #Get x and y frequency responses at target frequencies
    x1 = isolate_frequency(xfrequency, xfile)
    if xnormalized: x1 = x1 / getnormscale(xfile)
    y1 = isolate_frequency(yfrequency, yfile)
    if ynormalized: y1 = y1 / getnormscale(yfile)
    
    #Turns the maps into a nx2 matrix, then calculate the left inverse
    maps = np.squeeze(np.dstack((x1.flatten(), y1.flatten())))
    if real:
        maps = np.real(maps)
    maps_inv = manual_leftinv(maps)
    
    #reshape the inverted matrix to recreate the correct x/y shapes and return
    return maps_inv.T.reshape(shape+tuple([2]))

def calculate_snr(psd_vals, signal_values, maxval=None):
    '''Given results from a PSD and frequencies at which to measure the signal,
    return the ratio between the average signal power and the average non-signal power.
    Note: maxval is the maximum frequency to consider for noise'''
    if maxval is None:
        maxval = -1
    else:
        maxval = findclosestoset(psd_vals[1], [maxval])[0]
    signal_values = findclosestoset(psd_vals[1], signal_values)
    mask = np.full(len(psd_vals[1]), True)
    mask[signal_values] = False
    mask[maxval:] = False
    return np.mean(psd_vals[0][signal_values])/np.mean(psd_vals[0][mask])
        
def makeTransferFuncPlot(masks, xfile, yfile, xbeadfile, ybeadfile, zfile=None, xvals=None, ylim=None, datalength=np.inf, filepath = None, electrons=9, plotbase=None, dims=2, **plotargs):
    '''Plots a transfer function from x, y, and optionally z camera files. Optionally save the file to a given filepath
    If frequencies are specified, only those frequencies will be shown, and the transfer functions will be converted to
    force units based on those files
    Inputs:
        masks (array): x and y mask to test, stacked along last axis
        xfile (string): filepath of h5 file with camera data for x transfer data
        yfile (string): filepath of h5 file with camera data for y transfer data
        zfile (string): filepath of h5 file with camera data for z transfer data (default None)
        xbeadfile (string): filepath of h5 file with bead data for x transfer data
        ybeadfile (string): filepath of h5 file with bead data for y transfer data
        
        ylim: y limits of plots (default None)
        datalength (int): maximum number of frames to keep from the camera datasets (default infinity)
        filepath (string): filepath to save save the image to (default None)
        dims (int): number of dimensions per image (default 2)
        
        xvals (array[int]): the frequencies to display and measure the force (default None, shows all frequencies)
        electrons (int): number of electrons for force normalization (default 9)
        
        plotbase (fig, array[ax]): pass existing transfer function plot fig/axs as input to plot multiple transfer functions on the same graph
        **plotargs: any additional elements to add to the scatterplot
    Outputs:
        (fig, axs): figure and axes for plot
        '''
    
    #Calculates force calibration if xvals specified
    calib = [1,1]
    if xvals is not None:
        calib = force_calibration(masks, xfile, yfile, xbeadfile, ybeadfile, electrons=electrons, xvals=xvals, datalength=datalength)
      
    par = parallelsummer(masks, dims=dims)
    if zfile is not None:
        files = [xfile, yfile, zfile]
    else:
        files = [xfile,yfile]
    
    titles = ["x", "y", "z"]
    if plotbase is not None:
        fig, axs = plotbase
    else:
        fig, axs = plt.subplots(2,len(files), figsize=(18,3*len(files)), sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0.175,wspace=0.1)
    
    for j in range(len(files)):
        #Gets x/y map weighted sums for each file 
        vals = par.parallelSumsMasked_h5(files[j], datalength)
        samplingrate = getsamplingrate(files[j])
        for i in range(2):
            #Splits x/y measurements, calculates/plots psd
            data = windowed_psd(np.abs(np.array(vals)[:,i])*calib[i], samplingrate, winsize=int(samplingrate*10))
            title = f"Response in {titles[i]} to drive in {titles[j]}"    
            make_scatterplot(fig, axs[i,j], data, title, xvals=xvals, ylim=ylim, **plotargs)
    
    #Ensures y axis is properly labeled with force units or arbitrary
    if calib == [1,1]:
        axs[0,0].set_ylabel(r'$\sqrt{S_x} [Arb/\sqrt{Hz}]$')
        axs[1,0].set_ylabel(r'$\sqrt{S_y} [Arb/\sqrt{Hz}]$')
    else:
        axs[0,0].set_ylabel(r'$\sqrt{S_x} [N/\sqrt{Hz}]$')
        axs[1,0].set_ylabel(r'$\sqrt{S_y} [N/\sqrt{Hz}]$')
    
    for j in range(len(files)):
        axs[-1,j].set_xlabel(r'$Freq [Hz]$')
        
    if filepath is not None:
        plt.savefig(filepath)
    return (fig, axs)

def force_calibration(masks, xfile, yfile, xbeadfile, ybeadfile, electrons=9, xvals=np.arange(1,100), datalength=np.inf, dims=2):
    '''Given x/y masks to test, files for camera data and bead data corresponding to x and y transfer functions,
    return the coefficient needed to convert to force units.
    Inputs:
        masks (array): x and y mask to test, stacked along last axis
        xfile (string): filepath of h5 file with camera data for x transfer data
        yfile (string): filepath of h5 file with camera data for y transfer data
        xbeadfile (string): filepath of h5 file with bead data for x transfer data
        ybeadfile (string): filepath of h5 file with bead data for y transfer data
        electrons (int): number of electrons on bead (default 9)
        xvals (array[int]): the frequencies of the comb at which to measure the force
        datalength (int): maximum number of frames to keep from the camera dataset (default infinity)
        dims (int): number of dimensions per image (default 2)'''
    files = [xfile, yfile]
    calib = [0,0]
    
    #Getting force from bead datafiles
    force = get_driveforce(xbeadfile, ybeadfile, electrons=electrons, xvals = xvals)
    
    #Calculates force from camera data as mean fft over all tested frequencies
    par = parallelsummer(masks, dims=dims)
    for i in range(2):
        vals = par.parallelSumsMasked_h5(files[i], datalength)
        samplingrate = getsamplingrate(files[i])
        data = windowed_fft(np.abs(np.array(vals)[:,i]), samplingrate, winsize=int(samplingrate*10)) 
        indices = findclosestoset(data[1], xvals)
        calib[i] = force[i]/np.mean(data[0][indices])
    return calib

def get_driveforce(xfile, yfile, electrons=9, xvals = np.arange(1,100)):
    '''Given bead data files for X and Y drive, a list of comb frequencies,
    and the number of electrons, calculates the mean force applied over all
    target frequencies.
    '''
    
    xt_electrodes = BeadDataFile(xfile)
    yt_electrodes = BeadDataFile(yfile)
    xdrive_efield = (xt_electrodes.electrode_data[0]-xt_electrodes.electrode_data[1])*100*0.66/8.6e-3
    ydrive_efield = (yt_electrodes.electrode_data[0]-yt_electrodes.electrode_data[1])*100*0.66/8.6e-3

    xdrive_force = xdrive_efield*scipy.constants.e*electrons
    ydrive_force = ydrive_efield*scipy.constants.e*electrons

    #Using 10 seconds at QPD sampling rate of 5 kHz
    xforce_amp = windowed_fft(xdrive_force,5000,winsize=50000)
    yforce_amp = windowed_fft(ydrive_force,5000,winsize=50000)
    
    indices = findclosestoset(xforce_amp[1], xvals)
    
    xforce_avg = np.mean(xforce_amp[0][indices])
    yforce_avg = np.mean(yforce_amp[0][indices])
    return (xforce_avg, yforce_avg)
