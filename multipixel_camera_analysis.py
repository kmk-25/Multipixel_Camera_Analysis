import numpy as np
import multiprocessing
import h5py
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import scipy

def start_process():
    '''
    for multiprocessing
    '''
    print('Starting', multiprocessing.current_process().name)
    
def lin_detrend(vals):
    slope, intercept, _, _, _ = scipy.stats.linregress(np.array(range(0, len(vals))), vals)

    # Calculate fitted values
    fitted_values = slope * np.array(range(0, len(vals))) + intercept

    # Subtract fitted values from original data
    residuals = vals - fitted_values
    return residuals

def custom_cmap():
    '''Wrapper for a custom color map that matches the seismic color map, but has low and high values as the same color. Intended for plotting frequency maps.'''
    colors = [(0, 'blue'), (0.25, 'black'), (0.5, 'red'), (0.75, 'white'), (1, 'blue')]  # Color points at 0, 0.5, and 1
    return LinearSegmentedColormap.from_list("custom_colormap", colors)

def findclosestoset(series, setseries):
    return np.array([np.argmin(np.abs(series-setser)) for setser in setseries])

def getsamplingrate(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        samplingrate = f['auxdata']['samplingrate'][()]
    return samplingrate

def getimage0(h5filepath):
    with h5py.File(h5filepath, 'r') as f:
        image0 = f['cameradata']['arrays'][0]
    return image0

def h5_fft(h5filepath, datalength=-1, sectionlength=8000):
    with h5py.File(h5filepath, 'r') as f:
        if datalength==-1: datalength=len(f['cameradata']['arrays'])
        nsections = len(f['cameradata']['arrays'])//sectionlength
        imageshape = f['cameradata']['arrays'].shape
        
        #index represents the xy coordinate of the pixel with greatest magnitude in the first image.
        #it is arbitrarily chosen to normalize the phase
        index = np.unravel_index(np.argmax(f['cameradata']['arrays'][0]), f['cameradata']['arrays'][0].shape)
        
        #calculate fft, then divide by phase of index pixel. 
        #Dividing by nsections allows us to get the average with a simple sum.
        fft_transf=np.fft.rfft(f['cameradata']['arrays'][:sectionlength], axis=0)/nsections
        bigpixel_phase = fft_transf[:,index[0],index[1]]/np.abs(fft_transf[:,index[0],index[1]])
        fft_transf = fft_transf/np.tile(np.reshape(bigpixel_phase, (len(fft_transf),1,1)), (1,imageshape[1],imageshape[2]))
        
        #splits the data into sections, takes the fft of each, and averages.
        for i in range(1,nsections):
            temp = np.fft.rfft(f['cameradata']['arrays'][i*sectionlength:(i+1)*sectionlength], axis=0)/nsections
            bigpixel_phase = temp[:,index[0],index[1]]/np.abs(temp[:,index[0],index[1]])
            temp = temp/np.tile(np.reshape(bigpixel_phase, (len(fft_transf),1,1)), (1,imageshape[1],imageshape[2]))
            fft_transf += temp
            
        #calculates and returns numpy fft frequency conventions for sampling rate and section length
        samplingrate_transf = np.round(f['auxdata']['samplingrate'][()])
        freqs_transf = np.fft.rfftfreq(sectionlength, 1/samplingrate_transf)
    return freqs_transf, fft_transf

def isolate_frequency(target_frequency, full_fft, sectionlength=8000):
    if type(full_fft) is str:
        full_fft = h5_fft(full_fft, sectionlength=8000)
    target_index = findclosestoset(full_fft[0], [target_frequency])[0]
    return full_fft[1][target_index]

def singlefreq_fft(h5filepath, frequency, hz=True):
    if hz: frequency = frequency*2*np.pi
    with h5py.File(h5filepath, 'r') as f:
        samplingrate = f['auxdata']['samplingrate'][()]
        deltatime = 1/samplingrate
        counter = np.zeros((f['cameradata']['arrays'].shape[1],f['cameradata']['arrays'].shape[2]), dtype='complex128')
        for n, val in enumerate(f['cameradata']['arrays'][:-1]):
            counter += val*np.exp(-1j*deltatime*frequency)
    return counter

def make_angleplot(fig, ax, arr, title, mask=None):
    if mask is not None:
        color_map_overlay = ax.imshow(np.angle(arr), cmap=custom_cmap(), alpha=mask/np.max(mask))
    else:
        color_map_overlay = ax.imshow(np.angle(arr), cmap=custom_cmap(), alpha=1)
    cbar = fig.colorbar(color_map_overlay, cmap=custom_cmap, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.ax.set_yticklabels([r'$-\pi$', r'$\frac{\pi}{2}$', '0', r'$\frac{\pi}{2}$', r'$\pi$'])
    cbar.ax.set_ylabel("complex phase")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

def save_angleplot(arr, title, filename, mask=None):
    fig, ax = plt.subplots()
    make_angleplot(fig, ax, arr, title, mask)
    plt.savefig(filename)
    plt.clf()
    
def make_scatterplot(fig, ax, values, title, xvals=None, ylim=None):
    if xvals is not None:
        indices = findclosestoset(values[1],xvals)
    else:
        indices = slice(None)
    ax.scatter(values[1][indices], values[0][indices], color='blue', label='', alpha=1, s=10)

    ax.tick_params(axis='both',which='both', bottom=False, left=False)

    ax.set_xscale('log')
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlim(1, 800)
    ax.grid(which='major', linestyle='--', linewidth='0.5', color='black', alpha=0.5)
    ax.set_title(title)
    
def save_scatterplot(values, title, filename, xvals=None, ylim=None):
    fig, ax = plt.subplots()
    make_scatterplot(fig, ax, values, title, xvals=xvals, ylim=ylim)
    plt.savefig(filename)
    plt.clf()
    
def phase_to_mask(phasegrid, std=10):
    blurred = np.zeros(phasegrid.shape)
    phasegrid = np.abs(phasegrid)
    blurred = scipy.ndimage.gaussian_filter(phasegrid, std)
    blurred[np.where(blurred<np.pi/2)]=-1
    blurred[np.where(blurred>np.pi/2)]=1
    return blurred

def psd(series, samplingrate, sqrt = True, detrend='linear'):
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
    for sec in sections:
        spectrum = np.array(np.fft.rfft(sec*win))
        p = 2/(samplingrate*S_2)*np.abs(spectrum)**2
        spec.append(p)
    p = np.average(np.array(spec), axis=0)
    if sqrt: return np.sqrt(p), freqs
    else: return p, freqs
    
class parallelsummer:
    def __init__(self, mask=None):
        self.mask = mask
        if len(self.mask.shape) == 2: self.mask = self.mask[:,:,np.newaxis]
        
    def findSectionSumsMasked(self, frame):
        '''See parallelSums'''
        return np.sum(frame[:,:,np.newaxis]*self.mask, axis=(0,1))

    def parallelSumsMasked_h5(self,h5filepath,datalength=-1):
        nums = []

        f = h5py.File(h5filepath, 'r')
        if datalength==-1: datalength=len(f['cameradata']['arrays'])
        returnval=[]

        # ### parallel processing ###
        pool = multiprocessing.Pool(processes=6, \
                initializer=start_process, maxtasksperchild=50)
        returnval = pool.starmap(self.findSectionSumsMasked, zip(f['cameradata']['arrays'][:datalength]))
        f.close()

        return returnval
    
    def parallelSumsMasked_array(self,arr,datalength=-1):
        nums = []

        if datalength==-1: datalength=len(arr)
        returnval=[]

        # ### parallel processing ###
        pool = multiprocessing.Pool(processes=6, \
                initializer=start_process, maxtasksperchild=50)
        returnval = pool.starmap(self.findSectionSumsMasked, zip(arr))

        return returnval
    
    def set_mask(self, mask):
        self.mask = mask   
        
def generate_masks(xfile, yfile, frequency, blurred=True):
    xmap = np.angle(isolate_frequency(frequency, xfile))
    ymap = np.angle(isolate_frequency(frequency, yfile))
    if blurred:
        xmap = phase_to_mask(xmap)
        ymap = phase_to_mask(ymap)
    return np.dstack((xmap, ymap))

def manual_leftinv(matrix):
    return np.matmul(np.linalg.inv(np.matmul(matrix.T, matrix)), matrix.T)

def generate_diagonal_masks(xfile, yfile, frequency, real=True):
    shape = getimage0(xfile).shape
    if shape != getimage0(yfile).shape:
        raise ValueError("X File and Y file aren't the same shape")
    x1 = isolate_frequency(xfrequency, xfile)
    y1 = isolate_frequency(yfrequency, yfile)
    maps = np.squeeze(np.dstack((x1.flatten(), y1.flatten())))
    if real:
        maps = np.real(maps)
    maps_inv = manual_leftinv(maps)
    return maps_inv.T.reshape((shape[0],shape[1],2))

def calculate_snr(psd_vals, signal_values, maxval=None):
    if maxval is None:
        maxval = -1
    else:
        maxval = findclosestoset(psd_vals[1], [maxval])[0]
    signal_values = findclosestoset(psd_vals[1], signal_values)
    mask = np.full(len(psd_vals[1]), True)
    mask[signal_values] = False
    mask[maxval:] = False
    return np.mean(psd_vals[0][signal_values])/np.mean(psd_vals[0][mask])
        
def makeTransferFuncPlot(masks, xfile, yfile, xbeadfile, ybeadfile, zfile=None, xvals=None, ylim=None, datalength=-1, plotname = None):
    force = get_driveforce(xbeadfile, ybeadfile, electrons=9, xvals = np.arange(1,100))
    print(force)
    par = parallelsummer(masks)
    if zfile is not None:
        files = [xfile, yfile, zfile]
    else:
        files = [xfile,yfile]
    
    titles = ["x", "y", "z"]
    fig, axs = plt.subplots(2,len(files), figsize=(18,3*len(files)), sharex=True,sharey=True)
    fig.subplots_adjust(hspace=0.175,wspace=0.1)
    
    m = [0,0,0,0]
    calib = [0,0]
    
    for j in range(len(files)):
        vals = par.parallelSumsMasked_h5(files[j], datalength)
        samplingrate = getsamplingrate(files[j])
        for i in range(2):
            data = windowed_psd(np.abs(np.array(vals)[:,i]), samplingrate, winsize=int(samplingrate*10))
            if i == j:   
                indices = findclosestoset(data[1], np.arange(1,100))
                calib[j] = force[j]/np.mean(data[0][indices])
            m[2*j+i] = data
    for j in range(len(files)):
        for i in range(2):
            data = (m[2*j+i][0]*calib[i], m[2*j+i][1])
            title = f"Response in {titles[i]} to drive in {titles[j]}"    
            print(xvals)
            make_scatterplot(fig, axs[i,j], data, title, xvals=xvals, ylim=ylim)
            if xvals is not None:
                print(f"SNR for response in {titles[i]} to drive in {titles[j]} = {calculate_snr(data, xvals, maxval=300)}")
            if i != j:
                print(f"Cross coupling in {titles[i]} from drive in {titles[j]} = {np.mean(data[0][indices])/force[j]:.00%}")
    
    axs[0,0].set_ylabel(r'$\sqrt{S_x} [N/\sqrt{Hz}]$')
    axs[1,0].set_ylabel(r'$\sqrt{S_y} [N/\sqrt{Hz}]$')
    for j in range(len(files)):
        axs[-1,j].set_xlabel(r'$Freq [Hz]$')
        
    if plotname is not None:
        plt.savefig(plotname)

import sys
sys.path.append("../../Tools") # go to parent dir
from BeadDataFile import *

def get_driveforce(xfile, yfile, electrons=9, xvals = np.arange(1,100)):
    xt_electrodes = BeadDataFile('/data/new_trap/20231109/Bead0/TransFunc/trapFocus/3/TF_X_0.h5')
    yt_electrodes = BeadDataFile('/data/new_trap/20231109/Bead0/TransFunc/trapFocus/3/TF_Y_0.h5')
    xdrive_efield = (xt_electrodes.electrode_data[0]-xt_electrodes.electrode_data[1])*100*0.66/8.6e-3
    ydrive_efield = (yt_electrodes.electrode_data[0]-yt_electrodes.electrode_data[1])*100*0.66/8.6e-3

    xdrive_force = xdrive_efield*scipy.constants.e*electrons
    ydrive_force = ydrive_efield*scipy.constants.e*electrons

    xforce_psd = windowed_psd(xdrive_force,5000,winsize=50000)
    yforce_psd = windowed_psd(ydrive_force,5000,winsize=50000)
    
    indices = findclosestoset(xforce_psd[1], xvals)
    
    xforce_avgpsd = np.mean(xforce_psd[0][indices])
    yforce_avgpsd = np.mean(yforce_psd[0][indices])
    return (xforce_avgpsd, yforce_avgpsd)
