#Note: to be fixed
from multipixel_camera_analysis import *

def main():
    mask_279 = isolate_frequency(279, '/data/beadVideos/bead1127/transfer_funcX/data.h5')
    
    save_angleplot(mask_279, "mask_279_test", 'main_test_angle.png')
    par = parallelsummer(mask_279)
    sums = par.parallelSumsMasked('/data/beadVideos/bead1127/transfer_funcX/data.h5')
    save_scatterplot(psd(np.abs(sums), getsamplingrate('/data/beadVideos/bead1127/transfer_funcX/data.h5')), "mask_279_test", 'main_test_scatter.png')    
    
    
if __name__ == "__main__":
    # Execute the main function if this script is run as the main module
    main()