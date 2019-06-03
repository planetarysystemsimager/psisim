import matplotlib.pyplot as plt
import numpy as np

def make_plots():
    '''
    A dummy function. 
    '''
    pass

    
def plot_detected_planet_contrasts(planet_table,wv_index,detected,flux_ratios,instrument,telescope,
    show=True,save=False,ymin=1e-9,ymax=1e-4,xmin=0.,xmax=1.,alt_data=None,alt_label=""):
    '''
    Make a plot of the planets detected at a given wavelenth_index

    Inputs: 
        planet_table    - a Universe.planets table
        wv_index        - the index from the instrument.current_wvs 
                            wavelength array to consider
        detected        - a boolean array of shape [n_planets,n_wvs] 
                        that indicates whether or not a planet was detected 
                        at a given wavelength
        flux_ratios     - an array of flux ratios between the planet and the star
                        at the given wavelength. sape [n_planets,n_wvs]
        instuemnt       - an instance of the psisim.instrument class
        telescope       - an instance of the psisim.telescope class

    Keyword Arguments: 
        show            - do you want to show the plot? Boolean
        save            - do you want to save the plot? Boolean
        ymin,ymax,xmin,xmax - the limits on the plot
        alt_data        - An optional argument to pass to show a secondary set of data.
                        This could be e.g. detection limits, or another set of atmospheric models
        alt_label       - This sets the legend label for the alt_data
    '''


    fig,ax = plt.subplots(1,1,figsize=(7,5))



    seps = np.array([planet_table_entry['AngSep']/1000 for planet_table_entry in planet_table])
    
    # import pdb; pdb.set_trace()
    #Plot the non-detections
    ax.scatter(seps[~detected[:,wv_index]],flux_ratios[:,wv_index][~detected[:,wv_index]],
        marker='.',label="Full Sample",s=20)

    print(seps[~detected[:,wv_index]],flux_ratios[:,wv_index][~detected[:,wv_index]])

    #Plot the detections
    ax.scatter(seps[detected[:,wv_index]],flux_ratios[:,wv_index][detected[:,wv_index]],marker='o',label="Detected",color='r')

    #Plot 1 and 2 lambda/d
    ax.plot([instrument.current_wvs[wv_index]*1e-6/telescope.diameter*206265,instrument.current_wvs[wv_index]*1e-6/telescope.diameter*206265],
        [0,1.],label=r"$\lambda/D$ at $\lambda=${}$\mu m$".format(instrument.current_wvs[wv_index]),color='k')
    ax.plot([2*instrument.current_wvs[wv_index]*1e-6/telescope.diameter*206265,2*instrument.current_wvs[wv_index]*1e-6/telescope.diameter*206265],
        [0,1.],'-.',label=r"$2\lambda/D$ at $\lambda=${}$\mu m$".format(instrument.current_wvs[wv_index]),color='k')


    #If detection_limits is passed, then plot the 5-sigma detection limits for each source
    if alt_data is not None:
        ax.scatter(seps[~detected[:,wv_index]],alt_data[:,wv_index][~detected[:,wv_index]],marker='.',
            label=alt_label,color='darkviolet',s=20)
        for i,sep in enumerate(seps[~detected[:,wv_index]]):
            ax.plot([sep,sep],[flux_ratios[i,wv_index][~detected[i,wv_index]],alt_data[i,wv_index][~detected[i,wv_index]]],
                color='k',alpha=0.1,linewidth=1)

    #Axis title
    ax.set_title("Planet Detection Yield at {}um".format(instrument.current_wvs[wv_index]))

    #Legend
    ax.legend(loc='upper right')

    #Plot setup
    ax.set_ylabel("Total Intensity Flux Ratio")
    ax.set_xlabel("Separation ['']")
    # ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_yscale('log')
    ax.set_xscale('log')

    #Do we show it?
    if show:
        plt.show()
    
    plt.tight_layout()

    #Do we save it? 
    if save:
        plt.savefig("Detected_Planets_flux_v_sma.png",bbox_inches="tight")

    #Return the figure so that the user can manipulate it more if they so please
    return fig,ax