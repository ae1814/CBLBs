import matplotlib.pyplot as plt 
from solver import * 
import numpy as np  
import os.path      
import pickle     
import seaborn as sns
import pandas as pd
import multiprocessing
from random import randint

from model_clb_2_4_Decoder import model_clb_2_4_Decoder, param_references
 


def getParamDistrib(number_points = 0, file_name = ""):    
    rand_samples = []    
    
    region = model_regions[0]   
    if number_points:
        samples = region.points[np.random.choice(region.points.shape[0], number_points, replace=False), :]
    else:
        samples = region.points

    rand_samples.append(samples)

    rand_samples = np.array(rand_samples)

    param_names = [r"$\delta_L$", 
                    r"$\gamma_L$", 
                    r"$n_y$", 
                    r"$\theta_L$", 
                    r"$\eta_x$", 
                    r"$\omega_x$", 
                    r"$m_x$", 
                    r"$\delta_x$",
                    #r"$\delta_y$", #to remove
                    r"$\gamma_x$",
                    r"$\theta_x$",
                    r"$\rho_x$"]
    
    df = pd.DataFrame(rand_samples[0])
    df.columns = param_names

    if file_name:
        df.to_csv(file_name, index=False)
    return df

def plotParamsdf(df=None, number_points = 0, box=False):
    if not type(df):
        df = getParamDistrib(number_points)
    
    param_names = [ r"$\gamma_L$", 
                    r"$\eta_x$", 
                    r"$\gamma_x$",
                    r"$\theta_L$", 
                    r"$\omega_x$", 
                    r"$\theta_x$",
                    r"$\delta_L$",  
                    r"$\delta_x$",
                    #r"$\delta_y$", #to remove                  
                    r"$\rho_x$",
                    r"$n_y$",                   
                    r"$m_x$"]
    units = [r"$nM/min$", r"$nM/min$", r"$nM/min$", r"$nM^{-1}$", r"$nM^{-1}$", r"$nM^{-1}$", r"$min^{-1}$", r"$min^{-1}$", r"$min^{-1}$", "", ""]

    fig, axes = plt.subplots(4,3)

    for i, (param_name,unit) in enumerate(zip(param_names, units)):
        if param_name:
            ax = axes.flat[i]
            if box:
                sns.boxplot(data=df[param_name], ax = ax) #,palette="Pastel1")
            else:    
                sns.violinplot(data=df[param_name], ax = ax, cut=0, color="#3274a1") #,palette="Pastel1")
            ax.set_xticks([])
            #ax.set_xticks([0])
            #ax.set_xticklabels([param_name])    
            if unit:    
                ax.set_ylabel(param_name + " [" + unit + "]")
            else:
                ax.set_ylabel(param_name)

            #ax.set_yscale('log')

    """
    for param_id in range(len(param_names)):
        ax = axes.flat[param_id]

        sns.violinplot(y = param_names[param_id], x="Model id", data=df[[param_names[param_id], "Model id"]], ax = ax) #,palette="Pastel1")
    """ 
    fig=plt.gcf()

    fig.set_size_inches([15,12])
    if box:
        plt.savefig('results_robustness\\params_distrib_sns_box.pdf', bbox_inches = 'tight')
    else:
        plt.savefig('results_robustness\\params_distrib_sns.pdf', bbox_inches = 'tight')
    plt.show()

def test_random_point():
    points = model_regions[0].points
    candidate = tuple(points[randint(0,len(points)-1)])
    model.simulate(candidate, plot_on=True)

if __name__ == "__main__":

    sns.set_style("white")
    #flatui = ['#d9d9d9','#bdbdbd','#969696','#636363']
    #sns.set_palette(flatui)

    #
    # SETTINGS
    #
    read_data = True
    ga_solutions = False
    local_solutions = True

    
    
    base_paths_opt = ["results_optimization\\cblb", "results_optimization\\0cblb"]
    

    #
    # END OF SETTINGS
    #

    if read_data:


        #folders = [os.path.join(base_path, "one_bit_model"), os.path.join(base_path, "two_bit_model"), os.path.join(base_path, "three_bit_model")]   
        model = model_clb_2_4_Decoder()
        solver = Solver(model)

        model_regions = []

        
        region_files =  []
        for base_path_opt in base_paths_opt:
            if ga_solutions:
                region_files.append(base_path_opt + "_ViableSet_IterGA.p")
            if local_solutions:
                for i in range(10):
                    region_files.append(base_path_opt+"_Region0ViableSet_Iter" + str(i+1) + ".p")

        
        viablePoints = []   
        for region_file in region_files: 
            try:
                viablePointsRegion = pickle.load(open(region_file, "rb"))   
                print(len(viablePointsRegion))   
                viablePoints.extend(viablePointsRegion)
            except:
                print("Load of " + region_file + " failed!")
        print("Number of points: ",len(viablePoints))
        region = Region(viablePoints, model, "region")              
        model_regions.append(region)                                                                        
        
    


 
    #df = getParamDistrib(file_name="results_robustness\\params.csv")
    df = pd.read_csv("results_robustness\\params.csv")
    plotParamsdf(df)

    #test_random_point()
    