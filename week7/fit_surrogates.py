import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from idaes.core.surrogate.pysmo_surrogate import PysmoPolyTrainer, PysmoRBFTrainer, PysmoSurrogate
from idaes.core.surrogate.metrics import compute_fit_metrics
from idaes.core.surrogate.pysmo.sampling import HammersleySampling, LatinHypercubeSampling
import os


class surrogateFitting:
    def __init__(self,
                 data=None,
                 input_labels=[],
                 output_labels=[],
                 initial_sample_type='Hammersley',
                 n_init=None,
                 n_mid=None, # misclassified - should be 0 if the output is general corrosion
                 n_final=None, # largest error
                 n_add=1,
                 basis=None
                 ):
        self.data = data
        for output in output_labels:
            self.data[output+'_True'] = self.data[output]>0

        self.n_total = len(data)
        self.input_labels = input_labels
        self.input_bounds = {name: (min(data[name]), max(data[name])) for i, name in enumerate(input_labels)}
        self.output_labels = output_labels
        self.initial_sample_type = initial_sample_type
        self.input_data = pd.DataFrame(data,columns=input_labels)

        if n_init is None:
            self.n_init = round(self.n_total/10)
        else:
            self.n_init = n_init
        if n_mid is None:
            self.n_mid = round(self.n_total/10)
        else:
            self.n_mid = n_mid
        if n_final is None:
            self.n_final = round(self.n_total/10)
        else:
            self.n_final = n_final

        if basis is None:
            self.basis = 'cubic'
        else:
            self.basis = basis

        # store surrogates for each output
        self.surrogates = {}
        self.n_iter = {}
        for output in output_labels:
            self.surrogates[output] = {}
            self.surrogates[output]['MAE'] = []
            self.surrogates[output]['MSE'] = []
            self.surrogates[output]['R2'] = []
            self.surrogates[output]['RMSE'] = []
            self.surrogates[output]['SSE'] = []
            self.surrogates[output]['maxAE'] = []
            self.surrogates[output]['false_positive'] = []
            self.surrogates[output]['false_negative'] = []
            self.surrogates[output]['true_positive'] = []
            self.surrogates[output]['true_negative'] = []
            self.surrogates[output]['misclassification'] = []
            self.surrogates[output]['sensitivity'] = []
            self.surrogates[output]['specificity'] = []
            self.surrogates[output]['balanced_accuracy'] = []
            self.n_iter[output] = 0

    # training data
        self.training_samples = {}
        for output in output_labels:
            # get initial uniform sample
            self.get_initial_sample(output)

        self.n_add = n_add

    def get_initial_sample(self,output):
        df = self.input_data
        df[output] = self.data[output]
        if self.initial_sample_type == 'Hammersley':
            self.training_samples[output] = HammersleySampling(df, self.n_init, sampling_type='selection').sample_points()

        elif self.initial_sample_type == 'LatinHypercube':
            self.training_samples[output] = LatinHypercubeSampling(df, self.n_init, sampling_type='selection').sample_points()

        else:
            raise ValueError('initial sample type not included')

    def fit_surrogate(self, output):
        trainer = PysmoRBFTrainer(input_labels=self.input_labels,
                                  output_labels=[output],
                                  training_dataframe=self.training_samples[output]
                                  )

        trainer.config.basis_function = self.basis
        rbf_train = trainer.train_surrogate()
        rbf_surr = PysmoSurrogate(rbf_train,
                                  self.input_labels,
                                  [output],
                                  self.input_bounds)
        self.surrogates[output]['surrogate'] = rbf_surr
        self.get_surrogate_metrics(output)
        # New fit, so increased an iteration
        self.n_iter[output] += 1

    def get_surrogate_metrics(self, output):
        err = compute_fit_metrics(self.surrogates[output]['surrogate'], self.data)
        err = pd.DataFrame.from_dict(err)
        for err_type in err.index.to_list():
            self.surrogates[output][err_type] += [err.loc[err_type, output]]

        # Predicted values
        self.data_pred = self.surrogates[output]['surrogate'].evaluate_surrogate(self.data)

        # Get misclassification rates
        if output == 'repassivation_corrosion_potential_difference' or 'synthetic_potential_difference_V':
            label = output+'_True'
            self.data_pred[label] = self.data_pred[output] >=0
            self.surrogates[output]['true_positive'] += [((self.data[label] == True) & (self.data_pred[label] == True)).sum()]
            self.surrogates[output]['true_negative'] += [((self.data[label] == False) & (self.data_pred[label] == False)).sum()]
            self.surrogates[output]['false_positive'] += [((self.data[label] == False) & (self.data_pred[label] == True)).sum()]
            self.surrogates[output]['false_negative'] += [((self.data[label] == True) & (self.data_pred[label] == False)).sum()]
            misclassified = self.surrogates[output]['false_negative'][self.n_iter[output]] + self.surrogates[output]['false_positive'][self.n_iter[output]]
            self.surrogates[output]['misclassification'] += [misclassified/self.n_total*100]
            tp = ((self.data[label] == True) & (self.data_pred[label] == True)).sum()
            tn = ((self.data[label] == False) & (self.data_pred[label] == False)).sum()
            fp = ((self.data[label] == False) & (self.data_pred[label] == True)).sum()
            fn = ((self.data[label] == True) & (self.data_pred[label] == False)).sum()
            self.surrogates[output]['sensitivity'] += [tp/(tp+fn)*100]
            self.surrogates[output]['specificity'] += [tn/(tn+fp)*100]
            self.surrogates[output]['balanced_accuracy'] += [0.5*(tp/(tp+fn)+tn/(tn+fp))*100]
        else:
            self.surrogates[output]['true_positive'] += [0]
            self.surrogates[output]['true_negative'] += [0]
            self.surrogates[output]['false_positive'] += [0]
            self.surrogates[output]['false_negative'] += [0]
            self.surrogates[output]['misclassification'] += [0]
            self.surrogates[output]['sensitivity'] += [0]
            self.surrogates[output]['specificity'] += [0]
            self.surrogates[output]['balanced_accuracy'] += [0]

        # Calculate absolute errors
        self.data[output+'_AE'] = abs(self.data[output]-self.data_pred[output])
        # Get point with worst error
        self.index_max_error = self.data.nlargest(self.n_add,output+'_AE').index

        # Get point with worst error in range of -0.1 to 0.1
        if output == 'repassivation_corrosion_potential_difference' or output == 'synthetic_potential_difference_V':
            data_scaling_range = self.data[(self.data[output]>-0.1) & (self.data[output]<0.1)]
            self.index_max_error_scaling_range = data_scaling_range.nlargest(self.n_add, output+'_AE').index
            # Get classification rate and worst misclassified point
            misclassified = self.data[self.data[label] != self.data_pred[label]]
            sorted = misclassified.sort_values(by=output+'_AE', ascending=False)
            self.index_max_error_misclassified = []
            for i in range(self.n_add):
                try:
                    self.index_max_error_misclassified += [sorted.index[i-1]]
                except (ValueError, IndexError):
                    self.index_max_error_misclassified += list(self.index_max_error_scaling_range[:self.n_add-i])
                    break
        else:
            self.index_max_error_scaling_range = self.index_max_error
            self.index_max_error_misclassified = self.index_max_error

    def add_training_sample(self,type='worst_misclassified',output=None):
        if type == 'worst_misclassified':
            idx = self.index_max_error_misclassified
        elif type == 'worst_error_scaling_range':
            idx = self.index_max_error_scaling_range
        elif type == 'worst_error':
            idx = self.index_max_error
        for i in range(self.n_add):
            new_sample = self.data.loc[idx[i],self.input_labels+[output]]
            self.training_samples[output].loc[len(self.training_samples[output])] = new_sample

    def plot_errors(self,output,folder):
        errors = ['MSE', 'misclassification', 'maxAE']
        error_name = ['Mean square error', 'Misclassification rate (%)', 'Maximum absolute error']
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 8))
        for i,err in enumerate(errors):
            ax[i].plot(range(1,self.n_iter[output]+1), self.surrogates[output][err],color='r',linewidth=2,label=error_name[i])
            ax[i].set_xlabel('Iterations')
            ax[i].set_ylabel(error_name[i])
            # check for NAN values
            ax[i].set_ylim([0,max(self.surrogates[output][err])*1.1])
            ax[i].set_xlim([1, self.n_iter[output]])
            # ax[i].axvline(x=self.n_mid, color='k',linestyle='--')
            ax[i].axvspan(1,self.n_mid+1,color='slategray',alpha=0.3)
            # ax[i].text(1,np.mean(self.surrogates[output][err]), 'Middle phase', fontsize=10)
            # ax[i].text(self.n_mid+2,np.mean(self.surrogates[output][err]), 'Final phase', fontsize=10)
            ax[i].legend(frameon=False)
        fig.tight_layout()
        out_file = folder + output+'_' +str(self.n_init) +'_' + str(self.n_mid) + '_' + str(self.n_final) + '_' + str(self.n_add)+ f'_{self.basis}_error_vs_iteration.png'
        long_path = r"\\?\\" + os.path.abspath(str(out_file))
        plt.savefig(long_path, bbox_inches='tight', dpi=300)

    def save_errors(self, output, folder):
        df_errors = pd.DataFrame(self.surrogates[output])
        out_file = folder + output+'_' +str(self.n_init) +'_' + str(self.n_mid) + '_' + str(self.n_final) + '_' + str(self.n_add)+ f'_{self.basis}_errors_vs_iteration.csv'
        long_path = r"\\?\\" + os.path.abspath(str(out_file))
        df_errors.to_csv(long_path, index=False)

    def surrogate_validation_plots(self,output,folder, material, ph=7.5):
        if output == 'corrosionRateMmPerYear':
            ylabel = 'Corrosion rate (mm/yr)'
        elif output == 'repassivation_corrosion_potential_difference':
            ylabel = r'$V_{r}-V_{c}$ V(SHE)'
        # add predictions to dataframe
        self.data['pred'] = self.data_pred[output]

        # plot output at 2 temps, 2 do's vs recovery
        unique_temp_values = self.data['temperature_C'].unique()
        n_temps = len(unique_temp_values)
        unique_dissolved_oxy = self.data['do_mg_L'].unique()
        temp_1 = unique_temp_values[1]
        temp_2 = unique_temp_values[n_temps-2]
        do_1 = unique_dissolved_oxy[0]
        do_2 = unique_dissolved_oxy[-1]

        df_temp_1_do_1 = self.data[(self.data['temperature_C'] == temp_1)&(self.data['do_mg_L'] == do_1)]
        df_temp_1_do_2 = self.data[(self.data['temperature_C'] == temp_1)&(self.data['do_mg_L'] == do_2)]
        df_temp_2_do_1 = self.data[(self.data['temperature_C'] == temp_2)&(self.data['do_mg_L'] == do_1)]
        df_temp_2_do_2 = self.data[(self.data['temperature_C'] == temp_2)&(self.data['do_mg_L'] == do_2)]

        df_temp_1_do_1 = df_temp_1_do_1[df_temp_1_do_1['pH'] == ph]
        df_temp_1_do_2 = df_temp_1_do_2[df_temp_1_do_2['pH'] == ph]
        df_temp_2_do_1 = df_temp_2_do_1[df_temp_2_do_1['pH'] == ph]
        df_temp_2_do_2 = df_temp_2_do_2[df_temp_2_do_2['pH'] == ph]

        colormap = plt.cm.viridis(np.linspace(0,1,4))  # different color for case
        size = 10
        plt.figure(figsize=(3.25, 3.25))
        # plot surrogate fit
        plt.plot(df_temp_1_do_1['salinity_kg_kg']*1000,
                 df_temp_1_do_1['pred'],
                 color=colormap[0],
                 label=f'T = {temp_1:.0f} C, DO = {do_1:.1f} mg/L')
        plt.plot(df_temp_2_do_1['salinity_kg_kg']*1000,
                 df_temp_2_do_1['pred'],
                 color=colormap[1],
                 label=f'T = {temp_2:.0f} C, DO = {do_1:.1f} mg/L')
        plt.plot(df_temp_1_do_2['salinity_kg_kg']*1000,
                 df_temp_1_do_2['pred'],
                 color=colormap[2],
                 label=f'T = {temp_1:.0f} C, DO = {do_2:.1f} mg/L')
        plt.plot(df_temp_2_do_2['salinity_kg_kg']*1000,
                 df_temp_2_do_2['pred'],
                 color=colormap[3],
                 label=f'T = {temp_2:.0f} C, DO = {do_2:.1f} mg/L')
        # plot OLI data
        plt.scatter(df_temp_1_do_1['salinity_kg_kg']*1000,
                    df_temp_1_do_1[output],
                    edgecolor=colormap[0],
                    marker='o',
                    label='OLI',
                    facecolor=colormap[0],
                    s=size)
        plt.scatter(df_temp_2_do_1['salinity_kg_kg']*1000,
                    df_temp_2_do_1[output],
                    edgecolor=colormap[1],
                    marker='o',
                    # label='OLI',
                    facecolor=colormap[1],
                    s=size)
        plt.scatter(df_temp_1_do_2['salinity_kg_kg']*1000,
                    df_temp_1_do_2[output],
                    edgecolor=colormap[2],
                    marker='o',
                    # label='OLI',
                    facecolor=colormap[2],
                    s=size)
        plt.scatter(df_temp_2_do_2['salinity_kg_kg']*1000,
                    df_temp_2_do_2[output],
                    edgecolor=colormap[3],
                    marker='o',
                    # label='OLI',
                    facecolor=colormap[3],
                    s=size)

        plt.xlabel('Brine salinity (g/kg)')
        plt.ylabel(ylabel)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(frameon=False, fontsize=6)
        plt.title(f'{material}, pH = {ph}')
        plt.xlim(min(self.data['salinity_kg_kg'])*1000, max(self.data['salinity_kg_kg'])*1000)
        plt.tight_layout()
        out_file = folder + 'error_fit_figures/'+ output+'_' +str(self.n_init) +'_' + str(self.n_mid) + '_' + str(self.n_final) + '_' + str(self.n_add)+ f'_{self.basis}_surrogate OLI data'
        long_path = r"\\?\\" + os.path.abspath(str(out_file + '.png'))
        plt.savefig(long_path, bbox_inches='tight', dpi=300)
        long_path = r"\\?\\" + os.path.abspath(str(out_file + '.svg'))
        plt.savefig(long_path, bbox_inches='tight', dpi=300)

def create_error_metrics_dict():
    error_metrics = {}
    error_metrics['output'] = []
    error_metrics['n_init'] = []
    error_metrics['n_mid'] = []
    error_metrics['n_final'] = []
    error_metrics['n_iter'] = []
    error_metrics['n_add'] = []
    error_metrics['n_training'] = []
    error_metrics['MAE'] = []
    error_metrics['MSE'] = []
    error_metrics['R2'] = []
    error_metrics['RMSE'] = []
    error_metrics['SSE'] = []
    error_metrics['maxAE'] = []
    error_metrics['false_positive'] = []
    error_metrics['false_negative'] = []
    error_metrics['true_positive'] = []
    error_metrics['true_negative'] = []
    error_metrics['misclassification'] = []
    error_metrics['sensitivity'] = []
    error_metrics['specificity'] = []
    error_metrics['balanced_accuracy'] = []

    return error_metrics

def update_error_metrics(error_metrics, sf_object,output):
    error_metrics['output'] += [output]
    error_metrics['n_init'] += [sf_object.n_init]
    error_metrics['n_mid'] += [sf_object.n_mid]
    error_metrics['n_final'] += [sf_object.n_final]
    error_metrics['n_iter'] += [sf_object.n_iter[output]]
    error_metrics['n_add'] += [sf_object.n_add]
    error_metrics['n_training'] += [len(sf_object.training_samples[output])]
    error_metrics['MAE'] += [sf_object.surrogates[output]['MAE'][-1]]
    error_metrics['MSE'] += [sf_object.surrogates[output]['MSE'][-1]]
    error_metrics['R2'] += [sf_object.surrogates[output]['R2'][-1]]
    error_metrics['RMSE'] += [sf_object.surrogates[output]['RMSE'][-1]]
    error_metrics['SSE'] += [sf_object.surrogates[output]['SSE'][-1]]
    error_metrics['maxAE'] += [sf_object.surrogates[output]['maxAE'][-1]]
    error_metrics['false_positive'] += [sf_object.surrogates[output]['false_positive'][-1]]
    error_metrics['false_negative'] += [sf_object.surrogates[output]['false_negative'][-1]]
    error_metrics['true_positive'] += [sf_object.surrogates[output]['true_positive'][-1]]
    error_metrics['true_negative'] += [sf_object.surrogates[output]['true_negative'][-1]]
    error_metrics['misclassification'] += [sf_object.surrogates[output]['misclassification'][-1]]
    error_metrics['sensitivity'] += [sf_object.surrogates[output]['sensitivity'][-1]]
    error_metrics['specificity'] += [sf_object.surrogates[output]['specificity'][-1]]
    error_metrics['balanced_accuracy'] += [sf_object.surrogates[output]['balanced_accuracy'][-1]]


def fit_corrosion_surrogate(material, n_values, output, basis):
    """
        Fits surrogate for provide output for given material.
        material: str
        n_values: list of number of points to try [(initial points,
                                                    middle phase iteration,
                                                    final phase iterations,
                                                    number of points to add at each iteration)]
        output: str, column name of output variable
    """
    folder = f"corrosion_example/temp_ph_do_recovery_surrogate_models/{material}/surrogate_fitting/{output}/"
    survey_path = f"corrosion_example/temp_ph_do_recovery_surrogate_models/surveys/{material}_extend_do.parquet"
    data = pd.read_parquet(survey_path)
    error_metrics = create_error_metrics_dict()
    output_labels = [output]

    # try different numbers initial points,adaptive iterations,  additional points
    for n in n_values:
        # create surrogate object
        sf = surrogateFitting(data=data,
                              input_labels=['temperature_C', 'salinity_kg_kg', 'do_mg_L', 'pH'],
                              output_labels=output_labels,
                              n_init=n[0],
                              n_mid=n[1],
                              n_final=n[2],
                              n_add=n[3],
                              initial_sample_type='Hammersley',
                              basis=basis)
        # Initial fit
        sf.fit_surrogate(output)

        # Add points based on worst misclassified point
        for i in range(sf.n_mid):
            sf.add_training_sample(type='worst_misclassified', output=output)
            sf.fit_surrogate(output)

        # Add points based on worst error
        for i in range(sf.n_final):
            sf.add_training_sample(type='worst_error', output=output)
            sf.fit_surrogate(output)

        sf.save_errors(output, folder)


        # Save final errors
        update_error_metrics(error_metrics,sf,output)
        sf.plot_errors(output, folder)
        model = sf.surrogates[output]['surrogate'].save_to_file(folder+output+'_'+str(n[0])+'_'+str(n[1])+'_'+str(n[2])+'_'+str(n[3])+'_'+f'pysmo_rbf_{basis}_surrogate.json', overwrite=True)

        # plotting
        sf.surrogate_validation_plots(output,folder,material)

    # Save error metrics
    error_df = pd.DataFrame(error_metrics)
    error_df.to_csv(folder + output + f'_{basis}_surrogate_error_metrics.csv', index=False)

    return


def fit_synthetic_surrogate(n, basis='cubic'):
    folder = f"src/corrosion_analysis/surrogate_fitting/synthetic_surrogate/"
    survey_path = f"src/corrosion_analysis/data/synthetic_potential_difference.csv"
    data = pd.read_csv(survey_path)
    output = 'synthetic_potential_difference_V'

    # create surrogate object
    sf = surrogateFitting(data=data,
                          input_labels=['temperature_C','do_mg_L'],
                          output_labels=[output],
                          n_init=n[0],
                          n_mid=n[1],
                          n_final=n[2],
                          n_add=n[3],
                          initial_sample_type='Hammersley',
                          basis=basis)
    # Initial fit
    sf.fit_surrogate(output)
    # Add points based on worst misclassified point
    for i in range(sf.n_mid):
        sf.add_training_sample(type='worst_misclassified', output=output)
        sf.fit_surrogate(output)
    # Add points based on worst error
    for i in range(sf.n_final):
        sf.add_training_sample(type='worst_error', output=output)
        sf.fit_surrogate(output)

    # save errors vs iteration
    sf.save_errors(output, folder)

    # Plot errors
    sf.plot_errors(output, folder)

    # save final surrogate
    surrogate_file = folder + output + '_' + str(n[0]) + '_' + str(n[1]) + '_' + str(n[2]) + '_' + str(
        n[3]) + f'_pysmo_rbf_{basis}_surrogate.json'
    long_path = r"\\?\\" + os.path.abspath(str(surrogate_file))
    model = sf.surrogates[output]['surrogate'].save_to_file(long_path, overwrite=True)

    return

if __name__ == '__main__':
    fit_synthetic_surrogate((10, 3, 3, 5))