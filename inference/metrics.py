import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from skimage.metrics import structural_similarity as ssim

from dataloader.observation import load_data

class RelativeErrorObs():
    # TO calculate for Observation data
    def __init__(self, dict_minmax, dict_meanstd):
        self.dict_minmax = dict_minmax
        self.dict_meanstd = dict_meanstd

    def create_dataloader(self, model_type):
        if model_type=='diffusion':
            dataset_polair_generative = load_data(img_shape=(4,75,110), normalisation=True)
            dataloader = DataLoader(dataset_polair_generative, batch_size=32, shuffle=False)
        else:
            dataset_polair_discrim = load_data(img_shape=(4,75,110), normalisation=False)
            dataloader = DataLoader(dataset_polair_discrim, batch_size=32, shuffle=False)

        return dataloader

    def rel_err_discrim(self, pred_img, org_img, mask):
        '''
        Returns channel wise and mean relative error for held-out validation set for discriminative model
        '''

        chan_rela_err = []
        for i in range(pred_img.shape[1]):
            data_min = self.dict_minmax[i][0]
            data_max = self.dict_minmax[i][1]
            pred_img[:,i,:,:] = pred_img[:,i,:,:]*(data_max-data_min) + data_min

        eval_mask = (1-mask)*org_img
        eval_pred = torch.where(eval_mask > 0, pred_img, torch.tensor(0))

        abs_err = torch.abs(eval_mask - eval_pred)

        for i in range(4):
            mean_err = abs_err[:,i,:,:][eval_mask[:,i,:,:]>0]
            relative_err = mean_err/(eval_mask[:,i,:,:][eval_mask[:,i,:,:]>0])
            chan_rela_err.append(relative_err)
            
        # Mean error over all channels
        mean_err = abs_err[eval_mask>0]
        relative_err = mean_err/(eval_mask[eval_mask>0])

        return relative_err, chan_rela_err
    
    def rel_err_generative(self, pred_img, org_img, mask):
        '''
        Returns channel wise and mean relative error for held-out validation set for generative model
        '''
        chan_rela_err = []
        eval_mask = (1-mask)*org_img
        eval_truth = (1-mask)*org_img
        eval_pred = torch.where(eval_mask != 0, pred_img, torch.tensor(0))
        
        for i in range(org_img.shape[1]):
            data_mean = self.dict_meanstd[i][0]
            data_std = self.dict_meanstd[i][1]
            eval_pred[:,i,:,:] = eval_pred[:,i,:,:]*data_std/0.5 + data_mean
            eval_truth[:,i,:,:] = eval_truth[:,i,:,:]*data_std/0.5 + data_mean

        abs_err = torch.abs(eval_truth - eval_pred)

        for i in range(4):
            mean_err = abs_err[:,i,:,:][eval_mask[:,i,:,:]!=0]
            relative_err = mean_err/(eval_truth[:,i,:,:][eval_mask[:,i,:,:]!=0])
            chan_rela_err.append(relative_err)

        # Mean Error over channels
        mean_err = abs_err[eval_mask!=0]
        relative_err = mean_err/(eval_truth[eval_mask!=0])

        return relative_err, chan_rela_err
    
    def save_numpy(self,chan_rel_err, all_rel_err, type):
        # Save relative error numpy files
        np.savez("result/observation_data/relative_error/numpy_files/channel_relative_error_"+ type +".npz", **chan_rel_err)
        np.save('result/observation_data/relative_error/numpy_files/all_relative_error_'+ type +'.npy', all_rel_err)

    def create_channel_plot(self, chan_rel_err, type):
        # Creates histogram plot for channel-wise relative error

        f,ax = plt.subplots(2, 2, figsize=(8,6))
        k=0
        title = ['NO2', 'O3', 'PM10', 'PM2.5']
        for i in range(2):
            for j in range(2):

                ax[i,j].hist(chan_rel_err[k], bins=500, color='blue', alpha=0.5, label='max: '+ str(chan_rel_err[k].max()), density=True)
                ax[i,j].axvline(chan_rel_err[k].mean(), color='blue', linestyle='--', linewidth=2, label='mean: ' + str(chan_rel_err[k].mean()))
                ax[i,j].axvline(np.median(chan_rel_err[k]), color='green', linestyle='--', linewidth=2, label='median: ' + str(np.median(chan_rel_err[k])))
                ax[i,j].legend()
                ax[i,j].grid()
                ax[i,j].set_title('(Real Data): '+ title[k])
                ax[i,j].set_xlim(0,3)
                k=k+1
        f.tight_layout()
        plt.savefig('result/observation_data/relative_error/rel_err_' + type + '_channel.svg')
        plt.close()


    def create_mean_plot(self, all_relative_error, type):
        # Creates histogram plot for mean relative error

        plt.hist(all_relative_error, bins=500, color='blue', alpha=0.5, label='max: '+ str(all_relative_error.max()), density=True)
        plt.axvline(all_relative_error.mean(), color='blue', linestyle='--', linewidth=2, label='mean: ' + str(all_relative_error.mean()))
        plt.axvline(np.median(all_relative_error), color='green', linestyle='--', linewidth=2, label='median: ' + str(np.median(all_relative_error)))
        plt.legend()
        plt.grid()
        plt.title(type + ' Relative Error Histogram (Real Data)')
        plt.xlim(0,6)
        plt.savefig('result/observation_data/relative_error/rel_err_' + type + '.svg')
        plt.close()

    
    def calculate_error(self, model, model_type, dataloader=None):
        chan_rel_err = {0:[], 1:[], 2:[], 3:[]}
        all_rel_err = []
        if dataloader is None:
            dataloader = self.create_dataloader(model_type)

        for mask, vt, org_img in dataloader:

            if model_type=='diffusion':
                sample_img = model.ensemble_prediction(org_img, mask, vt, num_inf_steps=10, num_ensem_steps=20, device='cuda')[-1]
            else:
                sample_img = model.call_model(vt)

            org_img = org_img.cpu()
            sample_img = sample_img.cpu()
            vt = vt.cpu()

            if model_type=='diffusion':
                rel_error, relative_err_chan = self.rel_err_generative(sample_img, org_img, mask)
            else:
                rel_error, relative_err_chan = self.rel_err_discrim(sample_img, org_img, mask)

            all_rel_err.append(rel_error)
            for k in range(4):
                chan_rel_err[k].append(relative_err_chan[k])

        all_rel_err = (torch.cat(all_rel_err)).numpy()

        for k in range(4):
            chan_rel_err[k] = (torch.cat(chan_rel_err[k])).numpy()

        # Save plots
        self.create_channel_plot(chan_rel_err, model_type)
        self.create_mean_plot(all_rel_err, model_type)

        # Save numpy files
        chan_rel_err = {str(k): v for k, v in chan_rel_err.items()}
        self.save_numpy(chan_rel_err, all_rel_err, model_type)
        
####

class EvaluationMetricsSim():
    # TO calculate for simulation data
    # For discriminative model
    def __init__(self, dict_minmax):
        self.dict_minmax = dict_minmax

    def save_metrics(self, relative_error, ssim_meas, model_type):
        data = {
            'Relative error': relative_error.numpy().mean(0).tolist(),
            'SSIM': ssim_meas
        }

        with open('result/simulation_data/output_metric_'+ model_type +'.json', 'w') as f:
            json.dump(data, f, indent=4)

        print("JSON file created successfully!")
        savepath = 'result/simulation_data/relative_error/numpy_files/relative_error_'+ model_type +'.npy'
        np.save(savepath, relative_error)

        return savepath

    def evaluate_model(self, model, dataloader, model_type, device='cuda'):
        '''
        Creates relative error, ssim and psnr metrics
        '''
        relative_error = []
        psnr_meas = torch.zeros(4)
        ssim_meas = {0:0, 1:0, 2:0, 3:0}

        with torch.no_grad():
            max_pixel = torch.Tensor([self.dict_minmax[0][1], self.dict_minmax[1][1], self.dict_minmax[2][1], self.dict_minmax[3][1]])
            max_pixel = max_pixel.unsqueeze(0)

            for org_img, _, vt in dataloader:

                org_img = org_img.to(device)
                vt = vt.to(device)

                predicted_img = model.call_model(vt)

                for k in range(4):
                    data_min = self.dict_minmax [k][0]
                    data_max = self.dict_minmax[k][1]
                    org_img[:,k,:,:] = org_img[:,k,:,:]*(data_max-data_min) + data_min
                    predicted_img[:,k,:,:] = predicted_img[:,k,:,:]*(data_max-data_min) + data_min


                # relative error
                error_img = torch.norm(predicted_img - org_img, p=2, dim=(2,3))

                ref = torch.norm(org_img, p=2, dim=(2,3))
                relative_error.append(error_img/ref)

                # ssim
                for j in range(4):
                    for i in range(predicted_img.shape[0]):
                        pred = predicted_img[i,j,:,:].cpu().numpy()
                        ssim_meas[j] += ssim(pred, org_img[i,j,:,:].cpu().numpy(), data_range=pred.max()-pred.min() )
        
        for j in range(4):
            ssim_meas[j] = float(ssim_meas[j]/len(dataloader.dataset))
        
        relative_error = torch.cat(relative_error, dim=0)

        savepath = self.save_metrics(relative_error.cpu(), ssim_meas, model_type)
        return savepath


class DiffusionEvaluation():
    def __init__(self, dict_meanstd):
        self.mean_std_data = dict_meanstd


    def evaluate_model(self, model, dataloader, inf_steps=10, ensem_size=20, device='cuda'):
        '''
        Creates relative error, ssim metrics for a specific ensemble size (E) and inference step (T)
        '''
        all_relative_error_dict = {}
        for i in range(ensem_size):
            all_relative_error_dict[i] = []
        ssim_meas = np.zeros((ensem_size, 4))

        for org_img, mask, vt in dataloader:

            sample_img_list = model.ensemble_prediction(org_img, mask, vt, num_inf_steps= inf_steps, num_ensem_steps=ensem_size, device=device)

            for k in range(4):
                data_mean = self.mean_std_data[k][0]
                data_std = self.mean_std_data[k][1]

                org_img[:,k,:,:] = org_img[:,k,:,:]*data_std/0.5 + data_mean
            
            for c, sample_img in enumerate(sample_img_list):

                # Denormalize
                for k in range(4):
                    data_mean = self.mean_std_data[k][0]
                    data_std = self.mean_std_data[k][1]

                    sample_img[:,k,:,:] = sample_img[:,k,:,:]*data_std/0.5 + data_mean


                # Relative error per channel
                error_img = torch.norm(sample_img - org_img, p=2, dim=(2,3))    # (32,4)

                ref = torch.norm(org_img, p=2, dim=(2,3))
                all_relative_error_dict[c].append(error_img/ref)         # [(32,4), ....., (32,4)]


                # SSIM per channel
                for j in range(4):
                    for i in range(sample_img.shape[0]):
                        pred = sample_img[i,j,:,:].cpu().numpy()
                        ssim_meas[c][j] += ssim(pred, org_img[i,j,:,:].cpu().numpy(), data_range=pred.max()-pred.min() )

                
            
        ssim_meas = ssim_meas/len(dataloader.dataset)
        last_relative_error = all_relative_error_dict[ensem_size-1]
            
        for c in range(ensem_size):
            all_relative_error_dict[c] = torch.cat(all_relative_error_dict[c], dim=0).numpy().mean(0).tolist()   #(1500, 4)

        return all_relative_error_dict, ssim_meas, last_relative_error


    
    def create_ensem_inf_metrics(self, dataloader, model, ensem_size=20, inf_steps_size=None):
        '''
        Create metrics for various inference step size and ensemble size
        '''
        if inf_steps_size is None:
            inf_steps_list = [2,5,8,10,15,20,25,30,40,50]

        for k in range(len(inf_steps_list)):


            relative_error, ssim_meas, last_rel_error = self.evaluate_model(model, dataloader, inf_steps=inf_steps_list[k], ensem_size=ensem_size)

            data = {
                'Relative error': relative_error,
                'SSIM': ssim_meas.tolist()
            }

            with open('result/simulation_data/diffusion_metrics/output_metric_inf_size_'+ str(inf_steps_list[k]) + '.json', 'w') as f:
                json.dump(data, f, indent=4)

        last_rel_error = torch.cat(last_rel_error, dim=0)
        savepath = 'result/simulation_data/relative_error/numpy_files/relative_error_diffusion.npy'
        np.save(savepath, last_rel_error.cpu())

        return savepath
        
