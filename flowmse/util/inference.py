import torch
from torchaudio import load
import torch.nn.functional as F
from pesq import pesq
from pystoi import stoi

from .other import si_sdr, pad_spec
from ..sampling import get_white_box_solver
# Settings
sr = 16000


N=5

def evaluate_model(model, num_eval_files, inference_N=5):
    T_rev = model.T_rev
    model.ode.T_rev = T_rev
    t_eps = model.t_eps
    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)
    mode_ = model.mode_
    try:
        if model.inference_N:
            inference_N = model.inference_N
    except:
        inference_N = N
    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    for (clean_file, noisy_file) in zip(clean_files, noisy_files):
        # Load wavs
        x, _ = load(clean_file)
        y, _ = load(noisy_file) 
        T_orig = x.size(1)   

        # Normalize per utterance
        norm_factor = y.abs().max()
        y = y / norm_factor

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        x1 ,z= model.ode.prior_sampling(Y.shape,Y)
        sigma = model.ode._std(1)
        # sigma = sigma.to(Y.device)
        z = z.to(Y.device)
        Y_plus_sigma_z = Y+sigma *z
        xt = x1
        # z = z.to(device=Y.device)
        timesteps = torch.linspace(T_rev, t_eps, N, device=Y.device)
        for i in range(len(timesteps)):
            t = timesteps[i]
            if i != len(timesteps) - 1:
                stepsize = t - timesteps[i+1]
            else:
                stepsize = timesteps[-1]
                
            vect = torch.ones(Y.shape[0], device=Y.device) * t
            dt = -stepsize 
            if mode_ == "noisemean_conditionfalse_timefalse":   
                xt = xt + dt * model(vect,xt)
            elif mode_ == "noisemean_noxt_conditiony_timefalse":
                xt = xt + dt * model(vect , Y)
            elif mode_ == "noisemean_y_plus_sigmaz":
                xt = xt + dt * model(vect , Y_plus_sigma_z)
            elif mode_ == "noisemean_xt_t": #noisemean_xt_t: v_theta(xt,t)
                xt = xt + dt * model(vect, xt)
            elif mode_ == "noisemean_xt_y": #noisemean_xt_y: v_theta(xt,y)
                xt = xt + dt * model(vect, xt,Y)
            elif mode_ == "noisemean_xt_y_plus_sigmaz": #v_theta(xt,y+sigma z)
                xt = xt + dt * model(vect, xt, Y_plus_sigma_z)
            elif mode_ == "noisemean_xtplusy_divide_2": #v_theta((xt+y)/2)
                xt = xt + dt *model(vect, (xt+Y)/2)
            elif mode_ == "noisemean_xt_y_t": # noisemean_xt_y_t v_theta (xt,y,t)
                xt = xt + dt * model(vect, xt, Y)
            elif mode_ == "noisemean_xt_t: v_theta(xt,t)": # noisemean_xt_t: v_theta(xt,t) 
                xt = xt + dt * model(vect, xt)
            elif mode_ == "noisemean_t_times_y_plus_sigmaz_1minust_times_s": # "noisemean_t_times_y_plus_sigmaz_1minust_times_s": #": v_theta(t(y+sigma z), (1-t)s)"
                first_variable = vect*(Y_plus_sigma_z) # t(y+sigma z)
                second_variable = xt - first_variable # (1-t)s
                xt = xt + dt * model(vect, first_variable, second_variable)
            elif (mode_ == "noisemean_xt_y_sigmaz") or (mode_ == "noisemean_xt_y_sigmaz_t"): #v_theta(t,xt,y,sigmaz)
                xt = xt + dt * model(vect, xt, Y, sigma * z)
            elif mode_ == "noisemean_t_y": #v_theta(t,y)
                xt = xt + dt * model(vect, Y)
            elif mode_ == "noisemean_xt_y_yplussigmaz": #v_theta(xt,y,y+sigma z)
                xt = xt + dt * model(vect, xt, Y, Y+sigma * z)
            elif mode_ == "noisemean_xt_y_sigmaz_yplussigmaz_t": #v_theta(t,xt,y,y+sigmaz, sigmaz)
                xt = xt + dt * model(vect, xt, Y, Y+sigma*z, sigma*z)
            elif mode_ == "flowse_KDfromclean":
                xt = xt + dt * model(vect, xt, Y)
            elif mode_ == "flowse_KD_enindg_without_t_sequential_update":
                xt = xt + dt * model(vect,xt,Y)
            elif mode_ == "CTFSE_KDfromclean":
                xt = xt + dt * model(vect, xt, Y)
            elif mode_ == "noisemean_direct_estimation_xt_y_t": #s_theta(t,xt,y)
                xt = xt + dt * (Y+sigma*z -model(vect,xt,Y))
        sample = xt
        

        sample = sample.squeeze()

   
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        _si_sdr += si_sdr(x, x_hat)
       
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files

