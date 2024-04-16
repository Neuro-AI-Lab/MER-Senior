import argparse
import pandas as pd
import numpy as np
import torch
import os

from models import GraphConvolution, GraphConvolutionalEncoder, GRACE, learner
from utils import setup_config_args, fix_seed, get_logger, get_activation, save_np, save_heatmap, make_route, get_device
from functionals import symmetric_normalization, normalization, similarity_matrix, ssm_fusion, concatenate_fusion


def get_dataset(args):
    # csv data to pandas dataframe
    data = pd.read_csv(args.data_load_path)

    # get certain attributes
    label = torch.from_numpy(data.loc[:, 'Label'].to_numpy()).to(torch.long)
    audio = torch.from_numpy(
        data.loc[:, 'audio_feature1':'audio_feature' + str(args.n_audio_features)].to_numpy()).to(
        torch.torch.float32)
    text = torch.from_numpy(data.loc[:, 'text_feature1':].to_numpy()).to(
        torch.torch.float32)

    # convert nan values to zero
    audio = torch.where(torch.isnan(audio), torch.zeros_like(audio), audio)
    text = torch.where(torch.isnan(text), torch.zeros_like(text), text)
    label = torch.where(torch.isnan(label), torch.zeros_like(label), label)

    return label, audio, text


def generate_random_unique_indices(n_rows, row_size, max_val):
    """Generates a tensor of shape (n_rows, row_size) with unique values in each row."""
    tensor = torch.empty(n_rows, row_size, dtype=torch.long)
    for i in range(n_rows):
        tensor[i] = torch.randperm(max_val)[:row_size]
    return tensor


def main(args):

    # Fix random seed
    fix_seed(args.seed)

    # Get stream and file logger
    file_name = args.dataset+'.log'
    make_route(args.log_save_path, args.dataset+'.log')
    filepath = os.path.join(args.log_save_path, file_name)
    log = get_logger(filepath=filepath)

    log.info(args)
    # select CUDA or CPU
    device = get_device(args.cuda_id)
    log.info(f'Using device: {device}')

    label, audio, text = get_dataset(args)
    
    log.info(f'Label shape : {label.shape}')
    log.info(f'Label type : {label.dtype}')
    log.info(f'Audio shape : {audio.shape}')
    log.info(f'Audio type : {audio.dtype}')
    log.info(f'Text shape : {text.shape}')
    log.info(f'Text type : {text.dtype}')


    # randomly select subjects to be tested based on the number of folds -> 6 out of 24 subjects when 4 folds
    # Generate random indices with replacement (might contain duplicates within rows)
    # Note: This approach does not guarantee uniqueness within each draw
    n_subjects_by_fold = args.n_subjects // args.n_folds

    # Generate the tensor
    random_unique_indices = generate_random_unique_indices(args.n_times_draw, n_subjects_by_fold, args.n_subjects)

    log.info("---------------- Subject-independent experiments - number of subjects: {}, number of folds: {}, number of iterations: {} --------------".format(
        int(args.n_subjects), args.n_folds, args.n_times_draw))

    log.info(f'Random unique indices : {random_unique_indices}')

    im = torch.eye(args.n_samples, dtype=torch.float32)

    """
     We conduct subject-independent experiments 30 times for reliability.
    In each experiment, we test performances based on 4-fold cross validation due to the lack of data
    """
    total_avg_list, total_std_list, total_cfm_list, total_f1_micro_list, total_f1_macro_list = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    date = '240321'
    for j in range(args.n_times_draw):
        best_acc_list, cfm_list, f1_micro_list, f1_macro_list = np.array([]), np.array([]), np.array([]), np.array([])
        log.info("=================================== Iteration {} ======================================".format(j + 1))
        for i in range(args.n_folds):
            log.info("******************* TEST fold 3:1 / Fold {} *********************".format(i + 1))
            fold_idx = 'fold_' + str(i + 1)

            audio = normalization(audio, axis=0, ntype='standardization')
            text = normalization(text, axis=0, ntype='standardization')

            log.info("similarity matrix construction start...")
            asm = similarity_matrix(audio, scale=0.9)
            sasm = (asm + asm.T) / 2
            nsasm = symmetric_normalization(sasm, im)

            tsm = similarity_matrix(text, scale=0.9)
            stsm = (tsm + tsm.T) / 2
            nstsm = symmetric_normalization(stsm, im)

            fsm = ssm_fusion(asm, nsasm, tsm, nstsm, args.k_neighbor, args.timestep)
            log.info("similarity matrix construction Done...")

            make_route(args.structure_save_path)
            save_heatmap(asm, "Audio Similarity Matrix", "Samples", "Samples",
                         args.structure_save_path + 'asm_' + fold_idx + '_' + date + '.png', clim_min=None, clim_max=None,
                         _dpi=300, _facecolor="#eeeeee", _bbox_inches='tight')
            save_heatmap(nsasm, "Normalized Symmetric Audio Similarity Matrix", "Samples", "Samples",
                         args.structure_save_path + 'nsasm_' + fold_idx + '_' + date + '.png', clim_min=None, clim_max=None,
                         _dpi=300, _facecolor="#eeeeee", _bbox_inches='tight')
            save_heatmap(tsm, "Text Similarity Matrix", "Samples", "Samples",
                         args.structure_save_path + 'tsm_' + fold_idx + '_' + date + '.png', clim_min=None, clim_max=None,
                         _dpi=300, _facecolor="#eeeeee", _bbox_inches='tight')
            save_heatmap(nstsm, "Normalized Symmetric Text Similarity Matrix", "Samples", "Samples",
                         args.structure_save_path + 'nstsm_' + fold_idx + '_' + date + '.png', clim_min=None, clim_max=None,
                         _dpi=300, _facecolor="#eeeeee", _bbox_inches='tight')

            save_heatmap(fsm, "Fused Symmetric Similarity Matrix", "Samples", "Samples",
                         args.structure_save_path + '/fsm_' + fold_idx + '_' + date + '.png', clim_min=None, clim_max=None,
                         _dpi=300, _facecolor="#eeeeee", _bbox_inches='tight')
            log.info(f"graph structures are saved to {args.structure_save_path}")

            ffm = concatenate_fusion(audio, text)
            make_route(args.feature_save_path)
            save_heatmap(ffm, "Fused Input Features", "Feature dimensions", "Samples",
                         args.feature_save_path + '/fused_feature_' + fold_idx + '_' + date + '.png', clim_min=None, clim_max=None,
                         _dpi=300, _facecolor="#eeeeee", _bbox_inches='tight')

            adj = fsm.to(device)
            feature = ffm.to(device)
            label = label.to(device)

            log.info(f"graph node features are saved to {args.feature_save_path}")

            identifier = torch.ones(args.n_subjects, args.n_trials).bool()
            identifier[random_unique_indices[j]] = False
            identifier = identifier.reshape(-1)
            train_identifier = identifier.squeeze().to(device)
            test_identifier = ~train_identifier

            activation = get_activation('celu')
            gcn = GraphConvolution

            encoder = GraphConvolutionalEncoder(args.n_features, args.gcn_hid_channels, args.gcn_out_channels, activation,
                                                base_model=gcn).to(device)
            model = GRACE(encoder, args.n_features, args.gcn_out_channels, args.proj_hid_channels, args.out_channels,
                          args.ptau).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

            model_id = 'iteration'+str(j+1)+'-fold'+str(i+1)

            make_route(args.model_save_path)
            best_acc, best_z, cfm, f1_micro, f1_macro, out_trigger, best_epoch = learner(model_id, model, optimizer, feature, adj, label,
                                                                  train_identifier, test_identifier,
                                                                  args, isdeap=False, verbose=False, earlystop=False)

            if out_trigger == 0:
                log.info('model id: {}, Epoch : {} - Find best epoch'.format(model_id, best_epoch))
            elif out_trigger == 1:
                log.info('model id: {}, Epoch : {} - Accuracy - 100.%'.format(model_id, best_epoch))
            elif out_trigger == 2:
                log.info('model id: {}, Epoch : {} - Early Stopping'.format(model_id, best_epoch))
            else:
                log.error('unidentifiable model save trigger')


            model_save_file = args.model_save_path+'subject_independent_'+model_id+'.pt'
            torch.save(model.state_dict(), model_save_file)

            make_route(args.tensor_save_path)
            # save_np(args.tensor_save_path, 'confusion_matrix_' + fold_idx, cfm)

            log.info("*** Best ACC : {} ***".format(round(best_acc.item(), 2)))
            best_acc_list = np.append(best_acc_list, best_acc.item())
            cfm_list = np.concatenate([cfm_list, cfm[np.newaxis]]) if cfm_list.size else cfm[np.newaxis]
            f1_micro_list = np.append(f1_micro_list, f1_micro.item())
            f1_macro_list = np.append(f1_macro_list, f1_macro.item())            

        avg = np.mean(best_acc_list)
        std = np.std(best_acc_list)
        log.info(f"**************** Best and Average acc of 24 folds by subject in iteration {j+1}*********************")
        log.info("** Best ACC : {},    Avearge acc : {},    std : {} **\n".format(np.round(np.max(best_acc_list), 2),
                                                                                   np.round(avg, 2),
                                                                                   np.round(std, 2)))
        total_avg_list = np.append(total_avg_list, avg)
        total_std_list = np.append(total_std_list, std)
        total_cfm_list = np.concatenate([total_cfm_list, np.mean(cfm_list, axis=0, keepdims=True)]) if total_cfm_list.size else np.mean(cfm_list, axis=0, keepdims=True)
        total_f1_micro_list = np.append(total_f1_micro_list, np.mean(f1_micro_list))
        total_f1_macro_list = np.append(total_f1_macro_list, np.mean(f1_macro_list))

    log.info(f" The average accuracy, standard deviation, f1_score(micro), f1_score(macro) : {np.mean(total_avg_list)}, {np.mean(total_std_list)}, {np.mean(total_f1_micro_list)}, {np.mean(total_f1_macro_list)}")
    save_np(args.tensor_save_path, 'subject_independent_n_folds_' + str(args.n_folds) + '_total_avg_list_' + date +'_'+ args.dataset, total_avg_list)
    save_np(args.tensor_save_path, 'subject_independent_n_folds_' + str(args.n_folds) + '_total_std_list_' + date +'_'+ args.dataset, total_std_list)
    save_np(args.tensor_save_path, 'subject_independent_n_folds_' + str(args.n_folds) + '_average_confusion_matrix_' + date +'_'+ args.dataset, np.mean(total_cfm_list, axis=0))
    save_np(args.tensor_save_path, 'subject_independent_n_folds_' + str(args.n_folds) + '_total_f1_micro_list_' + date +'_'+ args.dataset, total_f1_micro_list)
    save_np(args.tensor_save_path, 'subject_independent_n_folds_' + str(args.n_folds) + '_total_f1_macro_list_' + date +'_'+ args.dataset, total_f1_macro_list)
    # log.info(f"{np.mean(total_cfm_list, axis=0)}")
    return

if __name__ == "__main__":
    # Load config from YAML file and Setup argparse with the config
    parser = argparse.ArgumentParser(description='Arguments set from prompt and YAML configuration.')
    parser.add_argument('--dataset', type=str, default='IITP-SMED', help='dataset (default: IITP-SMED)')
    parser.add_argument('--cuda_id', type=str, default='0', help='Cuda device ID (default: 0)')
    args = setup_config_args(parser, filepath='config.yaml')

    main(args)