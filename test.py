import glob
import pickle
from torch.utils.data.dataloader import DataLoader
from utils import *
from metrics import *
from model_baseline import TrajectoryModel
from model_groupwrapper import GPGraph


@torch.no_grad()
def test(SAMPLES=20, TRIALS=100):
    global loader_test, model

    model.eval()
    ade_all, fde_all, col_all, tcc_all = [], [], [], []
    for batch in tqdm(loader_test):
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = [tensor.cuda(non_blocking=True) for tensor in batch[:4]]
        V_obs, V_tr = [tensor.cuda(non_blocking=True) for tensor in batch[-2:]]

        V_obs_abs = obs_traj.permute(0, 2, 3, 1)
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, indices = model(V_obs_abs, V_obs_tmp)
        V_pred = V_pred.permute(0, 2, 3, 1)

        V_pred = V_pred.squeeze()
        V_obs_traj = obs_traj.permute(0, 3, 1, 2).squeeze(dim=0)
        V_pred_traj_gt = pred_traj.permute(0, 3, 1, 2).squeeze(dim=0)

        mu, cov = generate_statistics_matrices(V_pred.squeeze(dim=0))
        ade_stack, fde_stack, tcc_stack, col_stack = [], [], [], []

        # Reduce randomness by repeating the evaluation process TRIALS=100 times.
        # To evaluate the execution time, TRIALS must be set to 1.
        for trial in range(TRIALS):
            sample_level = 'group'  # ['scene', 'pedestrian', 'group']
            if sample_level == 'scene':
                r_sample = random.randn(1, SAMPLES, 2)
            elif sample_level == 'pedestrian':
                r_sample = random.randn(indices.size(0), SAMPLES, 2)
            elif sample_level == 'group':
                r_sample = random.randn(indices.unique().size(0), SAMPLES, 2)[indices.detach().cpu().numpy()]
                r_sample = r_sample.take(indices.detach().cpu().numpy(), axis=0)
            else:
                raise NotImplementedError

            r_sample = torch.Tensor(r_sample).to(dtype=mu.dtype, device=mu.device)
            r_sample = r_sample.permute(1, 0, 2).unsqueeze(dim=1).expand((SAMPLES,) + mu.shape)
            V_pred_sample = mu + (torch.cholesky(cov) @ r_sample.unsqueeze(dim=-1)).squeeze(dim=-1)

            V_absl = V_pred_sample.cumsum(dim=1) + V_obs_traj[[-1], :, :]
            ADEs, FDEs, COLs, TCCs = compute_batch_metric(V_absl, V_pred_traj_gt)

            ade_stack.append(ADEs.detach().cpu().numpy())
            fde_stack.append(FDEs.detach().cpu().numpy())
            col_stack.append(COLs.detach().cpu().numpy())
            tcc_stack.append(TCCs.detach().cpu().numpy())

        ade_all.append(np.array(ade_stack))
        fde_all.append(np.array(fde_stack))
        col_all.append(np.array(col_stack))
        tcc_all.append(np.array(tcc_stack))

    ade_all = np.concatenate(ade_all, axis=1)
    fde_all = np.concatenate(fde_all, axis=1)
    col_all = np.concatenate(col_all, axis=1)
    tcc_all = np.concatenate(tcc_all, axis=1)

    mean_ade, mean_fde = ade_all.mean(axis=0).mean(), fde_all.mean(axis=0).mean()
    mean_col, mean_tcc = col_all.mean(axis=0).mean(), tcc_all.mean(axis=0).mean()
    return mean_ade, mean_fde, mean_col, mean_tcc


paths = ['./checkpoints/GPGraph-SGCN/*']
SAMPLES = 20

print("*" * 50)
print('Number of samples:', SAMPLES)
print("*" * 50)

for feta in range(len(paths)):
    SCENE_ls, ADE_ls, FDE_ls, COL_ls, TCC_ls = [], [], [], [], []
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:', exps)

    for exp_path in exps:
        print("*" * 50)
        print("Evaluating model:", exp_path)

        model_path = exp_path + '/val_best.pth'
        args_path = exp_path + '/args.pkl'
        with open(args_path, 'rb') as f:
            args = pickle.load(f)

        stats = exp_path + '/constant_metrics.pkl'
        with open(stats, 'rb') as f:
            cm = pickle.load(f)
        # print("Stats:", cm)

        # Data prep
        obs_seq_len = 8
        pred_seq_len = 12
        data_set = './dataset/' + args.dataset + '/'

        dset_test = TrajectoryDataset(data_set + 'test/', obs_len=obs_seq_len, pred_len=pred_seq_len, skip=1)
        loader_test = DataLoader(dset_test, batch_size=1, shuffle=False, num_workers=0)

        # Defining the model
        base_model = TrajectoryModel(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                                     obs_len=8, pred_len=12, n_tcn=5, out_dims=5).cuda()
        model = GPGraph(baseline_model=base_model, in_channels=2, out_channels=5,
                        obs_seq_len=8, pred_seq_len=12,
                        d_type='learned_l2norm', d_th='learned', mix_type='mlp',
                        group_type=(True, True, True), weight_share=True).cuda()
        model.load_state_dict(torch.load(model_path))

        print("Testing...", end=' ')
        ADE, FDE, COL, TCC = test()
        ADE_ls.append(ADE)
        FDE_ls.append(FDE)
        COL_ls.append(COL)
        TCC_ls.append(TCC)
        SCENE_ls.append(args.dataset)
        print("Scene: {} ADE: {:.8f} FDE: {:.8f} COL: {:.8f}, TCC: {:.8f}".format(args.dataset, ADE, FDE, COL, TCC))

    print("*" * 50)
    ADE_ls, FDE_ls, COL_ls, TCC_ls = np.array(ADE_ls), np.array(FDE_ls), np.array(COL_ls), np.array(TCC_ls)
    print("Average ADE: {:.8f} FDE: {:.8f} COL: {:.8f}, TCC: {:.8f}".format(ADE_ls.mean(), FDE_ls.mean(),
                                                                            COL_ls.mean(), TCC_ls.mean()))

    print("*" * 50)
    print('SCENE\tADE \tFDE \tCOL \tTCC')
    for scene, ade, fde, col, tcc in zip(SCENE_ls, ADE_ls, FDE_ls, COL_ls, TCC_ls):
        print('{} \t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(scene.upper(), ade, fde, col, tcc))
    print('AVG \t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(ADE_ls.mean(), FDE_ls.mean(), COL_ls.mean(), TCC_ls.mean()))
    print("*" * 50)
