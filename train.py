import argparse
import pickle
from torch import optim
from torch.utils.data.dataloader import DataLoader
from metrics import *
from model_baseline import TrajectoryModel
from model_groupwrapper import GPGraph
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)
parser.add_argument('--dataset', default='eth', help='eth,hotel,univ,zara1,zara2')
parser.add_argument('--batch_size', type=int, default=128, help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=10, help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum of lr')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay on l2 reg')
parser.add_argument('--lr_sh_rate', type=int, default=100, help='number of steps to drop the lr')
parser.add_argument('--milestones', type=int, default=[50, 100], help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=True, help='Use lr rate scheduler')
parser.add_argument('--tag', default='sgcn', help='personal tag for the model ')
parser.add_argument('--gpu_num', default="0", type=str)
args = parser.parse_args()

print("Training initiating....")
print(args)


def graph_loss(V_pred, V_target):
    return bivariate_loss(V_pred, V_target)


metrics = {'train_loss': [], 'val_loss': []}
constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 1e8, 'min_train_epoch': -1, 'min_train_loss': 1e8}


def train(epoch, model, optimizer, checkpoint_dir, loader_train):
    global metrics, constant_metrics
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train):
        batch_count += 1

        # Get data
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = [tensor.cuda(non_blocking=True) for tensor in batch[:4]]
        V_obs, V_tr = [tensor.cuda(non_blocking=True) for tensor in batch[-2:]]

        optimizer.zero_grad()

        V_obs_abs = obs_traj.permute(0, 2, 3, 1)
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_abs, V_obs_tmp)
        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            l[torch.isnan(l)] = 0
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['train_loss'].append(loss_batch / batch_count)

    if metrics['train_loss'][-1] < constant_metrics['min_train_loss']:
        constant_metrics['min_train_loss'] = metrics['train_loss'][-1]
        constant_metrics['min_train_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'train_best.pth')  # OK


@torch.no_grad()
def valid(epoch, model, checkpoint_dir, loader_val):
    global metrics, constant_metrics
    model.eval()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_val):
        batch_count += 1

        # Get data
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel = [tensor.cuda(non_blocking=True) for tensor in batch[:4]]
        V_obs, V_tr = [tensor.cuda(non_blocking=True) for tensor in batch[-2:]]

        V_obs_abs = obs_traj.permute(0, 2, 3, 1)
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_abs, V_obs_tmp)
        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            l[torch.isnan(l)] = 0
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            # Metrics
            loss_batch += loss.item()
            print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['val_loss'].append(loss_batch / batch_count)

    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(), checkpoint_dir + 'val_best.pth')  # OK


def main(args):
    obs_seq_len = args.obs_len
    pred_seq_len = args.pred_len
    data_set = './dataset/' + args.dataset + '/'

    dset_train = TrajectoryDataset(data_set + 'train/', obs_len=obs_seq_len, pred_len=pred_seq_len, skip=1)
    loader_train = DataLoader(dset_train, batch_size=1, shuffle=True, num_workers=0)
    dset_val = TrajectoryDataset(data_set + 'val/', obs_len=obs_seq_len, pred_len=pred_seq_len, skip=1)
    loader_val = DataLoader(dset_val, batch_size=1, shuffle=False, num_workers=0)

    print('Training started ...')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    base_model = TrajectoryModel(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                                 obs_len=8, pred_len=12, n_tcn=5, out_dims=5).cuda()
    model = GPGraph(baseline_model=base_model, in_channels=2, out_channels=5,
                    obs_seq_len=8, pred_seq_len=12,
                    d_type='learned_l2norm', d_th='learned', mix_type='mlp',
                    group_type=(True, True, True), weight_share=True).cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)

    if args.use_lrschd:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100], gamma=0.5)

    checkpoint_dir = './checkpoints/' + args.tag + '/' + args.dataset + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    print('Data and model loaded')
    print('Checkpoint dir:', checkpoint_dir)

    for epoch in range(args.num_epochs):
        train(epoch, model, optimizer, checkpoint_dir, loader_train)
        valid(epoch, model, checkpoint_dir, loader_val)

        if args.use_lrschd:
            scheduler.step()

        print('*' * 30)
        print('Epoch:', args.dataset + '/' + args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        print(constant_metrics)
        print('*' * 30)

        with open(checkpoint_dir + 'constant_metrics.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
