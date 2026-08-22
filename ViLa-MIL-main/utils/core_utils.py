import numpy as np
import torch
from utils.utils import *
import os
from datasets.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.metrics import auc as calc_auc
from utils.loss_utils import FocalLoss
import time
from tqdm import tqdm


# For long CV runs: once BiomedCLIP is successfully loaded from HF Hub,
# force subsequent folds to use local cache only to avoid network/proxy/SSL flakiness.
_HF_OFFLINE_LOCKED = False


def _lock_hf_offline_for_remaining_folds():
    global _HF_OFFLINE_LOCKED
    if _HF_OFFLINE_LOCKED:
        return
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    _HF_OFFLINE_LOCKED = True
    print('[Info] HF Hub offline mode enabled for remaining folds (uses local cache only).')


class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, stop_epoch=0, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            stop_epoch (int): Earliest epoch (1-based) allowed to stop. Default: 0 (no restriction)
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        # Use negative val_loss so that larger score is better
        score = -val_loss
        epoch_num = epoch + 1  # convert to 1-based for readability

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score <= self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience} (min_epoch={self.stop_epoch}, epoch={epoch_num})')
            # stop when patience reached and epoch passes the minimum epoch constraint
            if self.counter >= self.patience and epoch_num >= self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def train(datasets, cur, args):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'focal':
        loss_fn = FocalLoss().cuda()
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}

    if args.model_type == 'ViLa_MIL_BiomedCLIP':
        # BiomedCLIP版本ViLa-MIL
        print('🔬 Using BiomedCLIP-based ViLa-MIL')
        import ml_collections
        from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
        
        config = ml_collections.ConfigDict()
        config.input_size = 512  # BiomedCLIP特征维度
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.scale_mode = getattr(args, 'scale_mode', 'dual')
        # Control whether BiomedCLIP text encoder is finetuned (default: frozen)
        config.finetune_text_encoder = bool(getattr(args, 'finetune_text_encoder', False))
        # Finetune scope for BiomedCLIP text tower
        config.text_finetune_mode = str(getattr(args, 'text_finetune_mode', 'proj'))
        config.text_unfreeze_last_n = int(getattr(args, 'text_unfreeze_last_n', 2))
        
        model = ViLa_MIL_BiomedCLIP(config=config, num_classes=args.n_classes)
    
    elif args.model_type == 'ViLa_MIL':
        # 原始CLIP版本
        import ml_collections
        from models.model_ViLa_MIL import ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        config.input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        model_dict = {'config': config, 'num_classes':args.n_classes}
        model = ViLa_MIL_Model(**model_dict)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)


    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda:0'))
    print('Done!')
    print_network(model)

    # After BiomedCLIP loads successfully once, lock HF Hub into offline mode so later folds
    # won't fail due to transient SSL/proxy/network issues.
    if args.model_type == 'ViLa_MIL_BiomedCLIP':
        if os.environ.get('HF_HUB_OFFLINE', '0') != '1':
            _lock_hf_offline_for_remaining_folds()

    if args.model_type == 'ViLa_MIL_BiomedCLIP':
        finetune_flag = bool(getattr(args, 'finetune_text_encoder', False))
        text_trainable = 0
        prompt_trainable = 0
        if hasattr(model, 'text_encoder'):
            text_trainable = sum(p.numel() for p in model.text_encoder.parameters() if p.requires_grad)
        if hasattr(model, 'prompt_learner'):
            prompt_trainable = sum(p.numel() for p in model.prompt_learner.parameters() if p.requires_grad)

        # Derive default per-group LRs (used by get_optim) for visibility.
        base_lr = float(getattr(args, 'lr', 1e-4))
        prompt_lr = getattr(args, 'prompt_lr', None)
        text_lr = getattr(args, 'text_lr', None)
        if prompt_lr is None:
            prompt_lr = min(max(base_lr * 10.0, 1e-4), 1e-3)
        if text_lr is None:
            text_lr = min(base_lr, 1e-5)

        print(
            f"[Debug] finetune_text_encoder={finetune_flag} | "
            f"text_encoder_trainable_params={text_trainable} | "
            f"prompt_learner_trainable_params={prompt_trainable} | "
            f"text_finetune_mode={getattr(args, 'text_finetune_mode', 'proj')} | "
            f"text_unfreeze_last_n={getattr(args, 'text_unfreeze_last_n', 2)} | "
            f"lr(base)={base_lr:g} lr(prompt)={float(prompt_lr):g} lr(text)={float(text_lr):g}"
        )

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    if args.model_type == 'ViLa_MIL_BiomedCLIP':
        try:
            groups = getattr(optimizer, 'param_groups', [])
            if groups:
                parts = []
                for g in groups:
                    name = g.get('name', 'group')
                    lr = g.get('lr', None)
                    n = sum(p.numel() for p in g.get('params', []) if hasattr(p, 'numel'))
                    parts.append(f"{name}: lr={lr:g} params={n}")
                print('[Debug] optimizer param_groups -> ' + ' | '.join(parts))
        except Exception:
            pass

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True)

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample, mode=args.mode)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode)
    test_loader = get_split_loader(test_split, testing = args.testing, mode=args.mode)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        # 固定使用 patience=10，且不限制最小停止轮数（达到 patience 即可停止）
        early_stopping = EarlyStopping(patience=10, stop_epoch=0, verbose=True)
    else:
        early_stopping = None
    print('Done!')

    # 存储每个epoch的详细信息
    epoch_details = []
    
    for epoch in range(args.max_epochs):
        epoch_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"🔄 FOLD {cur+1} - EPOCH {epoch+1}/{args.max_epochs}")
        print(f"{'='*80}")
        
        # 训练阶段
        train_loss, train_error, train_auc, train_f1 = train_loop(args, epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn, cur)
        
        # 验证阶段
        val_loss, val_error, val_auc, val_f1, stop = validate(cur, epoch, model, val_loader, args.n_classes, 
            early_stopping, writer, loss_fn, args.results_dir)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # 记录epoch详情
        epoch_info = {
            'fold': cur + 1,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': 1 - train_error,
            'train_auc': train_auc,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': 1 - val_error,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'duration_seconds': round(epoch_duration, 2)
        }
        epoch_details.append(epoch_info)
        
        print(f"📈 Epoch {epoch+1} Summary: Train_AUC={train_auc:.4f}, Val_AUC={val_auc:.4f}, Duration={epoch_duration:.2f}s")
        
        if stop: 
            print(f"⏹️  Early stopping at epoch {epoch+1}")
            break

    if args.early_stopping: 
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    _, val_error, val_auc, _, val_f1 = summary(args.mode, model, val_loader, args.n_classes)
    print(f'🎯 FOLD {cur+1} Final Validation: Error={val_error:.4f}, AUC={val_auc:.4f}, F1={val_f1:.4f}')

    results_dict, test_error, test_auc, acc_logger, test_f1 = summary(args.mode, model, test_loader, args.n_classes)
    print(f'🎯 FOLD {cur+1} Final Test: Error={test_error:.4f}, AUC={test_auc:.4f}, F1={test_f1:.4f}')

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)
        print(f'   Class {i}: acc {acc:.4f}, correct {correct}/{count}')

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()
        
    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error, each_class_acc, test_f1, epoch_details


def train_loop(args, epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None, fold=0):

    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    running_correct = 0
    running_total = 0

    all_prob = []
    all_label = []
    all_pred = []

    print(f'\n🚀 [FOLD {fold+1}] [EPOCH {epoch+1}] Training...')
    pbar = tqdm(enumerate(loader), total=len(loader), ncols=100, desc=f'Train F{fold+1}E{epoch+1}')
    for batch_idx, (data_s, coord_s, data_l, coords_l, label) in pbar:
        data_s, coord_s, data_l, coords_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device)
        
        Y_prob, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        acc_logger.log(Y_hat, label)
        
        train_loss += loss.item()
        batch_error = calculate_error(Y_hat, label)
        train_error += batch_error

        # running accuracy for tqdm display
        # label/Y_hat are scalar per-bag in this pipeline, but keep it generic
        batch_correct = (Y_hat == label).float().sum().item()
        batch_total = float(label.numel())
        running_correct += batch_correct
        running_total += batch_total
        
        all_prob.append(Y_prob.detach().cpu().numpy())
        all_label.append(label.cpu().numpy())
        all_pred.append(Y_hat.cpu().numpy())

        # 更新进度条简要信息
        if (batch_idx + 1) % 10 == 0:
            running_acc = (running_correct / running_total) if running_total > 0 else 0.0
            pbar.set_postfix(
                loss=f"{train_loss/(batch_idx+1):.4f}",
                acc=f"{running_acc:.4f}",
            )

    train_loss /= len(loader)
    train_error /= len(loader)

    all_prob = np.concatenate(all_prob, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)

    if n_classes == 2:
        train_auc = roc_auc_score(all_label, all_prob[:, 1])
    else:
        train_auc = roc_auc_score(all_label, all_prob, multi_class='ovr')
    
    train_f1 = f1_score(all_label, all_pred, average='macro', zero_division=0)

    print(f'📈 [FOLD {fold+1}] [EPOCH {epoch+1}] Train: Loss={train_loss:.4f}, Acc={1-train_error:.4f}, AUC={train_auc:.4f}, F1={train_f1:.4f}')
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'   Class {i}: acc {acc:.4f}, correct {correct}/{count}')
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/auc', train_auc, epoch)
        writer.add_scalar('train/f1', train_f1, epoch)

    return train_loss, train_error, train_auc, train_f1
   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    with torch.no_grad():
        pbar = tqdm(enumerate(loader), total=len(loader), ncols=100, desc=f'Val   F{cur+1}E{epoch+1}')
        for batch_idx, (data_s, coord_s, data_l, coords_l, label) in pbar:
            data_s, coord_s, data_l, coords_l, label = data_s.to(device, non_blocking=True), coord_s.to(device, non_blocking=True), \
                                                                  data_l.to(device, non_blocking=True), coords_l.to(device, non_blocking=True), \
                                                                  label.to(device, non_blocking=True)
            Y_prob, Y_hat, loss = model(data_s, coord_s, data_l, coords_l, label)

            acc_logger.log(Y_hat, label)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error

            if (batch_idx + 1) % 10 == 0:
                pbar.set_postfix(loss=f"{val_loss/(batch_idx+1):.4f}")

    val_error /= len(loader)
    val_loss /= len(loader)

    # 使用收集好的概率矩阵与标签计算预测与 F1
    preds = np.argmax(prob, axis=1)
    val_f1 = f1_score(labels, preds, average='macro', zero_division=0)

    # 兼容验证集中只有单一类别的情况
    if len(np.unique(labels)) <= 1:
        auc = -1
    else:
        if n_classes == 2:
            auc = roc_auc_score(labels, prob[:, 1])
        else:
            auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/f1', val_f1, epoch)

    print(f'📊 [FOLD {cur+1}] [EPOCH {epoch+1}] Val: Loss={val_loss:.4f}, Acc={1-val_error:.4f}, AUC={auc:.4f}, F1={val_f1:.4f}')
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print(f'   Class {i}: acc {acc:.4f}, correct {correct}/{count}')

    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_error, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("🛑 Early stopping triggered!")
            return val_loss, val_error, auc, val_f1, True

    return val_loss, val_error, auc, val_f1, False

def summary(mode, model, loader, n_classes):
    start_time = time.time() # 计时开始
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_s, coord_s, data_l, coords_l, label) in enumerate(loader):
        data_s, coord_s, data_l, coords_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coords_l.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            Y_prob, Y_hat, _ = model(data_s, coord_s, data_l, coords_l, label)

        acc_logger.log(Y_hat, label)
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)
    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1
        
    else:
        if n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
            for class_idx in range(n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(calc_auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            auc_score = np.nanmean(np.array(aucs))

    all_pred = np.argmax(all_probs, axis=1)
    f1 = f1_score(all_labels, all_pred, average='macro', zero_division=0)
    
    end_time = time.time() # 计时结束
    duration = end_time - start_time
    print(f'   -> 评估耗时: {duration:.2f} 秒 ({duration/60:.2f} 分钟)')

    return patient_results, test_error, auc_score, acc_logger, f1
