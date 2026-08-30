import numpy as np
import torch
from models.model_mil import MIL_fc, MIL_fc_mc
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize


def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type == 'ViLa_MIL':
        import ml_collections
        from models.model_ViLa_MIL import ViLa_MIL_Model
        config = ml_collections.ConfigDict()
        config.input_size = 1024
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        model_dict = {'config': config, 'num_classes':args.n_classes}
        model = ViLa_MIL_Model(**model_dict)

    elif args.model_type == 'ViLa_MIL_BiomedCLIP':
        import ml_collections
        from models.model_ViLa_MIL_BiomedCLIP import ViLa_MIL_BiomedCLIP
        config = ml_collections.ConfigDict()
        config.input_size = 512
        config.hidden_size = 192
        config.text_prompt = args.text_prompt
        config.prototype_number = args.prototype_number
        config.scale_mode = getattr(args, 'scale_mode', 'dual')
        config.finetune_text_encoder = bool(getattr(args, 'finetune_text_encoder', False))
        config.use_global_proto_context = bool(getattr(args, 'use_global_proto_context', False))
        model = ViLa_MIL_BiomedCLIP(config=config, num_classes=args.n_classes)

    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    try:
        # Prefer weights_only for security (PyTorch 2.2+ supports this kwarg)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        print('[Info] torch.load using weights_only=True')
    except TypeError:
        # Older PyTorch versions don't support weights_only
        ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    if bool(getattr(args, 'use_global_proto_context', False)):
        missing, unexpected = model.load_state_dict(ckpt_clean, strict=False)
        allowed = {
            'global_proto_context_norm.weight',
            'global_proto_context_norm.bias',
            'global_proto_context_projection.weight',
            'global_proto_context_projection.bias',
            'global_proto_context_gamma',
        }
        unexpected = set(unexpected)
        missing = set(missing)
        if unexpected or not missing.issubset(allowed):
            raise RuntimeError(
                f'Incompatible checkpoint for global prototype conditioning: '
                f'missing={sorted(missing)}, unexpected={sorted(unexpected)}'
            )
        print(f'[Info] Legacy checkpoint loaded; initialized {len(missing)} Stage 3.3.8 parameters.')
    else:
        model.load_state_dict(ckpt_clean, strict=True)

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
        # pass
    model.eval()
    return model

def eval(mode, dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset, mode=args.mode)
    patient_results, test_error, auc, test_f1, df, acc_logger = summary(mode, model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    print('f1: ', test_f1)

    each_class_acc = []
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        each_class_acc.append(acc)

    return model, patient_results, test_error, auc, test_f1, df, each_class_acc

def summary(mode, model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.
    test_f1 = 0.

    if len(loader) == 0:
        raise ValueError(
            "Empty evaluation loader. For external datasets, use --split all so the full CSV is evaluated "
            "instead of filtering by the training split file."
        )

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    all_pred = []
    all_label = []

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    if(mode == 'transformer'):
        for batch_idx, (data_s, coord_s, data_l, coord_l, label) in enumerate(loader):
            data_s, coord_s, data_l, coord_l, label = data_s.to(device), coord_s.to(device), data_l.to(device), coord_l.to(device), label.to(device)
            slide_id = slide_ids.iloc[batch_idx]
            with torch.no_grad():
                Y_prob, Y_hat, loss = model(data_s, coord_s, data_l, coord_l, label)

            acc_logger.log(Y_hat, label)
            probs = Y_prob.cpu().numpy()
            all_probs[batch_idx] = probs
            all_labels[batch_idx] = label.item()
            all_preds[batch_idx] = Y_hat.item()
            all_pred.append(Y_hat.cpu().numpy())
            all_label.append(label.cpu().numpy())
            patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
            error = calculate_error(Y_hat, label)
            test_error += error

        if len(all_pred) == 0 or len(all_label) == 0:
            raise ValueError(
                "No predictions were produced during evaluation. "
                "Check whether the split is empty or whether feature files match slide_id values."
            )

        # 将列表转换为numpy数组并展平
        all_pred_np = np.concatenate(all_pred)
        all_label_np = np.concatenate(all_label)
        test_f1 = f1_score(all_label_np, all_pred_np, average='macro')
        test_error /= len(loader)

        aucs = []
        if len(np.unique(all_labels)) == 1:
            auc_score = -1

        else:
            if args.n_classes == 2:
                auc_score = roc_auc_score(all_labels, all_probs[:, 1])
            else:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                for class_idx in range(args.n_classes):
                    if class_idx in all_labels:
                        fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                        aucs.append(auc(fpr, tpr))
                    else:
                        aucs.append(float('nan'))
                if args.micro_average:
                    binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                    fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                    auc_score = auc(fpr, tpr)
                else:
                    auc_score = np.nanmean(np.array(aucs))

        results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
        for c in range(args.n_classes):
            results_dict.update({'p_{}'.format(c): all_probs[:,c]})
        df = pd.DataFrame(results_dict)
        return patient_results, test_error, auc_score, test_f1, df, acc_logger 
