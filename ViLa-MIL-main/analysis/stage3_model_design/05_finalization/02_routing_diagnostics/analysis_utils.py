"""Paired validation changes, class metrics and descriptive spatial diagnostics."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from bridge import NEAR_MARGIN, ROOT
from tools import stage336_context_shuffle_stability as s336


def align(left,right):
    if left.slide_id.duplicated().any() or right.slide_id.duplicated().any():
        raise ValueError('Duplicate Slide ID')
    a,b = left.sort_values('slide_id').reset_index(drop=True),right.sort_values('slide_id').reset_index(drop=True)
    if not a.slide_id.equals(b.slide_id) or not a.label.equals(b.label):
        raise ValueError('Paired Slide IDs or labels differ')
    return a,b


def old_columns(frame):
    return frame.rename(columns={'label':'Y','prediction':'Y_hat'})


def paired_changes(left,right):
    a,b = align(left,right)
    # Reuse Stage336 performance/probability pairing; enforce labels above (old helper does not).
    stats = s336._paired_metrics(old_columns(a),old_columns(b),2)
    ca,cb = (a.prediction==a.label).to_numpy(),(b.prediction==b.label).to_numpy()
    stats.update(correct_to_wrong=int((ca&~cb).sum()),wrong_to_correct=int((~ca&cb).sum()),
        mean_signed_margin_change=float(((b.p_1-b.p_0)-(a.p_1-a.p_0)).mean()),
        mean_absolute_margin_change=float(np.abs((b.p_1-b.p_0)-(a.p_1-a.p_0)).mean()),
        mean_confidence_margin_change=float((np.abs(b.p_1-b.p_0)-np.abs(a.p_1-a.p_0)).mean()),
        mean_true_class_margin_change=float((np.where(b.label==1,b.p_1-b.p_0,b.p_0-b.p_1)
                                           -np.where(a.label==1,a.p_1-a.p_0,a.p_0-a.p_1)).mean()))
    frame = pd.DataFrame({'slide_id':a.slide_id,'label':a.label,
        'prediction_before':a.prediction,'prediction_after':b.prediction,
        'p0_before':a.p_0,'p1_before':a.p_1,'p0_after':b.p_0,'p1_after':b.p_1,
        'margin_before':a.p_1-a.p_0,'margin_after':b.p_1-b.p_0,
        'absolute_probability_change':np.abs(b.p_1-a.p_1),
        'near_boundary_before':np.abs(a.p_1-a.p_0)<=NEAR_MARGIN,
        'correctness_group':np.select([ca&cb,ca&~cb,~ca&cb],['both_correct','harmed','rescued'],default='both_wrong')})
    return stats,frame


def class_metrics(frame):
    tn,fp,fn,tp = confusion_matrix(frame.label,frame.prediction,labels=[0,1]).ravel()
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        auc_positive_class=1,auc_positive_name='NonAdenocarcinoma',
        sensitivity_class1=float(tp/(tp+fn)),specificity_class1=float(tn/(tn+fp)),
        recall_adenocarcinoma=float(tn/(tn+fp)),recall_nonadenocarcinoma=float(tp/(tp+fn)),
        predicted_class0=int((frame.prediction==0).sum()),predicted_class1=int((frame.prediction==1).sum()),
        probability_class1_mean=float(frame.p_1.mean()),absolute_margin_mean=float(np.abs(frame.p_1-frame.p_0).mean()))


def ranking_changes(left,right):
    a,b = align(left,right)
    pos,neg = a.label.to_numpy()==1,a.label.to_numpy()==0
    x = np.sign(a.p_1.to_numpy()[pos,None]-a.p_1.to_numpy()[None,neg])
    y = np.sign(b.p_1.to_numpy()[pos,None]-b.p_1.to_numpy()[None,neg])
    credit_x,credit_y = (x+1)/2,(y+1)/2
    return dict(n_positive_negative_pairs=int(x.size),
                pairs_order_credit_improved=int((credit_y>credit_x).sum()),
                pairs_order_credit_worsened=int((credit_y<credit_x).sum()),
                delta_pair_auc=float((credit_y-credit_x).mean()),
                ties_before=int((x==0).sum()),ties_after=int((y==0).sum()))


def group_summary(changes,gate=None,residual=None):
    frame = changes.copy()
    for statistics in (gate,residual):
        if statistics is not None and not statistics.empty:
            cols = [c for c in statistics if c not in ('arm','mode','seed','checkpoint','status','split','fold')]
            frame = frame.merge(statistics[cols],on='slide_id',how='left',validate='one_to_one')
    rows = []
    quantities = ('absolute_probability_change','margin_before','margin_after','gate_mean','gate_variance',
                  'gate_near_one_fraction','between_parent_gate_mean_variance',
                  'mapped_residual_ratio_mean','aggregate_residual_high_ratio','no_usable_context_fraction')
    for group in ('both_correct','harmed','rescued','both_wrong'):
        part = frame[frame.correctness_group==group]
        row = dict(group=group,n=len(part),fraction=len(part)/len(frame),
                   class0_count=int((part.label==0).sum()),class1_count=int((part.label==1).sum()),
                   near_boundary_count=int(part.near_boundary_before.sum()),
                   near_boundary_fraction=float(part.near_boundary_before.mean()) if len(part) else np.nan)
        for key in quantities:
            if key in part:
                row[f'{key}_mean'] = float(part[key].mean()) if len(part) else np.nan
        rows.append(row)
    return frame,pd.DataFrame(rows)
