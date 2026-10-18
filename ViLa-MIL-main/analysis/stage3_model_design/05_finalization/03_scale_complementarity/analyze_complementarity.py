"""Pure descriptive Low/High pairing; fixed strata, no fusion/selector fitting."""
import numpy as np
import pandas as pd
from stage353_common import GROUPS, MARGIN_BINS, align, class_metrics, metrics_from_predictions


def divide(a, b):
    return float(a/b) if b else np.nan


def margin_stratum(margins):
    # Intervals [0,.10], (.10,.50], (.50,.90], (.90,1]; frozen before Low evaluation.
    return pd.cut(margins, bins=MARGIN_BINS, include_lowest=True,
                  labels=['0_to_0.10','0.10_to_0.50','0.50_to_0.90','0.90_to_1.00'])


def pair_cases(high, low, fold):
    for frame in (high, low):
        metrics_from_predictions(frame)
    h, l = align(high, low)
    hc = (h.prediction == h.label).to_numpy()
    lc = (l.prediction == l.label).to_numpy()
    cases = pd.DataFrame(dict(slide_id=h.slide_id, fold=fold, label=h.label,
        high_prediction=h.prediction, low_prediction=l.prediction,
        high_p0=h.p_0, high_p1=h.p_1, low_p0=l.p_0, low_p1=l.p_1,
        group=np.select([hc&lc,hc&~lc,~hc&lc],['G1','G2','G3'],default='G4'),
        high_correct=hc, low_correct=lc, prediction_agrees=h.prediction.eq(l.prediction),
        signed_probability_difference=l.p_1-h.p_1,
        absolute_probability_difference=np.abs(l.p_1-h.p_1)))
    for arm in ('high','low'):
        p0, p1 = cases[f'{arm}_p0'],cases[f'{arm}_p1']
        cases[f'{arm}_signed_margin'] = p1-p0
        cases[f'{arm}_absolute_margin'] = np.abs(p1-p0)
        cases[f'{arm}_confidence'] = np.maximum(p0,p1)
        cases[f'{arm}_true_class_margin'] = np.where(cases.label==1,p1-p0,p0-p1)
    cases['high_margin_stratum'] = margin_stratum(cases.high_absolute_margin)
    return cases


def summarize_cases(cases):
    fold = int(cases.fold.iloc[0]); n = len(cases)
    counts = {g:int(cases.group.eq(g).sum()) for g in GROUPS}
    hw, lw = counts['G3']+counts['G4'], counts['G2']+counts['G4']
    summary = dict(fold=fold, status='Complete', n_slides=n, **{g+'_count':v for g,v in counts.items()},
        **{g+'_fraction':v/n for g,v in counts.items()}, high_wrong_count=hw, low_wrong_count=lw,
        low_rescue_count=counts['G3'], high_error_rescue_rate=divide(counts['G3'],hw),
        low_interference_count=counts['G2'], high_correct_interference_rate=divide(counts['G2'],n-hw),
        potential_rescue_minus_interference=counts['G3']-counts['G2'],
        disagreement_count=counts['G2']+counts['G3'], prediction_agreement_rate=float(cases.prediction_agrees.mean()),
        error_intersection_count=counts['G4'], error_union_count=counts['G2']+counts['G3']+counts['G4'],
        error_jaccard=divide(counts['G4'],counts['G2']+counts['G3']+counts['G4']),
        oracle_accuracy_upper_bound=(n-counts['G4'])/n, oracle_accuracy_gain_over_high=counts['G3']/n,
        oracle_note='Label-dependent hard-classification upper bound; not a deployable model; no Oracle AUC reported')
    groups = []
    quantities = [c for c in cases if c.endswith(('_confidence','_absolute_margin','_true_class_margin'))]
    for g in GROUPS:
        part=cases[cases.group==g]
        groups.append(dict(fold=fold, group=g, n=len(part), fraction=len(part)/n,
            class0_count=int(part.label.eq(0).sum()), class1_count=int(part.label.eq(1).sum()),
            **{c+'_mean':float(part[c].mean()) for c in quantities},
            absolute_probability_difference_mean=float(part.absolute_probability_difference.mean())))
    strata=[]
    for stratum in ('0_to_0.10','0.10_to_0.50','0.50_to_0.90','0.90_to_1.00'):
        for label in ('all',0,1):
            part=cases[cases.high_margin_stratum==stratum]
            if label!='all': part=part[part.label==label]
            hwpart=part[~part.high_correct]
            hcpart=part[part.high_correct]
            strata.append(dict(fold=fold,high_margin_stratum=stratum,label=label,n=len(part),
                high_wrong_count=len(hwpart),G3_count=int(part.group.eq('G3').sum()),G2_count=int(part.group.eq('G2').sum()),
                low_correct_rate=divide(int(part.low_correct.sum()),len(part)),
                high_error_rescue_rate=divide(int(hwpart.low_correct.sum()),len(hwpart)),
                high_correct_interference_rate=divide(int((~hcpart.low_correct).sum()),len(hcpart))))
    return summary, pd.DataFrame(groups), pd.DataFrame(strata)


def model_metrics(high, low, fold):
    rows=[]
    for arm,frame in [('H',high),('L',low)]:
        rows.append(dict(fold=fold,arm=arm,status='Complete',split='val',n_slides=len(frame),
            checkpoint=frame.checkpoint.iloc[0] if 'checkpoint' in frame else 'Synthetic fixture',
            **metrics_from_predictions(frame),**class_metrics(frame)))
    return rows


def associate(cases, e0, e1, gate, residual, low_context):
    h, r = align(e0,e1)
    before = cases.sort_values('slide_id').reset_index(drop=True)
    if not before.slide_id.equals(h.slide_id) or not before.label.equals(h.label):
        raise ValueError('Stage353/Stage352 identity mismatch')
    if not np.array_equal(before[['high_p0','high_p1','high_prediction']].to_numpy(),h[['p_0','p_1','prediction']].to_numpy()):
        raise ValueError('High predictions must equal Stage352 E0')
    frame=before.copy()
    frame['routing_prediction']=r.prediction
    frame['routing_p1']=r.p_1
    frame['routing_correct']=r.prediction.eq(r.label)
    frame['routing_rescued']=~frame.high_correct & frame.routing_correct
    frame['routing_harmed']=frame.high_correct & ~frame.routing_correct
    frame['routing_absolute_probability_change']=np.abs(r.p_1-h.p_1)
    for stats in (gate,residual,low_context):
        if stats.slide_id.duplicated().any() or set(stats.slide_id)!=set(frame.slide_id):
            raise ValueError('Routing statistics duplicate/missing IDs')
        keys=[c for c in stats if c not in ('fold','arm','mode','seed','checkpoint','status','split')]
        frame=frame.merge(stats[keys],on='slide_id',validate='one_to_one')
    columns=['gate_mean','gate_variance','gate_near_one_fraction','between_parent_gate_mean_variance',
        'mapped_residual_ratio_mean','aggregate_residual_high_ratio','within_slide_cosine_mean',
        'low_feature_variance','low_feature_dispersion','no_usable_context_fraction']
    rows=[]
    for g in GROUPS:
        part=frame[frame.group==g]
        rows.append(dict(fold=int(cases.fold.iloc[0]),group=g,n=len(part),
            routing_rescued_count=int(part.routing_rescued.sum()),routing_harmed_count=int(part.routing_harmed.sum()),
            routing_correct_count=int(part.routing_correct.sum()),
            routing_rescued_fraction=divide(int(part.routing_rescued.sum()),len(part)),
            routing_harmed_fraction=divide(int(part.routing_harmed.sum()),len(part)),
            **{c+'_mean':float(part[c].mean()) for c in columns if c in part}))
    return frame,pd.DataFrame(rows)


def select_review_cases(cases):
    # All G3; class/fold matched G2 controls, stable original-ID ordering, no effects ranking.
    parts=[]
    for (fold,label),part in cases.groupby(['fold','label'],sort=True):
        g3=part[part.group=='G3'].sort_values('slide_id')
        g2=part[part.group=='G2'].sort_values('slide_id').head(max(1,len(g3)))
        parts.extend([g3.assign(selection='all_G3'),g2.assign(selection='G2_lexicographic_class_fold_control')])
    return pd.concat(parts,ignore_index=True)
