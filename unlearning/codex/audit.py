"""Sensitivity audit, NOT a leaderboard estimator.

Applies the published score exactly to synthetic single-model gold scenarios.
Actual gold CE moments and gold probabilities are unavailable, and ensemble
gold moments cannot in general be reconstructed from average T2 probabilities.
"""
import argparse
from common import *


def moments(z,y):
    loss=F.cross_entropy(z,y,reduction='none')
    return torch.stack([loss.mean(),loss.std()])


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--names',nargs='+',required=True); a=ap.parse_args()
    torch.set_num_threads(2)
    T=load(ROOT/'orehn/cache/teacher.pt')
    original=load(ROOT/'orehn/cache/orig_logits_views.pt')
    rows=[]
    for name in a.names:
        p=OUT/'logs'/f'{name}_logits.pt'
        if not p.exists(): continue
        Z=load(p); idx=Z['indices']; y=Z['y']
        ctl_kl=kl((Z['control_original']/2).softmax(-1),Z['control']).mean().item()
        ctl_agree=(Z['control_original'].argmax(-1)==Z['control'].argmax(-1)).float().mean().item()
        # This is agreement, NOT true accuracy ratio. Include only as explicit proxy.
        retain_proxy=ctl_agree/(1+ctl_kl)
        scenarios=[]
        for A in [.3,.45,.6,.7,.8,.9]:
            for alpha in [1.5,2.5,4.]:
                q=helpers.make_targets(T,A,alpha,verbose=False)[idx]
                g=2*q.clamp_min(1e-12).log()
                m=[]; pr=[]
                for group in ['clean','bright']:
                    orig=original[group][idx]
                    mg=moments(g,y)
                    do=(moments(orig,y)-mg).abs().sum().item()
                    dc=(moments(Z[group],y)-mg).abs().sum().item()
                    ko=kl(q,orig).mean().item(); kc=kl(q,Z[group]).mean().item()
                    m.append(max(0,1-dc/max(do,1e-12)))
                    pr.append(max(0,1-kc/max(ko,1e-12)))
                M=sum(m)/2; P=sum(pr)/2
                scenarios.append({'A':A,'alpha':alpha,'moments':M,'predictions':P,
                                  'score_proxy':M*P*retain_proxy,'moments_groups':m,'predictions_groups':pr})
        row={'name':name,'retain_agreement_proxy':retain_proxy,'scenarios':scenarios,
             'range_synthetic_scores':[min(s['score_proxy'] for s in scenarios),max(s['score_proxy'] for s in scenarios)]}
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='scenarios'}),flush=True)
    report={'warning':'Synthetic scenarios using existing teacher family only. NOT gold, NOT an official score prediction. Same surrogate q on clean and bright intentionally simplifies unknown gold behavior.',
            'models':rows}
    (OUT/'logs/sensitivity.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__': main()
