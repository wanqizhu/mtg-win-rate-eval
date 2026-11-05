"""Preserve compact audit artifacts and a noninteractive validation figure."""
import gzip,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;OUT=HERE/'artifacts'

def main():
    compressed=['historical_labels.csv','ensemble_cross_validation.csv','calibrated_sample_predictions.csv','weight_experiments.csv','features.csv','model_rows.csv','statistical_predictions.csv','initial_model/statistical_predictions.csv']
    for name in compressed:
        path=OUT/name
        Path(str(path)+'.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0))
    for name in ['calibration_sample.json','audit_sample.json']:
        rows=json.loads((OUT/name).read_text())
        (OUT/name).write_text(json.dumps([{'set':x['set'],'name':x['name']} for x in rows],ensure_ascii=False,indent=2)+'\n')
    (OUT/'validation_metrics_v1.csv').write_bytes((OUT/'initial_model/validation_metrics.csv').read_bytes())
    analogues=pd.read_csv(OUT/'historical_analogues.csv')
    analogues=analogues[(analogues['set']=='TRK')&(analogues['rank']<=5)]
    analogues.to_csv(OUT/'target_analogues.csv',index=False,float_format='%.6f')
    Path(str(OUT/'target_analogues.csv')+'.gz').write_bytes(gzip.compress((OUT/'target_analogues.csv').read_bytes(),mtime=0))
    d=pd.read_csv(OUT/'ensemble_cross_validation.csv');d=d[(d.gih_games>=500)&(d['format']=='PremierDraft')]
    s=pd.read_csv(OUT/'calibrated_sample_predictions.csv');s['blend']=.65*s.statistical+.35*s.qualitative_wr
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    ax=axs[0];ax.hexbin(d.truth,d.statistical,gridsize=35,mincnt=1,cmap='Blues');ax.plot([35,75],[35,75],c='#444444',lw=1,ls='--')
    ax.set(title='Whole-set statistical holdouts (n=2,298)',xlabel='Observed GiH win rate (%)',ylabel='Predicted GiH win rate (%)',xlim=(35,75),ylim=(35,75))
    ax=axs[1]
    for group,color in [('calibration','#9ca3af'),('audit','#7c3aed')]:
        q=s[s['sample']==group];ax.scatter(q.truth,q.blend,s=17,alpha=.7,label=f'{group} (n={len(q)})',color=color)
    ax.plot([35,75],[35,75],c='#444444',lw=1,ls='--');ax.set(title='35% qualitative / 65% statistical blend',xlabel='Observed GiH win rate (%)',ylabel='Predicted GiH win rate (%)',xlim=(35,75),ylim=(35,75));ax.legend(frameon=False)
    fig.savefig(OUT/'validation.png',dpi=140);plt.close(fig)
    include=[
      'source_hashes.json','historical_summary.json','historical_labels.csv.gz',
      'validation_metrics.csv','validation_metrics_v1.csv','ensemble_metrics.csv',
      'analogue_metrics.csv','ensemble_cross_validation.csv.gz',
      'calibration_report.json','qualitative_blend_metrics.csv',
      'calibrated_sample_predictions.csv.gz','calibration_sample.json','audit_sample.json',
      'calibration_cards.txt','audit_cards.txt','mechanic_scenarios.json',
      'prediction_components.csv','submission_manifest.json','checks.json','excluded_scope.csv',
      'weight_experiments.csv.gz','features.csv.gz','model_rows.csv.gz',
      'statistical_predictions.csv.gz','initial_model/statistical_predictions.csv.gz',
      'target_analogues.csv.gz','validation.png']
    lines=['# Generated caches are local; the explicit audit snapshot below is committed.','artifacts/*']
    for name in include:
        if '/' not in name:lines.append('!artifacts/'+name)
    lines.extend(['!artifacts/initial_model/','artifacts/initial_model/*','!artifacts/initial_model/statistical_predictions.csv.gz'])
    (HERE/'.gitignore').write_text('\n'.join(lines)+'\n')
    (OUT/'published_files.json').write_text(json.dumps(include,indent=2)+'\n')
    print('Published artifact bytes:',sum((OUT/x).stat().st_size for x in include))

if __name__=='__main__':main()
