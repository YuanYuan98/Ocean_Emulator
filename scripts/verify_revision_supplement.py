"""Check supplemental outputs against independently reduced original diagnostics."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient
from supplement_revision_paper import paper_timeseries, paper_table_figure, plt, REGIONS, LAYERS

root=Path(sys.argv[1]).resolve()
target=root/'paper_supplement'
keys=['samudra2','adaloss','frozen_final','tendency','neptune']
# Finalize the two cached series figures after the label-collision QA repair.
old={key:np.load(root/key/'diagnostics.npz') for key in keys}
plt.rcParams.update({'font.size':11,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False})
for var in ['thetao','so']:
    fig=paper_timeseries(var,old)
    fig.savefig(target/'figures'/f'{var}_paper_timeseries.png',dpi=190,bbox_inches='tight')
    fig.savefig(target/'figures'/f'{var}_paper_timeseries.pdf',bbox_inches='tight')
    plt.close(fig)
checks=[]
for key in keys:
    a=np.load(target/key/'fields.npz'); b=np.load(root/key/'diagnostics.npz')
    assert str(a['date']).startswith('2022-09-30')
    for level in [0,10,14,15]:
        for who in ['truth','pred']:
            np.testing.assert_allclose(a[f'thetao_{who}_anom_{level}'],b[f'thetao_snapshot_{who}_{level}'],equal_nan=True,rtol=1e-10,atol=1e-12)
    for var in ['thetao','so']:
        delta=a[f'{var}_pred_zonal_mean']-a[f'{var}_truth_zonal_mean']
        expected=np.stack([b[f'{var}_zonal_{i}'] for i in range(19)])
        np.testing.assert_allclose(delta,expected,equal_nan=True,atol=1e-10)
    checks.append({'model':key,'anomaly_snapshot_matches_previous':True,'zonal_bias_matches_previous':True,'reduction_checks':json.loads((target/key/'complete.json').read_text())['checks']})
tables=pd.read_csv(target/'paper_table_values.csv')
original=pd.read_csv(root/'all_metrics.csv')
# Final table layout: preserve all labels without collisions at document width.
for var,paper in [('thetao',1),('so',3),('ke',4)]:
    for layer,(_,_,label) in enumerate(LAYERS):
        rows=[];winners=[]
        for region in REGIONS:
            for metric,title in [('variance_corr','Var Corr ↑'),('variance_rmse','Var RMSE ↓'),('detrend_rmse','Detrend RMSE ↓'),('direct_rmse','Direct RMSE ↓')]:
                sub=original[(original.variable==var)&(original.layer==label)&(original.region==region)].set_index('model').reindex(keys)
                values=sub[metric].to_numpy();best=np.max(values) if metric=='variance_corr' else np.min(values)
                winners.append(np.flatnonzero(values==best))
                rows.append([region,title]+[f'{v:#.3g}' for v in values])
        fig=paper_table_figure(rows,winners,var,label,paper)
        fig.savefig(target/'figures'/f'table_{var}_{layer}.png',dpi=190,bbox_inches='tight')
        fig.savefig(target/'figures'/f'table_{var}_{layer}.pdf',bbox_inches='tight')
        plt.close(fig)
assert len(tables)==1260
for _,r in tables.iterrows():
    match=original[(original.model==r.model)&(original.variable==r.variable)&(original.layer==r.layer)&(original.region==r.region)]
    assert len(match)==1
    np.testing.assert_allclose(r.value,match.iloc[0][r.metric],rtol=1e-12)
manifest=json.loads((target/'manifest.json').read_text())
assert len(manifest)==25
assert sum(Path(f['file']).name.startswith('table_') for f in manifest)==9
assert all(Path(f['file']).exists() for f in manifest)
a=np.load(target/'samudra2/fields.npz')
t=a['thetao_truth_zonal_anom_series'];p=a['thetao_pred_zonal_anom_series']
good=np.isfinite(t)&np.isfinite(p)
paper_check={'zonal_temperature_anomaly_rmse':float(np.sqrt(np.mean((t[good]-p[good])**2))),
             'zonal_temperature_anomaly_corr':float(np.corrcoef(t[good],p[good])[0,1])}
np.testing.assert_allclose(paper_check['zonal_temperature_anomaly_rmse'],.085,atol=.0005)
np.testing.assert_allclose(paper_check['zonal_temperature_anomaly_corr'],.70,atol=.005)
summary={'status':'passed','real_data_checks':checks,'paper_figure_4_reference_check':paper_check,'table_values_checked':1260,'new_figures':16,'new_tables':9,'all_files_present':True}
(target/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
nb=nbformat.v4.new_notebook()
nb.cells=[nbformat.v4.new_markdown_cell('# Samudra 2 paper-aligned supplement\n\nFigures 4–7 and Tables 1/3/4: five-model OM4 comparison.\nSee figure captions for raw/anomaly, masking, depth and EKE conventions.'),nbformat.v4.new_code_cell('from pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\nroot = Path('+repr(str(target))+')\nvalues = pd.read_csv(root / "paper_table_values.csv")\ndisplay(values)')]
for f in manifest:
    nb.cells.extend([nbformat.v4.new_markdown_cell('## '+f['title']+'\n\n'+f['caption']),nbformat.v4.new_code_cell('display(Image(filename='+repr(f['file'])+'))')])
NotebookClient(nb,timeout=600,kernel_name='python3').execute()
nbformat.write(nb,target/'paper_supplement.ipynb')
print(json.dumps(summary),flush=True)
