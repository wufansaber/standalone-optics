"""Recheck the whole frozen material shortlist against the owner's private index."""
import argparse,gzip,json,hashlib,sys
from pathlib import Path
import numpy as np
CASE=Path(__file__).resolve().parent;sys.path[:0]=[str(CASE),str(CASE.parents[1])]
from problem import BASE,PINS
from scripts.ai_zemax.standalone_glass_catalog import GlassCatalogIndex
from scripts.ai_zemax.standalone_glass_search import freeze_combination


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():raise ValueError('Use a new report path')
    data=json.loads(gzip.decompress((CASE/'GLASS_BRANCHES.json.gz').read_bytes()));index=GlassCatalogIndex(args.index)
    records=index.search(vendor='OHARA',current_only=True,selection_eligible_only=True)
    if any(r['snapshot_sha256']!=PINS['catalog_sha256']for r in records):raise ValueError('Current private catalog differs from frozen case')
    if len(records)!=data['scan_current_eligible']:raise ValueError('Allowed material list changed')
    spectra={};rejected=[];waves=BASE['wavelengths_um'];primary=waves.index(.595)
    for r in records:
        try:spectra[r['name']]=freeze_combination(index,[{'vendor':'OHARA','name':r['name']}],waves)['materials'][0]
        except ValueError as e:rejected.append(dict(name=r['name'],reason=str(e)))
    if len(spectra)!=data['scan_calculable']:raise ValueError('Calculable allowed spectra changed')
    short=[];sites=[]
    for site in data['sites']:
        n=np.asarray(BASE['indices_by_wave'])[:,site['medium']];shape=(n-1)/(n[primary]-1);rank=[]
        for name,material in spectra.items():
            if name==site['current']:continue
            m=np.asarray(material['indices_by_wave']);score=np.sqrt(np.mean(((m-1)/(m[primary]-1)-shape)**2))+.05*abs(m[primary]-n[primary])/(n[primary]-1)
            rank.append((float(score),name))
        rank.sort();chosen=[rank[i]for i in (0,1,5)]
        sites.append(dict(site=site['site'],ranked=len(rank),closest=[dict(name=name,score=score)for score,name in rank[:8]]))
        for score,name in chosen:
            short.append(dict(id=site['site']+'-'+name,name=name,medium=site['medium'],family=site['site'],spectral_distance=score))
            expected=data['materials'][name]
            if not np.array_equal(spectra[name]['indices_by_wave'],expected['indices_by_wave']):raise ValueError('Frozen shortlisted exact dispersion differs')
    if [r['id']for r in short]!=[r['id']for r in data['branches']]:raise ValueError('Full-spectrum shortlist differs')
    for a,b in zip(short,data['branches']):
        if abs(a['spectral_distance']-b['spectral_distance'])>1e-14:raise ValueError('Shortlist score differs')
    result=dict(passed=True,allowed=len(records),calculable=len(spectra),wavelengths=len(waves),sites=sites,shortlist=short,rejected=rejected,
        catalog_sha256=PINS['catalog_sha256'],procurement_verified=False,full_catalog_published=False)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:result[k]for k in ('passed','allowed','calculable','wavelengths')}))

if __name__=='__main__':main()
