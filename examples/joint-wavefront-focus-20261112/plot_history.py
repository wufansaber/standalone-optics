"""Render only a completed, sanitized checkpoint history."""
import argparse,json
from pathlib import Path

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    report=json.loads(args.report.read_text(encoding='utf-8'))
    if report.get('final_all_declared_goals_passed')is not True:
        raise ValueError('This figure requires a completed final case')
    if args.out.exists():raise FileExistsError('Fresh output required')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=report['history'];x=[r['checkpoint_index']for r in history]
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for key,color in [('WFE','#0072B2'),('LCA','#D55E00'),('FC','#009E73')]:
        ax.plot(x,[r['normalized'][key]for r in history],marker='o',markersize=4,
                linewidth=1.8,color=color,label=f'{key} / limit')
    ax.axhline(1.,color='#555555',linestyle='--',linewidth=1.,label='Unchanged acceptance limit')
    ax.set(title='Complete checkpoint history',xlabel='Verified checkpoint index',
           ylabel='Metric divided by its original limit')
    ax.set_xticks(x);ax.grid(alpha=.2);ax.legend(loc='best',frameon=False)
    fig.savefig(args.out,dpi=160,metadata={'Description':'Dimensionless verified tail of a private design continuation. No prescription or catalog data.'})
    plt.close(fig)

if __name__=='__main__':main()
