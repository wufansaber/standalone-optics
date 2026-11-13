"""Equivalent sampled peak-to-valley feasibility with linear-size bounds."""
from copy import deepcopy
import math
import numpy as np

from .standalone_merit_graph import normalize
from .standalone_local_solver import EfficientSolveMixin


def span_epigraph(values, lower, upper, limit):
    """Slacks for L <= f_i <= U and U-L <= limit; all lengths same units."""
    f = np.asarray(values, dtype=float)
    if f.ndim != 1 or not len(f) or not np.isfinite(f).all() or not all(map(math.isfinite,[lower,upper,limit])) or limit < 0.:
        raise ValueError('Require finite focus values/bounds and a nonnegative span limit')
    return np.r_[f-lower, upper-f, limit-(upper-lower)]


def span_epigraph_jacobian(value_jacobian):
    j = np.asarray(value_jacobian, dtype=float)
    if j.ndim != 2 or not len(j) or not np.isfinite(j).all():
        raise ValueError('Require a finite values-by-design-parameters Jacobian')
    n, p = j.shape
    result = np.zeros((2*n+1,p+2))
    result[:n,:p]=j; result[:n,p]=-1.
    result[n:2*n,:p]=-j; result[n:2*n,p+1]=1.
    result[-1,p]=1.; result[-1,p+1]=-1.
    return result


def append_native_span(rows, value_rows, limit, *, weight, label='sampled_PV'):
    """Append contiguous identity proxies, MAXX/MINN, difference and OPLT.

    No existing constraint is removed here. Callers explicitly deactivate the
    proven pairwise bounds and compile the new root's dependency closure.
    The sampled feasible set is equivalent; the weighted penalty is different.
    """
    result=normalize(rows); refs=list(value_rows)
    if (not refs or len(set(refs))!=len(refs) or any(type(i)is not int or not 1<=i<=len(result) for i in refs)
        or not math.isfinite(limit) or limit<0. or not math.isfinite(weight) or weight<=0.):
        raise ValueError('Require unique valid value rows, finite span limit and positive weight')
    def add(op,p,target=0.,w=0.):
        result.append(dict(operand=op,params=list(p)+[0.]*(8-len(p)),target=float(target),weight=float(w),label=label))
        return len(result)
    first=len(result)+1
    for ref in refs:add('PROB',[ref,1,1.])
    last=len(result); maximum=add('MAXX',[first,last]); minimum=add('MINN',[first,last])
    span=add('DIFF',[maximum,minimum]); root=add('OPLT',[span],limit,weight)
    return result,dict(value_rows=refs,proxy_rows=[first,last],maximum_row=maximum,minimum_row=minimum,
        span_row=span,bound_row=root,limit=limit,weighted_penalty_preserved=False,
        sampled_span_feasible_set_preserved=True)


def replace_pairwise_span(rows, value_rows, pair_bounds, limit, *, weight, label='sampled_PV'):
    """Prove full symmetric pair coverage before replacing O(N^2) bounds.

    pair_bounds is a sequence of two weighted root-row IDs for each pair.
    Source DIFF orientation can vary, but every distinct unordered value pair
    must appear exactly once with bounds [-limit,+limit].
    """
    source=normalize(rows); value_rows=list(value_rows); values=set(value_rows); pairs=set(); removed=set()
    for bounds in pair_bounds:
        if len(bounds)!=2 or any(type(i)is not int or not 1<=i<=len(source)for i in bounds):raise ValueError('Each focus pair requires valid lower and upper root rows')
        lo,hi=[source[i-1] for i in bounds]
        if (lo['operand']!='OPGT' or hi['operand']!='OPLT' or lo['target']!=-limit or hi['target']!=limit
            or lo['params'][0]!=hi['params'][0] or min(lo['weight'],hi['weight'])<=0.):
            raise ValueError('Pairwise source bounds do not match the requested symmetric span')
        ref=lo['params'][0]
        if not math.isfinite(ref)or ref!=int(ref)or not 1<=ref<min(bounds):
            raise ValueError('Pair bound must reference an earlier valid difference row')
        diff=source[int(ref)-1]
        if any(v!=int(v)or not 1<=v<int(ref)for v in diff['params'][:2]):
            raise ValueError('Focus difference must reference earlier valid value rows')
        a,b=map(int,diff['params'][:2])
        pair=tuple(sorted((a,b)))
        if diff['operand']!='DIFF' or a not in values or b not in values or a==b or pair in pairs:
            raise ValueError('Invalid/duplicate focus difference')
        if removed.intersection(bounds):raise ValueError('A bound root was reused')
        pairs.add(pair);removed.update(bounds)
    if len(pairs)!=len(values)*(len(values)-1)//2:
        raise ValueError('Every distinct focal-value pair must be covered')
    result,proof=append_native_span(source,list(value_rows),limit,weight=weight,label=label)
    proof.update(deactivated_pair_bound_rows=sorted(removed),source_pairs=len(pairs),
                 epigraph_inequalities=2*len(values)+1)
    return result,proof


class SpanEpigraphProblem(EfficientSolveMixin):
    """Lift one sampled span into two explicit numerical bounds, not refocus.

    The wrapped adapter supplies row['span_values']; its derivative provider
    additionally returns 'span_jacobian'. No optical surface/image is moved by
    the two auxiliary coordinates. Caller specifies finite physical bound ranges.
    """
    def __init__(self,problem,limit,*,auxiliary_bounds,scale,jacobian_provider):
        self.problem=problem;self.design_size=len(problem.bounds);self.limit=float(limit);self.scale=float(scale)
        ab=np.asarray(auxiliary_bounds,dtype=float)
        if ab.shape!=(2,2)or not np.isfinite(ab).all()or np.any(ab[:,0]>ab[:,1])or not math.isfinite(scale)or scale<=0. or not math.isfinite(limit)or limit<=0.:
            raise ValueError('Specify finite auxiliary L/U ranges, positive scale and span limit')
        self.bounds=np.r_[np.asarray(problem.bounds),ab/scale];self.provider=jacobian_provider
        self.local_norm_specs=deepcopy(getattr(problem,'local_norm_specs',{}))
        self.vector_names={'wave_vector','spot_vector'}|{spec['vector']for spec in self.local_norm_specs.values()}
    def initial(self,design):
        base=self.problem.batch([design])[0]
        if base is None:raise ValueError('Initial physical design is invalid')
        f=np.asarray(base['span_values']);q=np.r_[design,min(f)/self.scale,max(f)/self.scale]
        if np.any(q<self.bounds[:,0])or np.any(q>self.bounds[:,1]):raise ValueError('Initial focus band lies outside declared auxiliary ranges')
        return q
    def batch(self,qs):
        output=[]
        for q in qs:
            q=np.asarray(q,dtype=float)
            if q.shape!=self.bounds.shape[:1]or not np.isfinite(q).all()or np.any(q<self.bounds[:,0])or np.any(q>self.bounds[:,1]):output.append(None);continue
            base=self.problem.batch([q[:self.design_size]])[0]
            if base is None:output.append(None);continue
            output.append(dict(underlying=base,band=q[-2:]*self.scale,
                **{k:base[k]for k in self.vector_names if k in base}))
        return output
    def slacks(self,row):
        result=dict(self.problem.slacks(row['underlying']))
        if '_span_bounds'in result:raise ValueError('Reserved span gate name')
        result['_span_bounds']=span_epigraph(row['underlying']['span_values'],*row['band'],self.limit)/self.limit
        return result
    def model(self,q):return self.problem.model(np.asarray(q)[:self.design_size])
    def jacobian(self,q,base,keys):
        original=tuple(k for k in keys if k!='_span_bounds');row=base['underlying']
        supplied=self.provider(np.asarray(q)[:self.design_size],row,original)
        if not isinstance(supplied,dict)or 'span_jacobian'not in supplied:raise ValueError('Span lifting needs an explicit focal-value Jacobian')
        j=np.asarray(supplied['slacks']);blocks=self.problem.slacks(row);chunks={};start=0
        for key in original:
            size=np.asarray(blocks[key]).size;chunks[key]=np.pad(j[start:start+size],((0,0),(0,2)));start+=size
        if start!=len(j):raise ValueError('Original span adapter Jacobian layout mismatch')
        span=span_epigraph_jacobian(supplied['span_jacobian'])/self.limit;span[:,-2:]*=self.scale
        chunks['_span_bounds']=span
        result=dict(slacks=np.concatenate([chunks[k]for k in keys]),vectors={k:np.pad(np.asarray(v),[(0,0)]*(np.asarray(v).ndim-1)+[(0,2)])for k,v in supplied.get('vectors',{}).items()})
        if supplied.get('compressed_vectors'):
            result['compressed_vectors']={k:np.concatenate((np.asarray(r)[...,:-1],np.zeros(np.asarray(r).shape[:-1]+(2,)),np.asarray(r)[...,-1:]),axis=-1)for k,r in supplied['compressed_vectors'].items()}
        return result
