"""Exact affine equality elimination and local equality-tangent directions."""
from copy import deepcopy
import numpy as np

from .standalone_local_solver import EfficientSolveMixin


def equality_nullspace(jacobian, *, relative_tolerance=1e-11):
    a=np.asarray(jacobian,dtype=float)
    if a.ndim!=2 or not a.shape[1] or not np.isfinite(a).all() or not np.isfinite(relative_tolerance) or relative_tolerance<=0.:
        raise ValueError('Require a finite equality Jacobian and positive relative tolerance')
    _,s,vt=np.linalg.svd(a,full_matrices=True)
    threshold=relative_tolerance*(s[0] if len(s)and s[0] else 1.)
    rank=int(np.sum(s>threshold))
    z=vt[rank:].T.copy()
    if np.linalg.norm(a@z)>max(1.,np.linalg.norm(a))*relative_tolerance*10:
        raise ValueError('Equality nullspace residual is too large')
    return z,dict(rank=rank,full_parameters=a.shape[1],reduced_parameters=z.shape[1],
                  singular_values=s.tolist(),linearized_only=True)


class AffineReducedProblem(EfficientSolveMixin):
    """Eliminate A x=b exactly via x=anchor+Z z; keep all physical gates.

    reduced_bounds is an explicit finite search box, not an inferred global
    equivalent domain. Original parameter bounds are retained as constraints
    and exactly checked before any physical evaluation. For nonlinear EFL or
    magnification equalities, equality_nullspace provides only local directions;
    those equalities still need exact nonlinear checks/retraction.
    """
    def __init__(self,problem,anchor,equalities,rhs,*,reduced_bounds,jacobian_provider=None):
        self.problem=problem;self.anchor=np.asarray(anchor,dtype=float).copy()
        self.a=np.asarray(equalities,dtype=float);self.b=np.asarray(rhs,dtype=float)
        bounds=np.asarray(problem.bounds,dtype=float)
        if (bounds.ndim!=2 or bounds.shape[1]!=2 or self.anchor.shape!=(len(bounds),)
            or self.a.ndim!=2 or self.a.shape[1]!=len(bounds)or self.b.shape!=(len(self.a),)
            or not all(np.isfinite(x).all()for x in [bounds,self.anchor,self.a,self.b])
            or np.any(self.anchor<bounds[:,0])or np.any(self.anchor>bounds[:,1])
            or not np.allclose(self.a@self.anchor,self.b,rtol=0.,atol=1e-11)):
            raise ValueError('Anchor must satisfy finite affine equalities and original bounds')
        self.basis,self.reduction=equality_nullspace(self.a)
        if not self.basis.shape[1]:raise ValueError('All parameters are fixed by the affine equalities')
        self.bounds=np.asarray(reduced_bounds,dtype=float)
        if (self.bounds.shape!=(self.basis.shape[1],2)or not np.isfinite(self.bounds).all()
            or np.any(self.bounds[:,0]>0.)or np.any(self.bounds[:,1]<0.)):
            raise ValueError('Reduced bounds must be a finite ordered box containing zero')
        self.original_bounds=bounds;self.provider=jacobian_provider
        self.local_norm_specs=deepcopy(getattr(problem,'local_norm_specs',{}))
        self.vector_names={'wave_vector','spot_vector'}|{spec['vector']for spec in self.local_norm_specs.values()}

    def expand(self,z):
        z=np.asarray(z,dtype=float)
        if z.shape!=(self.basis.shape[1],)or not np.isfinite(z).all():raise ValueError('Invalid reduced parameter vector')
        return self.anchor+self.basis@z

    def batch(self,zs):
        output=[]
        for z in zs:
            z=np.asarray(z,dtype=float)
            if z.shape!=self.bounds.shape[:1]or not np.isfinite(z).all()or np.any(z<self.bounds[:,0])or np.any(z>self.bounds[:,1]):output.append(None);continue
            x=self.expand(z)
            if np.any(x<self.original_bounds[:,0])or np.any(x>self.original_bounds[:,1]):output.append(None);continue
            row=self.problem.batch([x])[0]
            if row is None:output.append(None);continue
            blocks=self.problem.slacks(row)
            if '_parameter_bounds' in blocks:raise ValueError('Reserved affine-reduction gate name')
            output.append(dict(underlying=row,x=x,**{k:row[k]for k in self.vector_names if k in row}))
        return output

    def slacks(self,row):
        result=dict(self.problem.slacks(row['underlying']))
        result['_parameter_bounds']=np.r_[row['x']-self.original_bounds[:,0],self.original_bounds[:,1]-row['x']]
        return result

    def model(self,z):return self.problem.model(self.expand(z))

    def jacobian(self,z,base,keys):
        if self.provider is None:raise ValueError('No original analytic Jacobian provider supplied')
        original_keys=tuple(k for k in keys if k!='_parameter_bounds')
        supplied=self.provider(self.expand(z),base['underlying'],original_keys)
        if not isinstance(supplied,dict):supplied={'slacks':supplied}
        blocks=self.problem.slacks(base['underlying']);j=np.asarray(supplied['slacks'])
        chunks={};start=0
        for key in original_keys:
            size=np.asarray(blocks[key]).size;chunks[key]=j[start:start+size]@self.basis;start+=size
        if start!=len(j):raise ValueError('Original Jacobian does not match its declared gate layout')
        chunks['_parameter_bounds']=np.r_[self.basis,-self.basis]
        result=dict(slacks=np.concatenate([chunks[key]for key in keys]),
                    vectors={k:np.asarray(v)@self.basis for k,v in supplied.get('vectors',{}).items()})
        if supplied.get('compressed_vectors'):
            result['compressed_vectors']={}
            for key,r in supplied['compressed_vectors'].items():
                r=np.asarray(r);mapped=np.concatenate((r[...,:-1]@self.basis,r[...,-1:]),axis=-1)
                result['compressed_vectors'][key]=np.linalg.qr(mapped,mode='r')
        return result
