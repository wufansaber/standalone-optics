"""Adaptive budgets, exact span reformulation, streamed derivatives and reduction."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np

from scripts.ai_zemax.standalone import Surface
from scripts.ai_zemax.standalone_derivatives import RayParameter,cache_primal_trace,trace_with_jacobian
from scripts.ai_zemax.standalone_constraint_exchange import run_constraint_exchange
from scripts.ai_zemax.standalone_merit_graph import compact,evaluate,MeritGraph
from scripts.ai_zemax.standalone_merit_exchange import run_merit_exchange
from scripts.ai_zemax.standalone_span import append_native_span,replace_pairwise_span,span_epigraph,span_epigraph_jacobian
from scripts.ai_zemax.standalone_streaming import StreamingQR,compress_affine,validate_compressed,affine_rank_factor,bounded_affine_rank
from scripts.ai_zemax.standalone_reduced import AffineReducedProblem,equality_nullspace
from scripts.ai_zemax.standalone_branch_budget import run_budgeted_branches
from scripts.ai_zemax.standalone_local_solver import EfficientSolveMixin
from scripts.ai_zemax.standalone_robust_exchange import run_robust_exchange
from scripts.ai_zemax.standalone_scenarios import ScenarioProblem
from scripts.ai_zemax.standalone_solid_profiles import FullProfileGeometry


def row(op,p=(),target=0.,weight=0.):return dict(operand=op,params=list(p),target=target,weight=weight)


class ExchangeUpgradeTests(unittest.TestCase):
    def test_full_domain_filter_rolls_back_and_learns_omitted_failure(self):
        a,b={'field':0},{'field':1};seen=[]
        def solve(q,samples,index):seen.append(q.copy());return([1.]if index==0 else[.5]),{}
        def audit(q):return[dict(sample=a,gates={'wfe':q[0]-.1}),dict(sample=b,gates={'wfe':.6-q[0]})]
        def merit(records):
            worst=min(float(r['gates']['wfe'])for r in records);return 1-worst if worst<0. else-worst
        result=run_constraint_exchange(solve,audit,[a],[0.],audit_merit=merit,max_rounds=3)
        self.assertTrue(result['passed']);self.assertFalse(result['history'][0]['candidate_accepted'])
        np.testing.assert_array_equal(seen[1],[0.]);self.assertIn(b,result['history'][0]['added_samples'])
        self.assertEqual(result['q'],[.5])

    def test_discovered_points_move_without_changing_the_core_or_hiding_active_points(self):
        count=[0];base={'field':0};extra={'field':.37}
        def audit(q):
            count[0]+=1
            return[dict(sample=base,gates={'wfe':1.}),dict(sample=extra,gates={'wfe':-.1 if count[0]==1 else 1.},discovered=True)]
        result=run_constraint_exchange(lambda q,s,i:(q,{}),audit,[base],[0.],allow_discovered=True)
        self.assertTrue(result['passed']);self.assertEqual(len(result['history']),2)
        count[0]=0
        def hidden(q):
            count[0]+=1
            return[dict(sample=base,gates={'wfe':1.})]+([dict(sample=extra,gates={'wfe':-.1},discovered=True)]if count[0]==1 else[])
        with self.assertRaisesRegex(ValueError,'omitted an active'):run_constraint_exchange(lambda q,s,i:(q,{}),hidden,[base],[0.],allow_discovered=True)
    def test_drop_hysteresis_protected_and_fixed_domain(self):
        calls=[];count=[0]
        samples=[dict(field=i)for i in range(4)]
        def solve(q,active,i):calls.append(active);count[0]+=1;return q,{}
        def audit(q):
            # A new failure appears only after two full audits; field0 can be
            # dropped only after both observations, and field1 is protected.
            return[dict(sample=s,gates={'wfe':-.1 if s['field']==min(2+count[0]//2,3)else 2.})for s in samples]
        result=run_constraint_exchange(solve,audit,samples[:2],[0.],max_rounds=3,add_per_round=1,
            protected_samples=[samples[1]],drop_after=2,drop_slack=1.,max_active=3)
        self.assertFalse(result['passed'])
        self.assertTrue(any(r['dropped_samples']for r in result['history']))
        self.assertIn(samples[1],result['active_samples'])
        self.assertTrue(all(len(a)<=3 for a in calls))

    def test_budget_never_drops_a_protected_constraint_to_hide_failure(self):
        a,b=dict(field=0),dict(field=1)
        result=run_constraint_exchange(lambda q,s,i:(q,{}),
            lambda q:[dict(sample=a,gates={'wfe':1.}),dict(sample=b,gates={'wfe':-.1})],
            [a],[0.],protected_samples=[a],max_active=1,drop_after=1,drop_slack=.5)
        self.assertFalse(result['passed']);self.assertEqual(result['termination'],'working_budget_exhausted')

    def test_feasible_worst_margin_is_promoted(self):
        a,b=dict(field=0),dict(field=1)
        result=run_constraint_exchange(lambda q,s,i:(q,{}),
            lambda q:[dict(sample=a,gates={'wfe':1.}),dict(sample=b,gates={'wfe':.01})],
            [a],[0.],promote_worst=True,max_rounds=3)
        self.assertTrue(result['passed']);self.assertEqual(len(result['history']),2)
        self.assertTrue(result['worst_set_complete'])

    def test_mf_exchange_recompiles_and_requires_final_acceptance(self):
        source=[row('CONS',target=1.),row('CONS',target=2.),row('OPLT',[1],target=3.,weight=1.),row('OPLT',[2],target=3.,weight=1.)]
        groups=[dict(sample={'field':i},rows=[3+i])for i in range(2)]
        def audit(q):return[dict(sample={'field':i},gates={'wfe':1. if i==0 or q[0] else -.1})for i in range(2)]
        with tempfile.TemporaryDirectory()as d:
            def solve(q,rows,report,index):return np.array([float(index)]),{'rows':len(rows)}
            result=run_merit_exchange(source,groups,[{'field':0}],[0.],protected_rows=[],audit_identity='fixed',
                solve_compiled=solve,audit=audit,output_directory=Path(d)/'run',budget={'rows':5},
                strict_validator=lambda q:{'passed':True})
            self.assertTrue(result['passed']);self.assertEqual(len(result['compiled_rounds']),2)
            nofinal=run_merit_exchange(source,groups,[{'field':0}],[1.],protected_rows=[],audit_identity='fixed',
                solve_compiled=lambda q,*a:(q,{}),audit=audit,output_directory=Path(d)/'no-final',budget={'rows':5})
            self.assertTrue(nofinal['working_domain_passed']);self.assertFalse(nofinal['passed'])


class SpanTests(unittest.TestCase):
    def test_epigraph_exact_feasibility_and_jacobian(self):
        rng=np.random.default_rng(6)
        for _ in range(50):
            values=rng.normal(size=7);limit=2.
            slacks=span_epigraph(values,min(values),max(values),limit)
            self.assertEqual(np.all(slacks>=0.),np.ptp(values)<=limit)
        j=rng.normal(size=(7,3));a=span_epigraph_jacobian(j)
        p=rng.normal(size=5);eps=1e-6;values=rng.normal(size=7)
        fd=(span_epigraph(values+eps*j@p[:3],-.2+eps*p[3],.3+eps*p[4],2.)-
            span_epigraph(values-eps*j@p[:3],-.2-eps*p[3],.3-eps*p[4],2.))/(2*eps)
        np.testing.assert_allclose(a@p,fd,atol=1e-9)

    def test_range_pinning_prevents_noncontiguous_cse(self):
        rows=[row('CONS',target=4.),row('CONS',target=4.),row('CONS',target=2.),row('MAXX',[2,3]),row('MINN',[2,3])]
        rows.extend([row('DIFF',[4,5]),row('OPLT',[6],target=3.,weight=1.)])
        result,report=compact(MeritGraph(rows),active_rows=[7],audit_identity='test')
        np.testing.assert_allclose(evaluate(rows,lambda *_:0.)[4:],evaluate(result,lambda *_:0.)[3:])
        self.assertEqual(report['range_pinned_source_rows'],2)

    def test_pairwise_replacement_requires_complete_exact_coverage(self):
        rows=[row('CONS',target=v)for v in [.1,.2,.3]];pairs=[]
        for a,b in [(1,2),(1,3),(2,3)]:
            rows.append(row('DIFF',[a,b]));ref=len(rows)
            rows.extend([row('OPGT',[ref],target=-.3,weight=2.),row('OPLT',[ref],target=.3,weight=2.)]);pairs.append([len(rows)-1,len(rows)])
        result,proof=replace_pairwise_span(rows,[1,2,3],pairs,.3,weight=2.)
        active=[proof['bound_row']];compiled,report=compact(result,active_rows=active,audit_identity='all')
        self.assertAlmostEqual(evaluate(compiled,lambda *_:0.)[-1],.3)
        self.assertEqual(proof['source_pairs'],3)
        with self.assertRaises(ValueError):replace_pairwise_span(rows,[1,2,3],pairs[:-1],.3,weight=2.)
        bad=deepcopy(rows);bad[pairs[0][0]-1]['target']=-.2
        with self.assertRaises(ValueError):replace_pairwise_span(bad,[1,2,3],pairs,.3,weight=2.)
        for ref in (0.,-1.,3.5,9999.):
            bad=deepcopy(rows)
            for i in pairs[0]:bad[i-1]['params'][0]=ref
            with self.assertRaises(ValueError):replace_pairwise_span(bad,[1,2,3],pairs,.3,weight=2.)


class DerivativeUpgradeTests(unittest.TestCase):
    def test_real_case_field_blocks_preserve_the_physical_envelope_and_layout(self):
        case=Path(__file__).resolve().parents[1]/'examples/algorithm-suite-green8'
        sys.path.insert(0,str(case))
        try:
            from problem import GreenProblem,Design,apply_scenario,dense_first_order
            p=GreenProblem(np.linspace(0.,1.25,5),design=Design(scope='tube'))
            q=np.zeros(len(p.bounds));primal=p.batch([q])[0];analytic=p.evaluate(q,True)
            self.assertIsNotNone(primal);self.assertIsNotNone(analytic)
            for name,value in primal['gates'].items():
                self.assertEqual(np.asarray(value).shape,analytic['gate_J'][name].shape[:-1])
                np.testing.assert_allclose(value,analytic['gates'][name],rtol=1e-7,atol=1e-7)
            active=analytic['gate_J']['physical'][primal['gates']['physical']<.001]
            z,_=equality_nullspace(np.r_[active,analytic['gate_J']['first_order']])
            direction=z@np.random.default_rng(174).normal(size=z.shape[1]);direction/=np.linalg.norm(direction)
            step=1e-4;plus,minus=p.batch([q+step*direction,q-step*direction])
            self.assertIsNotNone(plus);self.assertIsNotNone(minus)
            flatten=lambda r:np.concatenate([np.ravel(v)for v in r['gates'].values()])
            finite=(flatten(plus)-flatten(minus))/(2*step)
            exact=np.concatenate([v.reshape(-1,len(q))for v in analytic['gate_J'].values()])@direction
            self.assertLess(np.linalg.norm(exact-finite)/max(1.,np.linalg.norm(exact)),2e-4)
            scenario=dict(kind='radius',surface=21,relative_R_error=.0001)
            parameters=tuple(replace(a,scale=a.scale/1.0001)if a.surface_index==20 and a.attribute=='curvature_mm_inv'else a for a in p.design.parameters)
            dense,j=dense_first_order(apply_scenario(p.design.model(q),scenario),parameters,derivatives=True,fields=17)
            values=[dense_first_order(apply_scenario(p.design.model(q+sign*step*direction),scenario),parameters,fields=17)[0]for sign in (1,-1)]
            finite=(values[0]-values[1])/(2*step);exact=j@direction
            self.assertLess(np.linalg.norm(exact-finite)/max(1.,np.linalg.norm(exact)),2e-4)
        finally:sys.path.pop(0)
    def test_rank_reduction_is_a_conservative_bound_over_the_entire_box(self):
        rng=np.random.default_rng(31);u,_=np.linalg.qr(rng.normal(size=(12,6)));v,_=np.linalg.qr(rng.normal(size=(6,6)))
        a=u@np.diag([5.,2.,1.,1e-9,1e-11,1e-13])@v
        factor,tail=affine_rank_factor(a,1e-8);lo=np.full(5,-.2);hi=np.full(5,.3)
        reduced=bounded_affine_rank(factor,tail,lo,hi);self.assertLess(len(reduced),len(a))
        for _ in range(100):
            x=np.r_[rng.uniform(lo,hi),1.]
            self.assertGreaterEqual(np.linalg.norm(reduced@x)+1e-13,np.linalg.norm(a@x))
        np.testing.assert_array_equal(affine_rank_factor(a,0.)[0],a)
    def test_streamed_qr_preserves_affine_norm_all_shapes(self):
        rng=np.random.default_rng(11)
        for rays,params in [(3,7),(200,5)]:
            j=rng.normal(size=(2,3,rays,params));v=rng.normal(size=(2,3,rays))
            r=compress_affine(j,v,block_rows=13)
            for _ in range(4):
                step=rng.normal(size=params)
                np.testing.assert_allclose(np.linalg.norm(np.einsum('...np,p->...n',j,step)+v,axis=-1),
                    np.linalg.norm(r@np.r_[step,1.],axis=-1),rtol=1e-12,atol=1e-12)
            validate_compressed(r,v,params)
        with self.assertRaises(ValueError):validate_compressed(r,v*2,params)

    def test_primal_cache_matches_and_invalidates_every_physical_input(self):
        sf=(Surface(2.,1.5,.03),Surface(5.,1.,-.02),Surface(60.))
        origins=np.array([[0.,0.,0.],[0.,.1,0.]]);directions=np.array([[.01,.02,1.],[.02,.01,1.]])
        params=(RayParameter(0,'curvature_mm_inv'),)
        cached=cache_primal_trace(sf,origins,directions)
        a=trace_with_jacobian(sf,origins,directions,params)
        b=trace_with_jacobian(sf,origins,directions,params,primal_cache=cached)
        np.testing.assert_array_equal(a.points_jacobian,b.points_jacobian)
        self.assertIsNotNone(cached.result.surface_points)
        for changed,origin,direction,index in [(sf,origins,directions,1.01),
            ((replace(sf[0],curvature_mm_inv=.031),*sf[1:]),origins,directions,1.),
            (sf,origins+.01,directions,1.),(sf,origins,directions+.001,1.)]:
            with self.assertRaises(ValueError):trace_with_jacobian(changed,origin,direction,params,object_index=index,primal_cache=cached)

    def test_solver_accepts_compressed_derivatives_and_checks_them(self):
        class P(EfficientSolveMixin):
            bounds=np.array([[-2.,2.]])
            local_norm_specs={'wfe':dict(vector='wave_vector',gain=1.,offset=1.)}
            def batch(self,qs):return[dict(wave_vector=np.array([q[0]-1.5]))for q in qs]
            def slacks(self,r):return{'wfe':1.-np.linalg.norm(r['wave_vector'])}
            def model(self,q):return{'q':list(q)}
        p=P()
        def jac(q,base,keys):return dict(slacks=np.array([[1.]]),compressed_vectors={'wave_vector':compress_affine(np.array([[1.]]),base['wave_vector'])})
        with tempfile.TemporaryDirectory()as d:
            r=p.solve(d,5,.5,.2,1e-5,jacobian_provider=jac,subproblem_backend='clarabel',phase2_iterations=2,verbose=False)
            self.assertTrue(r['strict_feasible'])


class ReducedAndBranchTests(unittest.TestCase):
    def test_wrappers_keep_custom_norm_vectors_and_reject_outside_boxes(self):
        from scripts.ai_zemax.standalone_span import SpanEpigraphProblem
        class P:
            bounds=np.array([[-1.,1.],[-1.,1.]])
            local_norm_specs={'quality':dict(vector='custom_phase',gain=1.,offset=1.)}
            def batch(self,qs):return[dict(custom_phase=np.array(q),span_values=np.array(q))for q in qs]
            def slacks(self,row):return{'quality':np.array([1-np.linalg.norm(row['custom_phase'])])}
            def model(self,q):return dict(q=list(q))
        p=P()
        def jac(q,base,keys):
            n=np.linalg.norm(q);return dict(slacks=(-np.array(q)/n if n else np.zeros(2))[None],vectors={'custom_phase':np.eye(2)},span_jacobian=np.eye(2))
        reduced=AffineReducedProblem(p,[0.,0.],[[1.,1.]],[0.],reduced_bounds=[[-.5,.5]],jacobian_provider=jac)
        self.assertIn('custom_phase',reduced.batch([[.1]])[0]);self.assertIsNone(reduced.batch([[.6]])[0])
        lifted=SpanEpigraphProblem(p,.7,auxiliary_bounds=[[-1.,1.],[-1.,1.]],scale=1.,jacobian_provider=jac)
        self.assertIn('custom_phase',lifted.batch([[.1,.2,.1,.2]])[0]);self.assertIsNone(lifted.batch([[.1,.2,.1,2.]])[0])

    def test_mixed_compressed_scenarios_lift_only_their_own_controls(self):
        def evaluate(q,payload,controls):
            v=np.array([q[0]+controls[0],2*q[0]-controls[0],payload*q[0]])
            return dict(gates={'wfe':1-np.linalg.norm(v)},vectors={'wave_vector':v})
        j=np.array([[1.,1.],[2.,-1.],[0.,0.]])
        def derivative(q,payload,controls,base):
            tangent=j.copy();tangent[-1,0]=payload
            gates={'wfe':(-base['vectors']['wave_vector']@tangent/np.linalg.norm(base['vectors']['wave_vector']))[None,:]}
            if payload:return dict(gates=gates,vectors={'wave_vector':tangent})
            return dict(gates=gates,vectors={},compressed_vectors={'wave_vector':compress_affine(tangent,base['vectors']['wave_vector'])})
        p=ScenarioProblem([[-1.,1.]],{'a':0.,'b':1.},evaluate,lambda q:{},
            compensator_bounds={'a':[[-1.,1.]],'b':[[-1.,1.]]},local_jacobian=derivative)
        q=np.array([.1,.2,-.1]);base=p.batch([q])[0];result=p.jacobian(q,base,('wfe',))
        self.assertFalse(result['vectors']);r=result['compressed_vectors']['wave_vector']
        step=np.array([.02,.03,.04]);full=np.stack([j@step[:2],(j+np.array([[0,0],[0,0],[1,0]]))@step[[0,2]]])+base['wave_vector']
        np.testing.assert_allclose(np.linalg.norm(r@np.r_[step,1.],axis=-1),np.linalg.norm(full,axis=-1),atol=1e-13)
        np.testing.assert_array_equal(r[0,:,2],0.);np.testing.assert_array_equal(r[1,:,1],0.)

    def test_branch_validation_binds_the_explicit_polished_state(self):
        advance=lambda b,s,k,t:dict(valid=True,score=0.,state={'x':0})
        with tempfile.TemporaryDirectory()as d:
            r=run_budgeted_branches([{'id':'a'}],target={},rung_budgets=[1],advance=advance,
                validate=lambda b,s,t:dict(passed=True,validated_state={'x':2}),output_directory=Path(d)/'a')
            self.assertEqual(r['finalists'][0]['state'],{'x':2})
            def wrong(b,s,t):s['x']=2;return dict(passed=True)
            with self.assertRaisesRegex(ValueError,'validated_state'):
                run_budgeted_branches([{'id':'a'}],target={},rung_budgets=[1],advance=advance,
                    validate=wrong,output_directory=Path(d)/'b')

    def test_full_solids_refuse_stale_apertures_and_topology(self):
        rows=[dict(curvature_mm_inv=c,clear_radius_mm=.8,mechanical_radius_mm=1.)for c in (.03,-.02,0.)]
        profiles=[dict(surface=i+1,optical_sphere_curvature_mm_inv=rows[i]['curvature_mm_inv'],
            clear_radius_mm=.8,sphere_face_radius_mm=1.,body_radius_mm=1.2,
            outer_annulus='plane at sphere-face-edge sag',imaging_rays_in_annulus=False)for i in range(2)]
        m=dict(vertices_mm=[0.,2.,10.],surface_rows=rows,material_names=['','glass','',''],explicit_nominal_solid_profiles=profiles)
        g=FullProfileGeometry(m,np.zeros((3,1)),np.zeros((3,1)),min_CT=.6,min_air=.25,
            minimum_domain_root=.1,tube_axial_span_max_mm=20.,tube_span_reserve_mm=0.)
        self.assertGreaterEqual(g.constraints(m)[0].min(),0.)
        changed=deepcopy(m);changed['surface_rows'][0]['curvature_mm_inv']=.04
        changed['explicit_nominal_solid_profiles'][0]['optical_sphere_curvature_mm_inv']=.04
        g.constraints(changed)
        for mutate in (lambda x:x['surface_rows'][0].update(clear_radius_mm=.7),
                       lambda x:x['explicit_nominal_solid_profiles'][0].update(body_radius_mm=1.3),
                       lambda x:x['material_names'].__setitem__(1,'')):
            changed=deepcopy(m);mutate(changed)
            with self.assertRaisesRegex(ValueError,'profiles or topology'):g.constraints(changed)

    def test_affine_reduction_retains_bounds_and_full_gate_derivatives(self):
        class P:
            bounds=np.array([[-2.,2.],[-2.,2.],[-2.,2.]])
            def batch(self,qs):return[{'value':np.array(q)**2}for q in qs]
            def slacks(self,r):return{'quadratic':2.-r['value']}
            def model(self,q):return{'q':list(q)}
        p=P();anchor=np.array([.2,-.2,0.]);a=np.array([[1.,1.,0.]])
        def derivative(q,base,keys):return -2*np.diag(q)
        reduced=AffineReducedProblem(p,anchor,a,[0.],reduced_bounds=[[-1,1],[-1,1]],jacobian_provider=derivative)
        z=np.array([.1,.2]);x=reduced.expand(z);np.testing.assert_allclose(a@x,0.,atol=1e-14)
        base=reduced.batch([z])[0];keys=tuple(reduced.slacks(base));j=reduced.jacobian(z,base,keys)['slacks'];eps=1e-6
        flat=lambda q:np.concatenate([np.ravel(v)for v in reduced.slacks(reduced.batch([q])[0]).values()])
        fd=np.column_stack([(flat(z+eps*np.eye(2)[i])-flat(z-eps*np.eye(2)[i]))/(2*eps)for i in range(2)])
        np.testing.assert_allclose(j,fd,atol=1e-9)

    def test_branch_diversity_resume_and_independent_acceptance(self):
        branches=[dict(id=str(i),family=str(i%2),offset=float(i))for i in range(8)]
        def advance(branch,state,budget,target):
            count=(state or {}).get('count',0)+budget
            return dict(valid=True,score=branch['offset']/count,state={'count':count})
        with tempfile.TemporaryDirectory()as d:
            r=run_budgeted_branches(branches,target={'cap':1},rung_budgets=[1,2,4],advance=advance,
                validate=lambda b,s,t:{'passed':b['id']=='0'},output_directory=Path(d)/'branches',workers=2,explore_slots=0)
            self.assertTrue(r['any_passed']);self.assertLess(r['allocated_work'],r['equal_full_budget_work'])
            self.assertEqual({int(i)%2 for i in r['history'][-1]['selected']},{0,1})
            self.assertTrue(all(v['state']['count']==7 for v in r['finalists']))

    def test_real_scenario_controller_promotes_failure_and_keeps_controls_shared(self):
        scenarios={'nominal':0.,'disturbed':1.}
        def factory(selected):
            def evaluate(q,error,controls):return{'gates':{'wave':np.array([q[0]-.5*error,2.-q[0]])}}
            def jac(q,error,controls,base):return{'gates':{'wave':np.array([[1.],[-1.]])},'vectors':{}}
            return ScenarioProblem([[-1.,1.5]],selected,evaluate,lambda q:{'q':q.tolist()},local_jacobian=jac)
        def audit(q,key,error,controls):return{'gates':{'wave':float(min(q[0]-.5*error,2.-q[0]))}}
        with tempfile.TemporaryDirectory()as d:
            r=run_robust_exchange(scenarios,['nominal'],[0.],problem_factory=factory,audit_scenario=audit,
                output_directory=Path(d)/'robust',protected_scenarios=['nominal'],max_rounds=4,
                solve_options=dict(iters=8,start_trust=.5,variant=.2,damping=1e-6,analytic=True,
                    phase2_iterations=3,subproblem_backend='clarabel',verbose=False))
            self.assertTrue(r['passed']);self.assertGreaterEqual(r['q'][0],.5)
            self.assertEqual(r['scenario_controls'],{'nominal':[],'disturbed':[]})


if __name__=='__main__':unittest.main()
