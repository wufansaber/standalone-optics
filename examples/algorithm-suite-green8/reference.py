"""Project-only native-compatible finite telecentric reference and exact tangent.

No installed backend/optimizer is changed. This helper is restricted to the
project's finite object-height, object-telecentric, fixed planar window models.
Its .01mm real-parabasal convention is explicit and versioned; it is not a
general claim about other OpticStudio versions or optical geometries.
"""
from __future__ import annotations
from dataclasses import dataclass,replace
import hashlib
from pathlib import Path
import sys
import numpy as np

from scripts.ai_zemax.standalone import trace,directions_from_slopes
from scripts.ai_zemax.standalone_system import OpticalSystem
from scripts.ai_zemax import standalone_derivatives as derivatives

MODULE_SOURCE=Path(__file__).read_bytes()
MODULE_SHA=hashlib.sha256(MODULE_SOURCE).hexdigest()
MODE='finite_telecentric_native_parabasal'
FIELD_STEP_MM=.01


@dataclass
class ReferenceTangent:
    distance_mm:float
    inverse_distance_mm_inv:float
    distance_jacobian:np.ndarray
    inverse_distance_jacobian:np.ndarray
    parameters:tuple
    information:dict


class NativeReferenceSystem(OpticalSystem):
    """Ordinary unsigned primal tracer with a separately defined reference.

    Actual physical surfaces, indices, rays and spot statistics are unchanged.
    The reference uses the primary real chief ray at fixed object h=.01mm.
    No per-field/per-wavelength/native-value fitting is performed.
    """

    def __init__(self,model,*,allow_signed_segments=False):
        if allow_signed_segments:raise ValueError('This project reference requires unsigned physical ray tracing')
        units=model.get('lens_units',model.get('units','mm'))
        if not isinstance(units,str) or units.lower() not in ('mm','millimeter','millimeters','millimetre','millimetres'):
            raise ValueError('This explicitly calibrated reference mode requires millimetre neutral-model units')
        super().__init__(model,allow_signed_segments=False)
        if self.infinite or model['field_type']!='object_height' or not model.get('object_space_telecentric') or model['aperture_type']!='obna':
            raise ValueError('Native parabasal mode is limited to finite object-height telecentric OBNA models')
        if not model.get('external_window') or len(self.surfaces[self.primary])<3:
            raise ValueError('Project fixed-window prefix must be explicit')
        first=self.surfaces[self.primary][:2]
        if abs(first[0].vertex_mm)>1e-12 or abs(first[1].vertex_mm-.725)>1e-12 or any(sf.curvature_mm_inv!=0. for sf in first):
            raise ValueError('Project window must remain planar and fixed at0/.725mm')
        self.fixed_prefix=2
        self.paraxial_reference_distance_mm=float(self.reference_distance_mm)
        self.probe_field=np.array([0.,FIELD_STEP_MM])
        probe=self._shoot(self.primary,np.zeros((1,2)),*self.launch_field(self.probe_field))
        if not probe.valid.all():raise ValueError('Primary parabasal reference probe is not a valid physical ray')
        py=float(probe.points[0,1]);dy=float(probe.directions[0,1]);dz=float(probe.directions[0,2])
        if abs(py*dz)<1e-20 or abs(dy)<1e-20:
            raise ValueError('Reference probe gives a singular/infinite exit pupil; use an explicitly supported limit mode')
        inverse=dy/(py*dz)
        if not np.isfinite(inverse) or abs(inverse)<1e-20:raise ValueError('Nonfinite/infinite parabasal reference')
        self.reference_distance_mm=float(1./inverse)
        self.reference_inverse_mm_inv=float(inverse)
        self.reference_mode=MODE
        self.reference_information=dict(mode=MODE,field_step_mm=FIELD_STEP_MM,
            lens_units='mm',validated_native_version='OpticStudio2022R2.02',
            primary_wavelength_um=float(self.wavelengths[self.primary]),primary_wave_zero_based=self.primary,
            window_prefix_surfaces=2,probe_image_y_mm=py,probe_direction_y=dy,probe_direction_z=dz,
            axial_reference_distance_mm=self.reference_distance_mm,inverse_reference_distance_mm_inv=inverse,
            prior_paraxial_reference_distance_mm=self.paraxial_reference_distance_mm,
            source_sha256=MODULE_SHA,trace_tangent_source_sha256=hashlib.sha256(Path(derivatives.__file__).read_bytes()).hexdigest(),
            scope='project finite telecentric native real-parabasal compatibility; no ray/material modification')

    def _active_trace(self,wave,pupil,field,parameters):
        """Differentiate after the constant window, retaining its physical OPL."""
        parameters=tuple(parameters)
        if any(p.surface_index<self.fixed_prefix for p in parameters):
            raise ValueError('Window/object launch prefix must be fixed in this reference tangent')
        pupil=np.asarray(pupil,dtype=float);origin,slopes=self.launch_field(field)
        directions=directions_from_slopes(slopes+self.pupil_scale*pupil)
        origins=np.broadcast_to(np.r_[origin,0.],directions.shape).copy()
        prefix=trace(self.surfaces[wave][:self.fixed_prefix],origins,directions,self.indices[wave,0])
        if not prefix.valid.all():raise ValueError('A fixed-window ray failed')
        shifted=tuple(replace(p,surface_index=p.surface_index-self.fixed_prefix) for p in parameters)
        result=derivatives.trace_with_jacobian(self.surfaces[wave][self.fixed_prefix:],prefix.points,prefix.directions,
            shifted,object_index=float(self.indices[wave,self.fixed_prefix]))
        if not result.differentiable.all():
            bad=int(np.flatnonzero(~result.differentiable)[0])
            raise ValueError('Reference tangent requires all valid smooth rays: '+str(result.derivative_reason[bad]))
        result.trace.opl_mm+=prefix.opl_mm
        return result

    def reference_tangent(self,parameters):
        parameters=tuple(parameters)
        value=self._active_trace(self.primary,np.zeros((1,2)),self.probe_field,parameters)
        py=float(value.trace.points[0,1]);dy=float(value.trace.directions[0,1]);dz=float(value.trace.directions[0,2])
        dpy=value.points_jacobian[0,1];ddy=value.directions_jacobian[0,1];ddz=value.directions_jacobian[0,2]
        inverse=dy/(py*dz)
        dinverse=ddy/(py*dz)-dy*dpy/(py*py*dz)-dy*ddz/(py*dz*dz)
        distance=1./inverse;ddistance=-dinverse/(inverse*inverse)
        return ReferenceTangent(float(distance),float(inverse),ddistance,dinverse,parameters,
            dict(self.reference_information,analytic_formula='invR=dirY/(pointY*dirZ); exact forward ray tangents; h/window fixed'))

    def opd_tangent(self,field,wave,nodes,parameters):
        """Return OPDmm/Jmm, primary reference, bundle and chief tangents."""
        parameters=tuple(parameters);reference=self.reference_tangent(parameters)
        bundle=self._active_trace(wave,nodes,field,parameters)
        chief=self._active_trace(wave,np.zeros((1,2)),field,parameters)
        dz=float(chief.trace.directions[0,2]);ddz=chief.directions_jacobian[0,2]
        radius=reference.distance_mm/dz
        dradius=reference.distance_jacobian/dz-reference.distance_mm*ddz/(dz*dz)
        image_jac=np.asarray([p.scale if p.attribute=='index_after' and
            p.surface_index==len(self.surfaces[wave])-1 else 0. for p in parameters])
        opd,jac=derivatives.reference_opd_with_jacobian(bundle,chief,radius,float(self.indices[wave,-1]),
            pupil_distance_jacobian=dradius,image_index_jacobian=image_jac)
        return opd,jac,reference,bundle,chief

    def wavefront_tangent(self,field,wave,nodes,weights,parameters):
        """Exact signed phase/RMS tangent with this moving primary reference."""
        opd,jac,reference,bundle,chief=self.opd_tangent(field,wave,nodes,parameters)
        phase_weights=self.pupil_weights(nodes,weights,for_spot=False)
        result=derivatives.wavefront_rms_with_jacobian(opd,jac,phase_weights,float(self.wavelengths[wave]),
            bundle.trace.directions[:,:2],tilt_jacobian=bundle.directions_jacobian[:,:2])
        return result,reference
