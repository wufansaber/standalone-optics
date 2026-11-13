"""Launch and evaluate a coaxial prescription in finite or infinite conjugates."""

import math

import numpy as np

from .standalone import (Surface, directions_from_slopes, disk_quadrature,
                         paraxial_transfer, reference_opd_mm, trace, wavefront_rms)
from .standalone_analysis import best_focus, spot_statistics


class OpticalSystem:
    """A fixed prescription with physical fields, mm units, and per-wavelength indices."""

    def __init__(self, model, *, allow_signed_segments=False):
        self.model = model
        self.allow_signed_segments = allow_signed_segments
        vertices = np.asarray(model['vertices_mm'], dtype=float)
        self.wavelengths = np.asarray(model['wavelengths_um'], dtype=float)
        self.indices = np.asarray(model['indices_by_wave'], dtype=float)
        self.fields = np.asarray(model['fields'], dtype=float)
        self.stop = int(model['stop_surface'])
        self.primary = int(model['primary_wave'])
        self.infinite = model['object_at_infinity']
        self.gaussian_apodization = float(model.get('gaussian_apodization',0.))
        if not math.isfinite(self.gaussian_apodization) or self.gaussian_apodization < 0:
            raise ValueError('Invalid Gaussian pupil factor')
        if (vertices.ndim != 1 or len(vertices) < 2 or not np.isfinite(vertices).all()
                or np.any(np.diff(vertices) < 0) or vertices[0] < 0
                or not 1 <= self.stop < len(vertices)):
            raise ValueError('Invalid sequential vertices or stop surface')
        if (self.wavelengths.ndim != 1 or not len(self.wavelengths)
                or not np.isfinite(self.wavelengths).all() or np.any(self.wavelengths <= 0)
                or self.indices.shape != (len(self.wavelengths), len(vertices)+1)
                or not np.isfinite(self.indices).all() or np.any(self.indices <= 0)
                or not 0 <= self.primary < len(self.wavelengths)):
            raise ValueError('Invalid wavelength or material-index table')
        if self.fields.ndim != 2 or self.fields.shape[1] != 2 or not len(self.fields) or not np.isfinite(self.fields).all():
            raise ValueError('Fields must be a finite nonempty list of XY pairs')
        if len(model['surface_rows']) != len(vertices):
            raise ValueError('Surface rows and vertices must align')
        self.surfaces = []
        for indices in self.indices:
            rows = []
            for z, index, data in zip(vertices, indices[1:], model['surface_rows']):
                aperture = data.get('aperture_radius_mm')
                rows.append(Surface(z, index, data.get('curvature_mm_inv', 0.), data.get('conic', 0.),
                                    tuple(data.get('even_coefficients', ())), np.inf if aperture is None else aperture))
            self.surfaces.append(tuple(rows))
        primary_surfaces = self.surfaces[self.primary]
        n_object = self.indices[self.primary,0]
        entrance_matrix = paraxial_transfer(primary_surfaces[:self.stop], 0., n_object)
        a,b = entrance_matrix[0]
        if abs(b) < 1e-14 and not self.infinite:
            raise ValueError('Object and stop conjugates give a singular launch convention')
        self.chief_slope_scale = -a/(b*n_object) if not self.infinite else 0.
        self.entrance_pupil_z_mm = b*n_object/a if abs(a)>1e-14 else np.inf
        telecentric = bool(model.get('object_space_telecentric',False))
        if telecentric:
            if self.infinite or model['aperture_type']!='obna':
                raise ValueError('Telecentric override currently requires a finite object and OBNA')
            self.entrance_pupil_z_mm = vertices[0]+1e10
            self.chief_slope_scale = -1./self.entrance_pupil_z_mm
        aperture_value = float(model['aperture_value'])
        if not math.isfinite(aperture_value) or aperture_value <= 0:
            raise ValueError('Aperture value must be finite and positive')
        if model['aperture_type']=='obna' and not self.infinite:
            sine = aperture_value/n_object
            if not 0 < sine < 1:
                raise ValueError('Object-space NA must be below the object index')
            self.pupil_scale = sine/math.sqrt(1.-sine*sine)
        elif model['aperture_type']=='enpd' and math.isfinite(self.entrance_pupil_z_mm):
            self.entrance_radius_mm = aperture_value/2.
            if not self.infinite:
                self.pupil_scale = self.entrance_radius_mm/self.entrance_pupil_z_mm
        else:
            raise ValueError('Use finite-conjugate OBNA or a finite entrance pupil with ENPD')
        if telecentric:
            transfer = paraxial_transfer(primary_surfaces,0.,n_object)
            numerator,denominator = transfer[:,0]
        else:
            transfer = paraxial_transfer(primary_surfaces[self.stop:],vertices[self.stop-1],self.indices[self.primary,self.stop])
            numerator,denominator = transfer[:,1]
        self.reference_distance_mm = self.indices[self.primary,-1]*numerator/denominator if abs(denominator)>1e-14 else 1e15
        self.image_z_mm = vertices[-1]

    def _shoot(self, wave, pupil, object_xy, slopes, *, record_path=False):
        pupil = np.asarray(pupil, dtype=float)
        if pupil.ndim != 2 or pupil.shape[1] != 2 or not np.isfinite(pupil).all():
            raise ValueError('Use finite pupil coordinates with shape (N,2)')
        surfaces = self.surfaces[wave]
        n_object = self.indices[wave,0]
        if not self.infinite:
            directions = directions_from_slopes(slopes + self.pupil_scale*pupil)
            return trace(surfaces,np.r_[object_xy,0.],directions,n_object,
                         allow_signed_segments=self.allow_signed_segments, record_path=record_path)
        # Parallel rays originate on a common phase plane. Build their lines
        # through the paraxial entrance pupil before translating the origins.
        direction = directions_from_slopes(np.asarray(slopes)[None,:])[0]
        pupil_xy = self.entrance_radius_mm*pupil
        at_vertex = pupil_xy-self.entrance_pupil_z_mm*slopes
        first = surfaces[0]
        sag = first.sag_and_gradient(np.sum(at_vertex*at_vertex, axis=1))[0]
        finite_sag = sag[np.isfinite(sag)]
        clearance = 2.*self.entrance_radius_mm+1.
        if len(finite_sag):
            clearance += float(np.max(abs(finite_sag)))
        points = np.c_[at_vertex-clearance*slopes, np.full(len(pupil),first.vertex_mm-clearance)]
        reference = np.r_[-(clearance+self.entrance_pupil_z_mm)*slopes,first.vertex_mm-clearance]
        points -= ((points-reference)@direction)[:,None]*direction
        result = trace(surfaces,points,np.broadcast_to(direction,points.shape),n_object,
                       allow_signed_segments=self.allow_signed_segments, record_path=record_path)
        # Reference the phase to d·P = 0 for both the bundle and its chief.
        result.opl_mm += n_object*float(reference@direction)
        return result

    def launch_field(self, field_xy, reference_wave=None):
        """Return physical object height and chief slope for a declared field."""
        field = np.asarray(field_xy,dtype=float)
        if field.shape != (2,) or not np.isfinite(field).all():
            raise ValueError('Field must contain two finite coordinates')
        kind = self.model['field_type']
        if kind=='object_height' and not self.infinite:
            return field,self.chief_slope_scale*field
        if kind=='angle':
            slopes = np.tan(np.deg2rad(field))
            if self.infinite or not np.any(slopes):
                return np.zeros(2),slopes
            if abs(self.chief_slope_scale)<1e-14:
                raise ValueError('Angle does not identify a finite telecentric field')
            return slopes/self.chief_slope_scale,slopes
        if kind!='real_image_height':
            raise ValueError(f'Unsupported field convention: {kind}')
        # Native normalized batch rays use the primary color as their target.
        # Monochromatic RMS operands use the selected measurement wavelength.
        reference_wave = self.primary if reference_wave is None else reference_wave
        if not 0 <= reference_wave < len(self.wavelengths):
            raise ValueError('Field reference wavelength is outside the model')
        origin = np.zeros(2)
        parameter = np.zeros(2)
        zero_pupil = np.zeros((1,2))
        def intercept(estimate):
            position,slope = (origin,estimate) if self.infinite else (estimate,self.chief_slope_scale*estimate)
            chief = self._shoot(reference_wave,zero_pupil,position,slope)
            if not chief.valid[0]:
                raise RuntimeError('Chief ray failed while solving the image-height field')
            return chief.points[0,:2]
        for _ in range(15):
            point = intercept(parameter)
            error = point-field
            if np.linalg.norm(error)<2e-11:
                return (origin,parameter) if self.infinite else (parameter,self.chief_slope_scale*parameter)
            step = 1e-6
            jacobian = np.column_stack([(intercept(parameter+delta)-point)/step for delta in step*np.eye(2)])
            parameter -= np.linalg.solve(jacobian,error)
        raise RuntimeError('Real-image-height chief ray did not converge')

    def evaluate(self, field_xy, wave, pupil=None, weights=None, *, use_primary_field=False):
        """Return fixed-image aberrations and the actual traced bundle."""
        if not 0 <= wave < len(self.wavelengths):
            raise ValueError('Wavelength index is outside the model')
        if pupil is None:
            pupil,weights = disk_quadrature(20,64)
        object_xy,slopes = self.launch_field(field_xy,None if use_primary_field else wave)
        rays = self._shoot(wave,pupil,object_xy,slopes)
        chief = self._shoot(wave,np.zeros((1,2)),object_xy,slopes)
        if not rays.valid.all() or not chief.valid[0]:
            failed = rays if not rays.valid.all() else chief
            bad = np.flatnonzero(~failed.valid)
            first = bad[0]
            raise ValueError(f'{len(bad)} rays failed. First failure at surface {failed.failed_surface[first]+1}: {failed.reason[first]}')
        radius = self.reference_distance_mm/chief.directions[0,2]
        opd = reference_opd_mm(rays,chief,radius,self.indices[wave,-1])
        phase_weights = self.pupil_weights(pupil,weights,for_spot=False)
        spot_weights = self.pupil_weights(pupil,weights,for_spot=True)
        rms,_ = wavefront_rms(opd,phase_weights,self.wavelengths[wave],rays.directions[:,:2])
        spot = spot_statistics(rays.points[:,:2],spot_weights)
        axis = np.asarray(field_xy,dtype=float)
        if not np.any(axis):
            axis = np.array([0.,1.])
        focus = best_focus(rays.points[:,:2],rays.directions,spot_weights,tangential_axis=axis)
        shortest = int(np.argmin(rays.shortest_segment_mm))
        return {'wavefront_rms_waves':rms,'spot_rms_um':spot['rms_radius']*1000.,
                'chief_xy_mm':chief.points[0,:2].tolist(),'focus_shifts_mm':focus,
                'rays_with_negative_segments':int(np.count_nonzero(rays.negative_segment_count)),
                'shortest_signed_segment_mm':float(rays.shortest_segment_mm[shortest]),
                'shortest_segment_target_surface':int(rays.shortest_segment_surface[shortest])+1},rays,chief

    def pupil_weights(self, pupil, weights, *, for_spot):
        """Apply the Gaussian amplitude for phase or intensity for spot moments."""
        pupil = np.asarray(pupil,dtype=float)
        weights = np.ones(len(pupil)) if weights is None else np.asarray(weights,dtype=float)
        if weights.shape != (len(pupil),):
            raise ValueError('Provide one pupil weight per ray')
        exponent = -self.gaussian_apodization*np.sum(pupil*pupil,axis=1)
        return weights*np.exp((2. if for_spot else 1.)*exponent)


def rectangular_pupil(sampling):
    """A unit disk sampled by centers of a 2n by 2n grid."""
    if not isinstance(sampling,int) or not 1 <= sampling <= 128:
        raise ValueError('Rectangular sampling must be an integer from 1 to 128')
    axis = (np.arange(2*sampling)+.5)/sampling-1.
    x,y = np.meshgrid(axis,axis)
    pupil = np.column_stack((x.ravel(),y.ravel()))
    pupil = pupil[np.sum(pupil*pupil,axis=1)<=1.]
    return pupil,np.full(len(pupil),1./len(pupil))
