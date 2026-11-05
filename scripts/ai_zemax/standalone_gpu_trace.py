"""Optional CuPy FP64 sequential ray trace with one CUDA thread per ray."""

from functools import lru_cache

import numpy as np

from .standalone import TraceResult


TRACE_REASONS = ('', 'no_intersection', 'negative_segment', 'aperture', 'tir', 'backward')


def require_cupy(device=0):
    """Load CuPy on demand and validate the requested CUDA device."""
    if type(device) is not int or device < 0:
        raise ValueError('CUDA device must be a nonnegative integer')
    try:
        import cupy
    except (ImportError, OSError) as exc:
        raise RuntimeError('GPU tracing requires CuPy and a working CUDA runtime') from exc
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError as exc:
        raise RuntimeError(f'Could not initialize the CUDA runtime: {exc}') from exc
    if device >= count:
        raise RuntimeError(f'CUDA device {device} is unavailable ({count} visible devices)')
    return cupy


# Parameter rows are [vertex, index after, curvature, conic, aperture, number of
# even coefficients, A2, A4, ...]. The optical conventions and tolerances
# match standalone.trace. Disabling fused multiply-add makes comparisons near
# a tangent intersection or the strict negative-segment boundary reproducible.
_TRACE_SOURCE = r"""
__device__ double axis_sum(double x, double y, double z) {
    return (x + y) + z;
}

__device__ double conic_start(double c, double k, double x, double y, double z,
                              double dx, double dy, double dz, bool signed_paths) {
    double plane = -z / dz;
    if (c == 0.) return plane;
    double scale = 1. + k;
    double a = c * axis_sum(dx*dx, dy*dy, (dz*dz)*scale);
    double b = c * axis_sum(x*dx, y*dy, (z*dz)*scale) - dz;
    double f = c * axis_sum(x*x, y*y, (z*z)*scale) - 2.*z;
    double q = -b - copysign(sqrt(b*b - a*f), b);
    double r0 = fabs(a) < 1e-15 ? -f / (2.*b) : q/a;
    double r1 = f/q;
    bool good0 = isfinite(r0) && c*scale*(z+r0*dz) <= 1. + 1e-12;
    bool good1 = isfinite(r1) && c*scale*(z+r1*dz) <= 1. + 1e-12;
    if (!signed_paths) {
        good0 = good0 && r0 >= -1e-9;
        good1 = good1 && r1 >= -1e-9;
    }
    if (good0 && good1) {
        return signed_paths ? (fabs(r0-plane) <= fabs(r1-plane) ? r0 : r1) : fmin(r0,r1);
    }
    return good0 ? r0 : (good1 ? r1 : plane);
}

__device__ void sag_gradient(const double* surface, double radius_squared,
                             double* sag, double* derivative) {
    double c = surface[2];
    double root = sqrt(1. - (((1. + surface[3]) * c) * c) * radius_squared);
    *sag = c * radius_squared / (1. + root);
    *derivative = c / (2. * root);
    double previous_power = 1.;
    for (int i=0; i<(int)surface[5]; ++i) {
        double coefficient = surface[6+i];
        double current_power = previous_power * radius_squared;
        *sag += coefficient * current_power;
        *derivative += ((i+1) * coefficient) * previous_power;
        previous_power = current_power;
    }
}

extern "C" __global__
void trace_all(const double* geometry, int surfaces, int stride, const double* object_indices,
               const double* origins, const double* incident, long long total, int rays_per_group,
               int signed_paths, int record_path, double* points, double* directions,
               double* opl, bool* valid, int* failure, unsigned char* reason,
               int* negative_count, double* shortest, int* shortest_surface, double* history) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    long long group = i / rays_per_group;
    long long offset = 3*i;
    double x = origins[offset], y = origins[offset+1], z = origins[offset+2];
    double dx = incident[offset], dy = incident[offset+1], dz = incident[offset+2];
    double length = sqrt(axis_sum(dx*dx,dy*dy,dz*dz));
    dx /= length; dy /= length; dz /= length;
    double path = 0.;
    double shortest_value = __longlong_as_double(0x7ff0000000000000LL);
    int neg = 0, shortest_at = -1, failed_at = -1;
    unsigned char code = 0;
    double before = object_indices[group];
    for (int s=0; s<surfaces; ++s) {
        const double* surface = geometry + (group * surfaces + s) * stride;
        double vertex = surface[0];
        double local_z = z - vertex;
        double distance = conic_start(surface[2], surface[3], x,y,local_z,dx,dy,dz,signed_paths);
        for (int step=0; step<24; ++step) {
            double hx=x+distance*dx, hy=y+distance*dy, hz=local_z+distance*dz;
            double sag, derivative;
            sag_gradient(surface, hx*hx+hy*hy, &sag, &derivative);
            double residual = hz - sag;
            if (isfinite(residual) && fabs(residual) <= 2e-12) break;
            double slope = dz - 2.*derivative*(hx*dx+hy*dy);
            if (!isfinite(slope) || fabs(slope) <= 1e-13) break;
            distance -= residual/slope;
        }
        double hx=x+distance*dx, hy=y+distance*dy, hz=local_z+distance*dz;
        double radius_squared = hx*hx+hy*hy;
        double sag, derivative;
        sag_gradient(surface, radius_squared, &sag, &derivative);
        double nx=(-2.*hx)*derivative, ny=(-2.*hy)*derivative, nz=1.;
        if (!(fabs(hz-sag) <= 1e-10 && isfinite(nx) && isfinite(ny))) {
            code = 1;
        } else if (!signed_paths && distance < -1e-9) {
            code = 2;
        } else if (sqrt(radius_squared) > surface[4]+1e-9) {
            code = 3;
        }
        if (code) {
            failed_at=s;
            break;
        }
        double normal_length=sqrt(axis_sum(nx*nx,ny*ny,1.));
        nx/=normal_length; ny/=normal_length; nz/=normal_length;
        double cosine=axis_sum(dx*nx,dy*ny,dz*nz);
        if (cosine < 0.) {
            nx=-nx; ny=-ny; nz=-nz;
        }
        double ratio=before/surface[1];
        double tx=ratio*(dx-fabs(cosine)*nx);
        double ty=ratio*(dy-fabs(cosine)*ny);
        double tz=ratio*(dz-fabs(cosine)*nz);
        double normal_squared=1.-axis_sum(tx*tx,ty*ty,tz*tz);
        if (!(normal_squared >= -1e-14)) {
            code=4; failed_at=s;
            break;
        }
        double normal_component=sqrt(fmax(normal_squared,0.));
        double out_z=tz+normal_component*nz;
        if (!(out_z > 0.)) {
            code=5; failed_at=s;
            break;
        }
        dx=tx+normal_component*nx; dy=ty+normal_component*ny; dz=out_z;
        x=hx; y=hy; z=hz+vertex;
        if (distance < -1e-9) ++neg;
        if (distance < shortest_value) {
            shortest_value=distance; shortest_at=s;
        }
        path += before*(signed_paths ? distance : fmax(distance,0.));
        if (record_path) {
            long long location=3*((long long)s*total+i);
            history[location]=x; history[location+1]=y; history[location+2]=z;
        }
        before=surface[1];
    }
    if (failed_at >= 0) {
        double missing=__longlong_as_double(0x7ff8000000000000LL);
        x=y=z=dx=dy=dz=path=missing;
    }
    points[offset]=x; points[offset+1]=y; points[offset+2]=z;
    directions[offset]=dx; directions[offset+1]=dy; directions[offset+2]=dz;
    opl[i]=path; valid[i]=failed_at < 0; failure[i]=failed_at; reason[i]=code;
    negative_count[i]=neg; shortest[i]=shortest_value; shortest_surface[i]=shortest_at;
}
"""


@lru_cache(maxsize=1)
def _kernel(cupy):
    return cupy.RawKernel(_TRACE_SOURCE, 'trace_all', options=('--fmad=false',))


def _pack_groups(surfaces_by_group):
    groups = tuple(tuple(surfaces) for surfaces in surfaces_by_group)
    if not groups or not groups[0] or any(len(group) != len(groups[0]) for group in groups):
        raise ValueError('GPU batch groups must have the same nonzero surface count')
    if any(b.vertex_mm < a.vertex_mm for group in groups for a, b in zip(group, group[1:])):
        raise ValueError('Provide surfaces in sequential order')
    stride = 6 + max(len(surface.even_coefficients) for group in groups for surface in group)
    packed = np.zeros((len(groups), len(groups[0]), stride), dtype=np.float64)
    cache = {}
    for i, group in enumerate(groups):
        identity = id(group)
        if identity in cache:
            packed[i] = packed[cache[identity]]
            continue
        for j, surface in enumerate(group):
            coefficients = surface.even_coefficients
            row = packed[i, j]
            row[:6] = (surface.vertex_mm, surface.index_after, surface.curvature_mm_inv,
                       surface.conic, surface.aperture_radius_mm, len(coefficients))
            row[6:6+len(coefficients)] = coefficients
        cache[identity] = i
    return packed


def _input_array(cupy, value):
    # Host launch data stays on the host for validation to avoid a device
    # synchronization before the fused trace.
    xp = cupy if isinstance(value, cupy.ndarray) else np
    return xp, xp.asarray(value, dtype=xp.float64)


def trace_gpu_batch(surfaces_by_group, origins, directions, object_indices, *, record_path=False,
                    allow_signed_segments=False, return_device=False, device=0):
    """Trace shape (group, ray, XYZ) against equal-length surface sequences.

    Device mode returns CuPy buffers with integer reason codes in
    TRACE_REASONS order. Host mode uses the NumPy TraceResult conventions.
    Optional surface history has shape (surface, group, ray, XYZ).
    """
    cp = require_cupy(device)
    geometry = _pack_groups(surfaces_by_group)
    with cp.cuda.Device(device):
        origin_xp, origins = _input_array(cp, origins)
        direction_xp, directions = _input_array(cp, directions)
        index_xp, indices = _input_array(cp, object_indices)
        if (directions.ndim != 3 or directions.shape[2] != 3 or not directions.shape[1]
                or directions.shape[0] != len(geometry) or origins.shape != directions.shape):
            raise ValueError('Provide matched origins and directions with shape (group, ray, 3)')
        if (not bool(origin_xp.isfinite(origins).all())
                or not bool(direction_xp.isfinite(directions).all())):
            raise ValueError('Origins and directions must contain finite XYZ coordinates')
        if bool(direction_xp.any(directions[..., 2] <= 0.)):
            raise ValueError('This sequential tracer requires forward launch directions')
        if (indices.shape != (len(geometry),) or not bool(index_xp.isfinite(indices).all())
                or bool(index_xp.any(indices <= 0.))):
            raise ValueError('Supply one finite positive object index for each group')

        shape = directions.shape[:2]
        count = shape[0]*shape[1]
        points = cp.empty(directions.shape, dtype=cp.float64)
        outgoing = cp.empty_like(points)
        optical_path = cp.empty(shape, dtype=cp.float64)
        valid = cp.empty(shape, dtype=cp.bool_)
        failed = cp.empty(shape, dtype=cp.int32)
        reason = cp.empty(shape, dtype=cp.uint8)
        negative = cp.empty_like(failed)
        shortest = cp.empty_like(optical_path)
        shortest_at = cp.empty_like(failed)
        history = cp.full((geometry.shape[1], *directions.shape), cp.nan) if record_path else cp.empty(0)
        threads = 128
        _kernel(cp)(((count+threads-1)//threads,), (threads,), (
            cp.asarray(geometry), np.int32(geometry.shape[1]), np.int32(geometry.shape[2]),
            cp.asarray(indices, order='C'), cp.asarray(origins, order='C'), cp.asarray(directions, order='C'),
            np.int64(count), np.int32(shape[1]), np.int32(allow_signed_segments), np.int32(record_path),
            points, outgoing, optical_path, valid, failed, reason, negative, shortest, shortest_at, history,
        ))
        arrays = (points, outgoing, optical_path, valid, failed, reason)
        diagnostics = (history if record_path else None, negative, shortest, shortest_at)
        if return_device:
            return TraceResult(*arrays, *diagnostics)
        host = [cp.asnumpy(array) for array in arrays]
        host[5] = np.asarray(TRACE_REASONS, dtype='U16')[host[5]]
        return TraceResult(*host, *(None if array is None else cp.asnumpy(array) for array in diagnostics))


def trace_gpu(surfaces, origins, directions, object_index=1.0, *, record_path=False,
              allow_signed_segments=False, device=0):
    """GPU equivalent of standalone.trace for a single surface sequence."""
    origins = np.asarray(origins, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != 3 or not len(directions):
        raise ValueError('Directions must have shape (N, 3)')
    if origins.shape == (3,):
        origins = np.broadcast_to(origins, directions.shape)
    if origins.shape != directions.shape:
        raise ValueError('Origins and directions must contain finite XYZ coordinates')
    result = trace_gpu_batch((surfaces,), origins[None, ...], directions[None, ...], (object_index,),
                             record_path=record_path, allow_signed_segments=allow_signed_segments,
                             device=device)
    return TraceResult(result.points[0], result.directions[0], result.opl_mm[0], result.valid[0],
                       result.failed_surface[0], result.reason[0],
                       None if result.surface_points is None else result.surface_points[:, 0],
                       result.negative_segment_count[0], result.shortest_segment_mm[0],
                       result.shortest_segment_surface[0])
