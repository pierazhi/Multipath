from __future__ import annotations
import plotly.graph_objects as go
import trimesh
import numpy as np
from campaign import *
from cir import *
from constants import *
from diffraction import *
from em_core import *
from engine_hybrid import *
from engine_image import *
from engine_image_parallel import *
from engine_sbr import *
from geometry import *
from legacy import *
from terrain_io import *
from timing import *
from viz import *
from scenarios import *
from orbit_helpers import *
import plotly.io as pio
import pickle
from polyscope_live import launch_live
from terrain_class import MultiPath
from pathlib import Path
import numpy as np
import warp as wp
from constants import SELF_INTERSECT_EPSILON

BASE_DIR = Path.cwd().resolve().parent
tifs_dir = BASE_DIR / "Multipath" / "tifs_new"
kernels_dir = BASE_DIR / "Multipath" / "kernels"
meshes_dir = BASE_DIR / "Multipath" / "meshes"

wp.init()

_WARP_MESH_CACHE: dict[int, "wp.Mesh"] = {}   # id(trimesh_mesh) -> wp.Mesh, mirrors the plan in memory


def _get_warp_mesh(mesh) -> "wp.Mesh":
    key = id(mesh)
    wm = _WARP_MESH_CACHE.get(key)
    if wm is not None:
        return wm
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    tris  = np.asarray(mesh.faces, dtype=np.int32).reshape(-1)
    wm = wp.Mesh(points=wp.array(verts, dtype=wp.vec3),
                 indices=wp.array(tris, dtype=wp.int32))
    _WARP_MESH_CACHE[key] = wm
    return wm


@wp.kernel
def _ray_kernel(
    mesh_id:  wp.uint64,
    origins:  wp.array(dtype=wp.vec3),   # already marched past SELF_INTERSECT_EPSILON
    dirs:     wp.array(dtype=wp.vec3),
    max_dist: float,
    hit_mask: wp.array(dtype=wp.int32),
    hit_pt:   wp.array(dtype=wp.vec3),
    hit_face: wp.array(dtype=wp.int32),
    hit_dist: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    o = origins[tid]
    d = dirs[tid]
    t = float(0.0); u = float(0.0); v = float(0.0)
    sign = float(0.0); n = wp.vec3(); face = int(-1)

    if wp.mesh_query_ray(mesh_id, o, d, max_dist, t, u, v, sign, n, face):
        hit_mask[tid] = 1
        hit_dist[tid] = t
        hit_face[tid] = face
        hit_pt[tid]   = o + d * t
    else:
        hit_mask[tid] = 0
        hit_dist[tid] = max_dist
        hit_face[tid] = -1
        hit_pt[tid]   = o + d * max_dist


def batch_intersect_warp(mesh, adv, dirs64, max_dist):
    """adv, dirs64: already bounding-sphere-advanced, float64, as computed by the
    existing preamble in geometry.batch_intersect(). We additionally march past
    SELF_INTERSECT_EPSILON here, since mesh_query_ray has no min-t to filter with."""
    N = len(adv)
    wm = _get_warp_mesh(mesh)

    march = float(SELF_INTERSECT_EPSILON) * 2.0
    o = (adv + dirs64 * march).astype(np.float32)

    origins_wp = wp.array(np.ascontiguousarray(o), dtype=wp.vec3)
    dirs_wp    = wp.array(np.ascontiguousarray(dirs64.astype(np.float32)), dtype=wp.vec3)
    hit_mask_wp = wp.zeros(N, dtype=wp.int32)
    hit_pt_wp   = wp.zeros(N, dtype=wp.vec3)
    hit_face_wp = wp.zeros(N, dtype=wp.int32)
    hit_dist_wp = wp.zeros(N, dtype=wp.float32)

    wp.launch(_ray_kernel, dim=N,
              inputs=[wm.id, origins_wp, dirs_wp, float(max_dist) - march],
              outputs=[hit_mask_wp, hit_pt_wp, hit_face_wp, hit_dist_wp])

    hit_mask  = hit_mask_wp.numpy().astype(bool)
    hit_dists = hit_dist_wp.numpy().astype(np.float64) + march   # re-express from adv, not the marched origin
    hit_pts   = hit_pt_wp.numpy().astype(np.float64)
    hit_faces = hit_face_wp.numpy()
    return hit_mask, hit_pts, hit_faces, hit_dists
pio.renderers.default = 'browser'
