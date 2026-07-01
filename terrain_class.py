import plotly.graph_objects as go
import trimesh
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from pyproj import Transformer
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
import plotly.io as pio
from pyproj import CRS
import spiceypy as spice
import pickle
from polyscope_live import launch_live
pio.renderers.default = 'browser'
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

@dataclass
class Antenna:
    polarization: Any = "RHCP"
    boresight: Optional[np.ndarray] = None     # (3,) in the ACTIVE frame
    pattern: Any = None
    up: Optional[np.ndarray] = None            # NEW: local +Z reference

    def __post_init__(self):
        if self.boresight is not None:
            b = np.asarray(self.boresight, float); n = np.linalg.norm(b)
            if n == 0: raise ValueError("boresight must be non-zero.")
            self.boresight = b / n
        if self.up is None:
            self.up = np.array([0.0, 0.0, 1.0])    # antenna_frame self-corrects if ∥ boresight
        else:
            u = np.asarray(self.up, float); self.up = u / np.linalg.norm(u)

class MultiPath:
    def __init__(self, dem_path=None, utc_time="2026-06-22T00:00:00", frame="local", 
                 num_rays=500_000, max_bounces = 3, launch_mode = "pyramid",
                 polarization = True, isotropic = False, verbose = True, 
                 pol_convention = "fixed", occlusion = "batched", g_max_db = 12, 
                 hpbw_h = 30, hpbw_v = 30, enable_profiling = False):
        self.dem_path = dem_path          # was None — this broke crop()/gen_mesh()
        self.utc_time = utc_time
        self.frame = frame
        self.dem_crs = None
        if dem_path is not None:          # allow scenario-only construction, no .tif needed
            with rasterio.open(dem_path) as src:
                self.dem_crs = CRS.from_user_input(src.crs)
        self.results = None
        self.cropped_path = None
        self.bounds_m = None
        self.mesh = None                  # was missing — see below
        self.mesh_pa = None
        self.mesh_local = None
        self.origin_pa = None
        self.A_local_to_pa = None
        self.pos_tx = np.empty((0, 3))
        self.pos_rx = np.empty((0, 3))
        self.tx_ant = []
        self.rx_ant = []
        self.frequency   = 2.4e9
        self.epsilon     = 2.87 - 0.01j
        self.num_rays    = num_rays
        self.max_bounces = max_bounces
        self.g_max_db    = g_max_db
        self.hpbw_h      = hpbw_h
        self.hpbw_v      = hpbw_v
        self.launch_mode = launch_mode
        self.polarization = polarization        # antenna-mismatch loss on/off
        self.isotropic = isotropic
        self.verbose = verbose
        self.pol_convention = pol_convention
        self.occlusion = occlusion
        self.enable_profiling = enable_profiling

    def crop(self, output_path, method, **kwargs):
        self.bounds_m = selection_to_bounds(self.dem_path, method, **kwargs)
        crop_dem_by_bounds(self.dem_path, output_path, self.bounds_m)
        self.cropped_path = output_path

    def gen_mesh(self, target_resolution_m):
        src_path = self.cropped_path or self.dem_path
        self.mesh_pa, _ = dem_to_pa_mesh(
            src_path, target_resolution_m=target_resolution_m, utc_time=self.utc_time,
        )
        self.origin_pa, self.A_local_to_pa = build_local_frame_from_pa_mesh(self.mesh_pa)
        self.mesh_local = mesh_pa_to_local(self.mesh_pa, self.origin_pa, self.A_local_to_pa)
        self.mesh = self.mesh_pa if self.frame == "pa" else self.mesh_local
    
    def _dir_to_frame(self, vec, coords, target=None):
        """Convert a DIRECTION (boresight, up) between PA and local.
        Rotation only — no origin offset. Map projections (latlon,
        stereographic) are position-only and invalid for directions."""
        target = target or self.frame
        vec = np.asarray(vec, dtype=float)

        if coords in ("latlon", "stereographic"):
            raise ValueError(f"coords={coords!r} is a position projection, not a "
                             f"direction frame — give a boresight in 'pa' or 'local'.")
        if coords == target:
            return vec
        if coords == "pa" and target == "local":
            return vec @ self.A_local_to_pa
        if coords == "local" and target == "pa":
            return vec @ self.A_local_to_pa.T
        raise ValueError(f"Cannot convert direction coords={coords!r} into frame={target!r}.")

    def _center_to_frame(self, center, coords, target=None):
        target = target or self.frame
        center = np.asarray(center, dtype=float)

        if coords == "latlon":
            x_m, y_m = latlon_to_dem_xy(center[0], center[1], self.dem_crs)
            center = np.array([x_m, y_m], dtype=float)
            coords = "stereographic"

        if coords == target:
            return center
        if coords == "stereographic":
            if target == "local":
                p, _ = stereographic_xy_to_local(center[0], center[1], self.dem_crs,
                                                self.origin_pa, self.A_local_to_pa, self.utc_time)
            else:
                p, _ = stereographic_xy_to_pa(center[0], center[1], self.dem_crs, self.utc_time)
            return p
        if coords == "pa" and target == "local":
            return pa_to_local(center, self.origin_pa, self.A_local_to_pa)
        if coords == "local" and target == "pa":
            return local_to_pa(center, self.origin_pa, self.A_local_to_pa)

        raise ValueError(f"Cannot convert coords={coords!r} into frame={target!r}.")
    
    def _resolve_nodes(self, center, n, spacing, height, mode, bounded, coords):
        # ── off-terrain absolute grid: a free-floating n×n plane in space ──
        if mode == "absolute":
            center = np.asarray(center, dtype=float)
            if center.shape != (3,):
                raise ValueError("absolute mode needs a full 3D center (x, y, z).")
            if coords not in ("pa", "local"):
                raise ValueError("absolute grids need coords='pa' or 'local'; "
                                "stereographic/latlon are 2D projections with no "
                                "defined plane.")
            # in-plane orthonormal basis — for now the input frame's X̂, Ŷ.
            # Future: swap these two for (v̂, r̂×v̂) from the orbit.
            e1 = np.array([1.0, 0.0, 0.0])
            e2 = np.array([0.0, 1.0, 0.0])
            half = (n - 1) / 2
            offs = (np.arange(n) - half) * spacing
            du, dv = np.meshgrid(offs, offs)
            grid = (center
                    + du.ravel()[:, None] * e1
                    + dv.ravel()[:, None] * e2)          # (n², 3) in `coords`
            nodes_local = self._center_to_frame(grid, coords, target="local")
        else:
            center_local = self._center_to_frame(center, coords, target="local")
            nodes_local = np.asarray(
                generate_grid_nodes(self.mesh_local, center_local[:2], n=n, spacing=spacing,
                                    height=height, mode=mode, bounded=bounded)
            ).reshape(-1, 3)

        if self.frame == "pa":
            return local_to_pa(nodes_local, self.origin_pa, self.A_local_to_pa)
        return nodes_local

    def add_tx(self, center, n=1, spacing=200, height=10.0, mode="terrain",
           bounded=False, coords="stereographic",
           polarization="RHCP", boresight=(0.0, 0.0, 1.0), pattern=None, boresight_coords="local", up=(0.0, 0.0, 1.0)):
           nodes = self._resolve_nodes(center, n, spacing, height, mode, bounded, coords)
           b = self._dir_to_frame(boresight, boresight_coords)        # ← rotate into active frame
           u = self._dir_to_frame(up,        boresight_coords)        # up shares the antenna frame
           ant = Antenna(polarization, b, self._make_pattern(pattern), up=u)   # lines 164 and 174
           self.pos_tx = np.vstack([self.pos_tx, nodes])
           self.tx_ant.extend([ant] * len(nodes))   # one spec, broadcast to every node in this call

    def add_rx(self, center, n=1, spacing=200, height=10.0, mode="terrain",
           bounded=False, coords="stereographic",
           polarization="RHCP", boresight=(0.0, 0.0, 1.0), pattern=None, boresight_coords="local", up=(0.0, 0.0, 1.0)):
           nodes = self._resolve_nodes(center, n, spacing, height, mode, bounded, coords)
           b = self._dir_to_frame(boresight, boresight_coords)        # ← rotate into active frame
           u = self._dir_to_frame(up,        boresight_coords)        # up shares the antenna frame
           ant = Antenna(polarization, b, self._make_pattern(pattern), up=u)   # lines 164 and 174
           self.pos_rx = np.vstack([self.pos_rx, nodes])
           self.rx_ant.extend([ant] * len(nodes))   # one spec, broadcast to every node in this call


    def _finalize_local(self, mesh_local):
        """Adopt a mesh already in a local (+Z-up) frame — e.g. a synthetic
        scenario. No .tif, no PA representation."""
        self.frame = "local"            # a dummy has no PA, so force local
        self.mesh_local = mesh_local
        self.mesh = mesh_local
        self.mesh_pa = None
        self.origin_pa = None
        self.A_local_to_pa = None
        self.dem_path = None

    def inspect(self, verbose=True):
        """Report the mesh's footprint. X/Y in km, vertical relief in m."""
        if self.mesh is None:
            raise RuntimeError("No mesh — call gen_mesh() or load_scenario() first.")

        dx, dy, dz = self.mesh.extents          # metres (max-min per axis)
        info = {
            "x_km":       dx / 1000.0,
            "y_km":       dy / 1000.0,
            "relief_m":   dz,                   # vertical span stays in metres
            "n_vertices": len(self.mesh.vertices),
            "n_faces":    len(self.mesh.faces),
        }
        if verbose:
            print(f"Mesh ({self.frame} frame): "
                  f"{info['x_km']:.2f} × {info['y_km']:.2f} km  "
                  f"| relief {info['relief_m']:.1f} m  "
                  f"| {info['n_faces']:,} faces")
        return info
    
    def load_scenario(self, case):
        self._finalize_local(choose_mesh(case))

    def _antenna_kwargs(self, ant, role):
        """Translate one Antenna into the engine's per-end kwargs. role: 'tx'|'rx'."""
        return {
            f"boresight_{role}": ant.boresight if ant.boresight is not None else np.array([0.0, 0.0, 1.0]),
            f"up_{role}":        ant.up,
            f"{role}_pol":       ant.polarization,
            f"pattern_{role}":   ant.pattern,
        }
    def path_solver(self, i_tx, i_rx, plot, long_distance, detailed, show_edges = False):
        self.results = self._link_setting(i_tx, i_rx)
        if plot:
            visualize_with_plotly(
                self.mesh, self.results, self.pos_tx[i_tx], self.pos_rx[i_rx], r_influence=0,
                boresight_tx=self.tx_ant[i_tx].boresight, boresight_rx=self.rx_ant[i_rx].boresight,
                up_tx=self.tx_ant[i_tx].up, up_rx=self.rx_ant[i_rx].up,
                long_distance=long_distance, show_edges=show_edges,
            )
        if detailed:
            print_full_ray_history_2(self.results)

    def _link_setting(self, i_tx, i_rx, **overrides):
        """Solve ONE TX->RX link with the hybrid engine. Returns the results dict
        (the same schema compute_cir_from_results expects)."""
        if self.mesh is None:
            raise RuntimeError("No mesh — call gen_mesh() or load_scenario() first.")

        kwargs = dict(
            mesh=self.mesh,
            pos_tx=self.pos_tx[i_tx],
            pos_rx=self.pos_rx[i_rx],
            frequenza=self.frequency,
            epsilon_luna=self.epsilon,
            num_rays=self.num_rays,
            max_bounces=self.max_bounces,
            G_max_db=self.g_max_db, hpbw_h=self.hpbw_h, hpbw_v=self.hpbw_v,
            launch_mode=self.launch_mode,
            polarization=self.polarization,
            isotropic = self.isotropic,
            verbose = self.verbose,
            enable_profiling = self.enable_profiling,
        )
        kwargs.update(self._antenna_kwargs(self.tx_ant[i_tx], "tx"))
        kwargs.update(self._antenna_kwargs(self.rx_ant[i_rx], "rx"))
        kwargs.update(overrides)        # per-call escape hatch, e.g. solve_link(0,0, max_bounces=3)

        return run_sbr_image_solver(**kwargs)
    
    def _make_pattern(self, spec):
        if spec is None:
            def default(theta, phi):
                return antenna_gain_pattern_2d(theta, phi, self.g_max_db, self.hpbw_h, self.hpbw_v)
            return default
        if spec == "iso":
            def iso(theta, phi): return 0.0
            return iso
        if spec == "galileo":
            return pattern_from_grap(r"C:\Users\Luna\Documents\MP\Multipath\GRAP_metadata\GRAP_File_E1_.xlsx")
        if isinstance(spec, str) and spec.endswith(".csv"):
            return pattern_from_csv(spec)
        raise ValueError(f"Unknown pattern: {spec!r}")


# pat = pattern_from_csv(r"C:\Users\Luna\Documents\MP\Multipath\Refactored\tiny_pattern.csv")
# print(pat(0, 0))

# input_path = r"C:\Users\Luna\Documents\MP\tifs_new\LDEM_875S_20M_cropped_3km.tif"
# t1 = MultiPath(input_path, frame = "local", max_bounces=3, num_rays=500_000, launch_mode="full_sphere")
# t1.gen_mesh(target_resolution_m=80)

# t1.inspect()
# t1.add_tx([t1.mesh.centroid[0] + 1e3, t1.mesh.centroid[1] + 1e3], n = 2, height = 1e3, spacing=2000, mode = "plane", bounded = False, coords="local", polarization="RHCP")
# t1.add_rx([t1.mesh.centroid[0], t1.mesh.centroid[1]], n = 2, height = 1.2, spacing=300, mode = "terrain", bounded = True, coords="local", polarization="RHCP")
# visualize_mesh_3D(t1.mesh, t1.pos_tx, t1.pos_rx, show_edges=True, vis_map=False)


# t1.path_solver(1, 3, plot=True)

# print("\n\n")
# t2 = MultiPath(frame = "local", max_bounces=3, num_rays=500_000, launch_mode="pyramid")
# t2.load_scenario("lunar")
# t2.inspect()
# t2.add_tx([t2.mesh.centroid[0] + 1e3, t2.mesh.centroid[1] + 1e3], n = 2, height = 1e3, spacing=2000, mode = "plane", bounded = False, coords="local", polarization="RHCP")
# t2.add_rx([t2.mesh.centroid[0], t2.mesh.centroid[1]], n = 2, height = 1.2, spacing=300, mode = "terrain", bounded = True, coords="local", polarization="RHCP")

# t2.path_solver(1, 3, plot=True, long_distance=False)

print("Class Loaded")