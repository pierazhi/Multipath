import plotly.graph_objects as go
import trimesh
import numpy as np
from campaign import *
from cir import *
from constants import *
from diffraction import *
from em_core import *
from engine_hybrid import _generate_candidate_sequences
from engine_image import (
    _find_first_order_paths,
    _assemble_reflection_path,
    _backtrace_image_chain,
    _backtrace_image_chain_batch,
    _segments_clear,
)
from scenarios import choose_mesh
from engine_image_parallel import *
from engine_sbr import *
from geometry import *
from legacy import *
from terrain_io import *
from timing import *
from viz import *
import plotly.io as pio
import pickle
from polyscope_live import launch_live
pio.renderers.default = 'browser'

FREQUENCY = 2.4e9
EPSILON_LUNA = 2.87 - 0.01j

class Multipath:
    def __init__(
        self,
        mesh,

        # Engine choice
        engine="hybrid",

        # Antenna setup
        boresight_tx=np.array([0, 0, -1]),
        boresight_rx=np.array([0, 0, 1]),
        up_tx=np.array([0, 0, 1]),
        up_rx=np.array([0, 0, 1]),
        tx_pol="RHCP",
        rx_pol="RHCP",
        G_max_db=12.0,
        hpbw_h=30.0,
        hpbw_v=60.0,
        isotropic=False,
        pol_convention="fixed",
        polarization=True,

        # Propagation setup
        frequency=2.4e9,
        epsilon_luna=2.87 - 0.01j,

        # Solver setup
        num_rays=1e5,
        max_bounces=3,
        r_influence=0.1,
        launch_mode="pyramid",
        cone_aim_point=None,
        horizon_margin_deg=3.0,

        # Debug / output
        verbose=True,
    ):
        self.mesh = mesh

        self.engine = engine

        self.boresight_tx = boresight_tx
        self.boresight_rx = boresight_rx
        self.up_tx = up_tx
        self.up_rx = up_rx
        self.tx_pol = tx_pol
        self.rx_pol = rx_pol
        self.G_max_db = G_max_db
        self.hpbw_h = hpbw_h
        self.hpbw_v = hpbw_v
        self.isotropic = isotropic
        self.pol_convention = pol_convention
        self.polarization = polarization

        self.frequency = frequency
        self.epsilon_luna = epsilon_luna

        self.num_rays = num_rays
        self.max_bounces = max_bounces
        self.r_influence = r_influence
        self.launch_mode = launch_mode
        self.cone_aim_point = cone_aim_point
        self.horizon_margin_deg = horizon_margin_deg


        self.verbose = verbose

        self.results = None

    def select_engine(self):
        if self.engine == "sbr":
            return run_SBR

        if self.engine == "image":
            return run_image_method_engine

        if self.engine == "hybrid":
            return run_sbr_image_solver

        raise ValueError(f"Unknown engine: {self.engine}")
    
    def scene(self, pos_tx, pos_rx):
        visualize_mesh_3D(self.mesh, pos_tx, pos_rx, size = 3, show_axes=True, show_edges=True)

    def run(self):
        engine_fn = self._select_engine()

        self.results = engine_fn(self.mesh,
                                pos_tx,
                                pos_rx,

                                boresight_tx=self.boresight_tx,
                                boresight_rx=self.boresight_rx,
                                up_tx=self.up_tx,
                                up_rx=self.up_rx,

                                tx_pol=self.tx_pol,
                                rx_pol=self.rx_pol,

                                G_max_db=self.G_max_db,
                                hpbw_h=self.hpbw_h,
                                hpbw_v=self.hpbw_v,

                                isotropic=self.isotropic,
                                pol_convention=self.pol_convention,
                                polarization=self.polarization,

                                frequenza=self.frequency,
                                epsilon_luna=self.epsilon_luna,

                                num_rays=self.num_rays,
                                max_bounces=self.max_bounces,

                                launch_mode=self.launch_mode,
                                cone_aim_point=self.cone_aim_point,
                                horizon_margin_deg=self.horizon_margin_deg,

                                verbose=self.verbose,
                            )

    def vis_paths(self):
        visualize_with_plotly(self.mesh, self.results, pos_tx, pos_rx, self.r_influence, 
                              boresight_rx=self.boresight_rx, boresight_tx=self.boresight_tx, up_rx=self.up_rx, 
                              up_tx = self.up_tx)

mesh = choose_mesh("lunar")
pos_tx = np.array([1e3, 1e3, 1e3])
pos_rx = np.array([0, 0, 0])

scene = Multipath(mesh, engine="sbr")
scene.scene(pos_tx, pos_rx)
scene.run()
scene.vis_paths()






