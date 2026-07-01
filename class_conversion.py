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

class HybridSolver:
    def __init__(
        self,
        frequency=2.4e9,
        epsilon_luna=2.87 - 0.01j,
        num_rays=1_000_000,
        max_bounces=2,
        launch_mode="pyramid",
        occlusion="batched",
        isotropic=True,
        verbose=True,
        tx_pol="RHCP",
        rx_pol="LHCP",
        polarization=False,
        pol_convention="fixed",
        exact_first_order=True,
        nee_to_rx=True,
        up_tx=None,
        up_rx=None,
        G_max_db=12.0,
        hpbw_h=75.0,
        hpbw_v=75.0,
        enable_profiling=False,
        debug_ray_direction=None,
        cone_aim_point=None,
        horizon_margin_deg=3.0,
        max_path_distance_factor=5.0,
        scenario = "lunar",
        vis_map = False
    ):
        self.frequency = frequency
        self.epsilon_luna = epsilon_luna
        self.num_rays = int(num_rays)
        self.max_bounces = max_bounces
        self.launch_mode = launch_mode
        self.occlusion = occlusion
        self.isotropic = isotropic
        self.polarization = polarization
        self.verbose = verbose
        self.tx_pol = tx_pol
        self.rx_pol = rx_pol
        self.pol_convention = pol_convention
        self.exact_first_order = exact_first_order
        self.nee_to_rx = nee_to_rx

        self.up_tx = np.array([0.0, 0.0, 1.0]) if up_tx is None else np.asarray(up_tx, dtype=np.float64)
        self.up_rx = np.array([0.0, 0.0, 1.0]) if up_rx is None else np.asarray(up_rx, dtype=np.float64)

        self.G_max_db = G_max_db
        self.hpbw_h = hpbw_h
        self.hpbw_v = hpbw_v
        self.enable_profiling = enable_profiling
        self.debug_ray_direction = debug_ray_direction
        self.cone_aim_point = cone_aim_point
        self.horizon_margin_deg = horizon_margin_deg
        self.max_path_distance_factor = max_path_distance_factor

        self.c = 299_792_458.0
        self.wavelength = self.c / self.frequency
        self.k = 2.0 * np.pi / self.wavelength
        self.scenario = scenario
        self.vis_map = vis_map

    def _pre(self, pos_tx, pos_rx, scenario):

        self.mesh = choose_mesh(scenario)

        if self.occlusion not in ("serial", "batched"):
            raise ValueError(f"occlusion must be 'serial' or 'batched', got {self.occlusion!r}")

        self.pos_tx = np.asarray(pos_tx, dtype=np.float64)
        self.pos_rx = np.asarray(pos_rx, dtype=np.float64)

        sep = float(np.linalg.norm(self.pos_rx - self.pos_tx))
        self.max_path_distance = self.max_path_distance_factor * sep

        boresight_tx = pos_rx - pos_tx # Il TX punta verso l'RX
        boresight_rx = pos_tx - pos_rx # L'RX punta verso il TX

        # ── Antenna frames / polarization (identical setup to the other engines)
        self.boresight_tx = self.pos_rx - self.pos_tx
        self.boresight_rx = self.pos_tx - self.pos_rx

        self.boresight_tx /= np.linalg.norm(self.boresight_tx)
        self.boresight_rx /= np.linalg.norm(self.boresight_rx)
        up_tx = np.asarray(self.up_tx, dtype=np.float64); self.up_tx /= np.linalg.norm(self.up_tx)
        up_rx = np.asarray(self.up_rx, dtype=np.float64); self.up_rx /= np.linalg.norm(self.up_rx)

        self.h_tx_ant, self.v_tx_ant = antenna_frame(self.boresight_tx, self.up_tx)
        self.h_rx_ant, self.v_rx_ant = antenna_frame(self.boresight_rx, self.up_rx)

        self.J_tx = jones_vector(self.tx_pol)
        self.J_rx = jones_vector(self.rx_pol)
        self.J_rx_rotated = np.conj(self.J_rx).T @ np.array([[-1, 0], [0, 1]])

        # EM kwargs bundle for the refiner (kept identical to run_image_method_engine)
        self.em = dict(
            mesh=self.mesh, pos_tx=pos_tx, pos_rx=pos_rx,
            wavelength=self.wavelength, k=self.k, epsilon=self.epsilon_luna,
            J_tx=self.J_tx, J_rx_rotated=self.J_rx_rotated,
            h_tx_ant=self.h_tx_ant, v_tx_ant=self.v_tx_ant, h_rx_ant=self.h_rx_ant, v_rx_ant=self.v_rx_ant,
            boresight_tx=boresight_tx, boresight_rx=boresight_rx, up_tx=up_tx,
            G_max_db=self.G_max_db, hpbw_h=self.hpbw_h, hpbw_v=self.hpbw_v, isotropic=self.isotropic,
            J_rx=self.J_rx, up_rx=up_rx, pol_convention=self.pol_convention,
            polarization=self.polarization,
        )

        self.results = {
            "los_segment":           None,
            "los_blocked":           False,
            "los_paths":             [],
            "nlos_paths":            [],
            "keller_events":         [],
            "primary_rays_captured": 0,
            "r_influence_mode":      "hybrid_sbr_image",
        }

        self.ray_id = 0
        self.found_per_order: dict = {}
        self.candidates = set()
        self.timer = SectionTimer()

    def _visual(self):
        visualize_mesh_3D(self.mesh, pos_tx=self.pos_tx, pos_rx=self.pos_rx, size = 3, show_axes=True, show_edges=True, vis_map=self.vis_map)

    def los_path(self):
        with self.timer("los_check"):
            los_vec  = self.pos_rx - self.pos_tx
            los_dist = float(np.linalg.norm(los_vec))
            los_dir  = los_vec / los_dist
            los_blocked, blocking_pt = check_los(self.mesh, self.pos_tx, self.pos_rx)

        if los_blocked:
            self.results["los_blocked"] = True
            self.results["los_segment"] = np.vstack((self.pos_tx, blocking_pt))
            if self.verbose:
                print(f"\n[LOS INFO] TX and RX are NOT in Line-of-Sight! "
                    f"Blocked at point: {np.round(blocking_pt, 2)}")
        else:
            self.results["los_segment"] = np.vstack((self.pos_tx, self.pos_rx))
            amp_los   = self.wavelength / (4.0 * np.pi * los_dist)
            fsl_db    = -20.0 * np.log10(amp_los)
            fase_geo  = -self.k * los_dist
            fase_norm = fase_geo % (2.0 * np.pi)

            if self.pol_convention == "sionna":
                h_tx_los, v_tx_los = sionna_hv_basis(ray_dir=los_dir, boresight=self.boresight_tx, up=self.up_tx)
                E_los = self.J_tx[0] * h_tx_los + self.J_tx[1] * v_tx_los
                h_rx_los, v_rx_los = sionna_hv_basis(ray_dir=-los_dir, boresight=self.boresight_rx, up=self.up_rx)
                e_rx_los = self.J_rx[0] * h_rx_los + self.J_rx[1] * v_rx_los
                pol_factor_los = complex(np.vdot(e_rx_los, E_los))
            else:
                h_ray_los, v_ray_los = antenna_frame(los_dir, self.up_tx)
                R_los = np.array([
                    [np.dot(h_ray_los, self.h_rx_ant), np.dot(v_ray_los, self.h_rx_ant)],
                    [np.dot(h_ray_los, self.v_rx_ant), np.dot(v_ray_los, self.v_rx_ant)],
                ])
                pol_factor_los = complex(np.atleast_1d(self.J_rx_rotated @ R_los @ self.J_tx)[0])
            if not self.polarization:
                _d = abs(pol_factor_los)
                pol_factor_los = pol_factor_los / _d if _d > 1e-12 else complex(1.0)
            IL_los_db = -20.0 * np.log10(np.clip(abs(pol_factor_los), 1e-10, 1.0))

            theta_tx_los, phi_tx_los = get_antenna_local_angles(los_dir, self.boresight_tx, self.h_tx_ant, self.v_tx_ant)
            gain_tx_los = antenna_gain_pattern_2d(theta_tx_los, phi_tx_los, self.G_max_db, self.hpbw_h, self.hpbw_v)
            theta_rx_los, phi_rx_los = get_antenna_local_angles(-los_dir, self.boresight_rx, self.h_rx_ant, self.v_rx_ant)
            gain_rx_los = antenna_gain_pattern_2d(theta_rx_los, phi_rx_los, self.G_max_db, self.hpbw_h, self.hpbw_v)

            if self.isotropic:
                gain_tx_los = 0.0; gain_rx_los = 0.0

            g_field_los    = 10.0 ** ((gain_tx_los + gain_rx_los) / 20.0)
            campo_los      = amp_los * g_field_los * pol_factor_los * np.exp(1j * fase_geo)
            total_loss_los = fsl_db + IL_los_db - gain_tx_los - gain_rx_los

            aoa_dir_los = -los_dir
            _ang_los = compute_link_angles(
                los_dir, los_dir,
                self.boresight_tx, self.h_tx_ant, self.v_tx_ant,
                self.boresight_rx, self.h_rx_ant, self.v_rx_ant,
            )

            if self.verbose:
                print(
                    f"\n[LOS INFO] LoS CLEAR | "
                    f"Dist: {los_dist:.2f} m | FSL: {fsl_db:.2f} dB | "
                    f"Pol Loss: {IL_los_db:.2f} dB | "
                    f"Gain TX: {gain_tx_los:.2f} dB | Gain RX: {gain_rx_los:.2f} dB | "
                    f"Total Loss: {total_loss_los:.2f} dB"
                )

            self.results["primary_rays_captured"] = 1
            self.results["nlos_paths"].append({
                "ray_id": -1, "bounces": 0, "has_diffracted": False, "faces_hit": [],
                "points": np.array([self.pos_tx, self.pos_rx]), "distanza_totale": los_dist,
                "campo_complesso": campo_los, "fsl_db": fsl_db, "IL_db": float(IL_los_db),
                "gain_tx_db": gain_tx_los, "gain_rx_db": gain_rx_los,
                "reflection_loss_db": 0.0, "diffraction_loss_db": 0.0,
                "path_loss_totale_db": total_loss_los,
                "sfasamento_totale_rad": fase_norm, "sfasamento_totale_deg": np.degrees(fase_norm),
                "ritardo_assoluto_ns": (los_dist / self.c) * 1e9,
                "aoa_direction": aoa_dir_los, **_ang_los,
                "theta_tx_deg": theta_tx_los, "phi_tx_deg": phi_tx_los,
                "theta_rx_deg": theta_rx_los, "phi_rx_deg": phi_rx_los,
                "dettaglio_rimbalzi": [], "r_influence_used": None,
            })

    # ── 2. ORDER 1 — exact, fully vectorized, NO SBR ──────────────────────
    # At a single bounce there is no combinatorial explosion to prune, so SBR
    # is pure overhead. `_find_first_order_paths` mirrors TX across every facet,
    # solves the specular point, and validates both legs in batched calls — it
    # *is* the exact first-bounce image method. Its output is already validated
    # (point-in-triangle + clear_in + clear_out), so we assemble straight from
    # it: no per-candidate `_backtrace_image_chain`, no re-running `_segments_clear`.
    
    def _order_1(self, ):
        if self.exact_first_order and self.max_bounces >= 1:
            with self.timer("order1_vectorized"):
                o1 = _find_first_order_paths(self.mesh, self.pos_tx, self.pos_rx, np.arange(len(self.mesh.faces)))
                for pts, fseq in o1:
                    d = _assemble_reflection_path(pts, list(fseq), self.ray_id, **self.em)
                    if d is None or d["distanza_totale"] > self.max_path_distance:
                        continue
                    d.setdefault("membro_raggruppamento", [self.ray_id])
                    d.setdefault("potenza_incoerente", abs(d["campo_complesso"]) ** 2)
                    self.found_per_order[1] = self.found_per_order.get(1, 0) + 1
                    if self.verbose:
                        ray_label = "[K]" if d["has_diffracted"] else "[P]"
                        print(
                            f"Ray ID #{self.ray_id:5d} | Bounces: {d['bounces']} | {ray_label} | "
                            f"Dist: {d['distanza_totale']:.2f} m | FSL: {d['fsl_db']:.2f} dB | "
                            f"Refl: {d['reflection_loss_db']:.2f} dB | Diff: {d['diffraction_loss_db']:.2f} dB | "
                            f"Pol Loss: {d['pol_loss']:.2f} | "
                            f"IL: {d['IL_db']:.2f} dB | "
                            f"Gain TX: {d['gain_tx_db']:.2f} dB | Gain RX: {d['gain_rx_db']:.2f} dB | "
                            f"Total: {d['path_loss_totale_db']:.2f} dB | Fase: {d['sfasamento_totale_deg']:.2f}°"
                        )
                    self.results["nlos_paths"].append(d)
                    self.ray_id += 1

    # ── 3. ORDER >= 2 — SBR discovers sequences, image method refines ─────
    # Only here does SBR earn its keep: it prunes the F^order sequence space
    # down to the handful actually traversed. Skipped entirely for max_bounces<2.
    def _order_n(self):
        if self.max_bounces >= 2:
            with self.timer("launch"):
                if self.debug_ray_direction is not None:
                    d = np.asarray(self.debug_ray_direction, dtype=np.float64)
                    launch_dirs = (d / np.linalg.norm(d)).reshape(1, 3).astype(np.float32)
                    n_eff, omega = 1, 0.0
                elif self.launch_mode == "pyramid":
                    launch_dirs, n_eff, omega = generate_pyramid_directions(
                        self.pos_tx, self.num_rays, self.mesh, self.cone_aim_point
                    )
                else:
                    launch_dirs, n_eff, omega = generate_launch_directions(
                        self.pos_tx, self.num_rays, mesh=self.mesh, mode=self.launch_mode,
                        horizon_margin_deg=self.horizon_margin_deg,
                    )
                if self.verbose:
                    print(f"\n--- Hybrid solver | launch={self.launch_mode} | "
                        f"{len(launch_dirs)} rays (n_eff={n_eff}, Ω={omega:.3f} sr) | "
                        f"max_bounces={self.max_bounces} ---")

            with self.timer("candidate_gen"):
                all_seqs = _generate_candidate_sequences(
                    self.mesh, self.pos_tx, self.pos_rx, launch_dirs,
                    max_bounces=self.max_bounces, max_path_distance=self.max_path_distance,
                    nee_to_rx=self.nee_to_rx, verbose=self.verbose,
                )
                # Order 1 is owned by the exact vectorized pass above; only refine >=2.
                self.candidates = {s for s in all_seqs if len(s) >= 2}

            if self.occlusion == "batched":
                # Same physics and same surviving paths as the serial branch:
                # one backtrace per order, then a SINGLE segments-clear over every
                # leg of every candidate. Only the ray_id labelling order differs
                # (set iteration), which does not affect received power.
                from collections import defaultdict
                with self.timer("refine_A_backtrace"):
                    by_order = defaultdict(list)
                    for seq in self.candidates:
                        by_order[len(seq)].append(seq)

                    valid: list = []                # (pts, seq) per surviving candidate
                    leg_start_chunks, leg_end_chunks   = [], []
                    leg_nudge_chunks, leg_owner_chunks = [], []
                    owner_base = 0

                    for K in sorted(by_order):
                        seqs_arr = np.asarray(by_order[K], dtype=np.int64)       # (nK, K)
                        pts_b, ok = _backtrace_image_chain_batch(self.mesh, self.pos_tx, self.pos_rx, seqs_arr)
                        keep = np.nonzero(ok)[0]
                        if keep.size == 0:
                            continue
                        pts_k  = pts_b[keep]                                     # (m, K+2, 3)
                        seqs_k = seqs_arr[keep]                                  # (m, K)
                        m = keep.size

                        starts = pts_k[:, :-1, :]                               # (m, K+1, 3)
                        ends   = pts_k[:, 1:, :]
                        nud = np.zeros_like(starts)
                        nud[:, 1:K + 1, :] = self.mesh.face_normals[seqs_k]
                        owners = owner_base + np.repeat(np.arange(m), K + 1)

                        leg_start_chunks.append(starts.reshape(-1, 3))
                        leg_end_chunks.append(ends.reshape(-1, 3))
                        leg_nudge_chunks.append(nud.reshape(-1, 3))
                        leg_owner_chunks.append(owners.astype(np.int64))
                        for j in range(m):
                            valid.append((pts_k[j], tuple(int(x) for x in seqs_k[j])))
                        owner_base += m

                with self.timer("refine_B_occlusion"):
                    if valid:
                        all_starts = np.concatenate(leg_start_chunks, axis=0)
                        all_ends   = np.concatenate(leg_end_chunks,   axis=0)
                        all_nudges = np.concatenate(leg_nudge_chunks, axis=0)
                        all_owner  = np.concatenate(leg_owner_chunks, axis=0)
                        leg_clear = _segments_clear(self.mesh, all_starts, all_ends, nudge_normals=all_nudges)
                        cand_ok = np.ones(len(valid), dtype=bool)
                        np.logical_and.at(cand_ok, all_owner, leg_clear)   # survives iff ALL legs clear
                    else:
                        cand_ok = np.zeros(0, dtype=bool)

                with self.timer("refine_C_assemble"):
                    for ci, (pts, seq) in enumerate(valid):
                        if not cand_ok[ci]:
                            continue
                        K = len(seq)
                        d = _assemble_reflection_path(pts, list(seq), self.ray_id, **self.em)
                        if d is None or d["distanza_totale"] > self.max_path_distance:
                            continue
                        d.setdefault("membro_raggruppamento", [self.ray_id])
                        d.setdefault("potenza_incoerente", abs(d["campo_complesso"]) ** 2)
                        self.found_per_order[K] = self.found_per_order.get(K, 0) + 1
                        if self.verbose:
                            ray_label = "[K]" if d["has_diffracted"] else "[P]"
                            print(
                                f"Ray ID #{self.ray_id:5d} | Bounces: {d['bounces']} | {ray_label} | "
                                f"Dist: {d['distanza_totale']:.2f} m | FSL: {d['fsl_db']:.2f} dB | "
                                f"Refl: {d['reflection_loss_db']:.2f} dB | Diff: {d['diffraction_loss_db']:.2f} dB | "
                                f"Pol Loss: {d['pol_loss']:.2f} | "
                                f"IL: {d['IL_db']:.2f} dB | "
                                f"Gain TX: {d['gain_tx_db']:.2f} dB | Gain RX: {d['gain_rx_db']:.2f} dB | "
                                f"Total: {d['path_loss_totale_db']:.2f} dB | Fase: {d['sfasamento_totale_deg']:.2f}°"
                            )
                        self.results["nlos_paths"].append(d)
                        self.ray_id += 1
            else:
                with self.timer("image_refine"):
                    for seq in sorted(self.candidates, key=lambda s: (len(s), s)):
                        K = len(seq)

                        # Exact specular vertices for this facet chain (or None if invalid).
                        pts = _backtrace_image_chain(self.mesh, self.pos_tx, self.pos_rx, seq)
                        if pts is None:
                            continue

                        # Per-leg occlusion validation (TX->P1, P1->P2, ..., PK->RX).
                        seg_starts = pts[:-1].copy()
                        seg_ends   = pts[1:].copy()
                        nudges = np.zeros_like(seg_starts)
                        for s in range(1, K + 1):
                            nudges[s] = self.mesh.face_normals[seq[s - 1]]
                        if not np.all(_segments_clear(self.mesh, seg_starts, seg_ends, nudge_normals=nudges)):
                            continue

                        # Exact complex field (identical EM physics to the other engines).
                        d = _assemble_reflection_path(pts, list(seq), self.ray_id, **self.em)
                        if d is None or d["distanza_totale"] > self.max_path_distance:
                            continue

                        d.setdefault("membro_raggruppamento", [self.ray_id])
                        d.setdefault("potenza_incoerente", abs(d["campo_complesso"]) ** 2)

                        self.found_per_order[K] = self.found_per_order.get(K, 0) + 1
                        if self.verbose:
                            ray_label = "[K]" if d["has_diffracted"] else "[P]"
                            print(
                                f"Ray ID #{self.ray_id:5d} | Bounces: {d['bounces']} | {ray_label} | "
                                f"Dist: {d['distanza_totale']:.2f} m | FSL: {d['fsl_db']:.2f} dB | "
                                f"Refl: {d['reflection_loss_db']:.2f} dB | Diff: {d['diffraction_loss_db']:.2f} dB | "
                                f"Pol Loss: {d['pol_loss']:.2f} | "
                                f"IL: {d['IL_db']:.2f} dB | "
                                f"Gain TX: {d['gain_tx_db']:.2f} dB | Gain RX: {d['gain_rx_db']:.2f} dB | "
                                f"Total: {d['path_loss_totale_db']:.2f} dB | Fase: {d['sfasamento_totale_deg']:.2f}°"
                            )
                        self.results["nlos_paths"].append(d)
                        self.ray_id += 1

        if self.enable_profiling:
            self.timer.report()
    
    # ── 4. Report ──────────────────────────────────────────────────────────
    def _print_screen(self):
        if self.verbose:
            print("\n" + "=" * 60)
            print(f"   HYBRID SBR→IMAGE FINAL REPORT | f = {self.frequency/1e9:.2f} GHz")
            print("=" * 60)
            print(f"  TX Pol: {self.tx_pol} | RX Pol: {self.rx_pol}")
            print(f"  Direct rays captured (SBR)  : {self.results['primary_rays_captured']}")
            print("-" * 60)

            for b in range(1, self.max_bounces + 1):
                comps_b = [p for p in self.results["nlos_paths"] if p["bounces"] == b]
                r_count = len([p for p in comps_b if not p["has_diffracted"]])
                d_count = len([p for p in comps_b if p["has_diffracted"]])
                total_b = r_count + d_count
                if total_b > 0:
                    micro_b = sum(len(p.get("membro_raggruppamento", [])) for p in comps_b)
                    print(f"  Paths reaching RX with exactly {b} bounce(s): {total_b} "
                        f"coherent component(s) from {micro_b} micro-ray(s)")
                    print(f"     ├─ Pure Reflections: {r_count}")
                    print(f"     └─ Diffracted Paths: {d_count}")
                    for p in comps_b:
                        tag = "K" if p["has_diffracted"] else "P"
                        print(f"        • [{tag}] faces {tuple(p['faces_hit'])}: "
                            f"{len(p.get('membro_raggruppamento', []))} micro-ray(s) "
                            f"→ {p['path_loss_totale_db']:.2f} dB")

            n_components = len([p for p in self.results["nlos_paths"] if p["bounces"] > 0])
            n_raw_multipath = sum(
                len(p.get("membro_raggruppamento", []))
                for p in self.results["nlos_paths"] if p["bounces"] > 0
            )
            print("-" * 60)
            print(f"  Keller Diffraction Events Triggered : 0")
            print(f"  Raw multipath micro-rays captured   : {n_raw_multipath}")
            print(f"  Coherent multipath components at RX : {n_components}")
            print("=" * 60 + "\n")

    def show_paths(self):
        visualize_with_plotly(self.mesh, self.results, self.pos_rx, self.pos_tx, 1, True, False, None, 
                              self.boresight_tx, self.boresight_rx, self.up_tx, self.up_rx)
        
    def run(self, pos_tx, pos_rx, scenario):
        
        self._pre(pos_tx, pos_rx, scenario)
        self._visual()
        self.los_path()
        self._order_1()
        self._order_n()
        self._print_screen()
        self.show_paths()

        return self.results
    
solver = HybridSolver(
    frequency=2.4e9,
    epsilon_luna=2.87 - 0.01j,
    num_rays=1e6,
    max_bounces=3,
    launch_mode="pyramid",
    occlusion="batched",
    isotropic=False,
    verbose=True,
    tx_pol="RHCP",
    rx_pol="RHCP",
    polarization=True,
    enable_profiling=False,
    vis_map = True
)

results = solver.run(np.zeros(3), 50*np.ones(3), "lunar")