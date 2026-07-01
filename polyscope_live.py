"""
polyscope_live.py
=================
Single-link (one TX, one RX) Polyscope + Dear ImGui viewer where the TX can be
**moved** — by dragging a 3D gizmo, typing coordinates, or playing through a list
of waypoints — and the multipath is **recomputed live** by your engine at each
new position.

Difference from polyscope_gui.py: that one replays a precomputed campaign grid.
This one calls your `engine_fn(mesh, pos_tx, pos_rx, **kwargs)` on demand. Because
the SBR solve can take a while, it runs on a background worker thread (latest
request wins, results cached by position), so the window stays responsive and
animation paces itself to however fast solves complete. Polyscope is only ever
touched from the main thread.

Reuses the data adapters from polyscope_gui.py (keep both files together).

Usage
-----
    from engine_sbr import engine_sbr          # or engine_hybrid, ...
    from polyscope_live import launch_live

    launch_live(
        mesh,
        pos_tx=[0, 0, 50.0],
        pos_rx=[100, 0, 2.0],
        engine_fn=engine_sbr,
        engine_kwargs=dict(r_influence=2.0, frequenza=2.4e9, num_rays=20000),
        tx_path=[[0,0,50],[40,0,45],[80,0,40]],   # optional animation waypoints
    )

Move the TX: enable the gizmo and drag it (it solves when you pause), or type
coordinates, or hit Play to sweep the waypoint path.
"""
from __future__ import annotations

import threading
import time
import numpy as np

# shared, display-free adapters
from polyscope_gui import (
    _rays_to_networks, _los_to_network, _stats, _bounce_color_map,
)

C_LIGHT = 299_792_458.0
_CYAN = (0.0, 1.0, 1.0)

# keeps state/solver alive for non-blocking (Jupyter) viewers so they aren't GC'd
_LIVE_HANDLES = []


# =============================================================================
#  Pure helpers (headless-testable)
# =============================================================================
def _solve(engine_fn, mesh, pos_tx, pos_rx, engine_kwargs=None, auto_boresight=True):
    """Run the engine once and wrap output as an 'entry' dict the adapters accept."""
    pos_tx = np.asarray(pos_tx, float)
    pos_rx = np.asarray(pos_rx, float)
    kw = dict(engine_kwargs or {})
    kw.setdefault("verbose", False)
    # The hybrid engine normalizes up_tx/up_rx unconditionally; None -> NaN crash.
    # run_campaign_2 always supplied these, so mirror that here.
    kw.setdefault("up_tx", (0.0, 0.0, 1.0))
    kw.setdefault("up_rx", (0.0, 0.0, 1.0))
    if auto_boresight:
        d = pos_rx - pos_tx
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            kw["boresight_tx"] = d / n     # TX looks at RX
            kw["boresight_rx"] = -d / n     # RX looks at TX
    res = engine_fn(mesh=mesh, pos_tx=pos_tx, pos_rx=pos_rx, **kw)
    return {"pos_tx": pos_tx, "pos_rx": pos_rx, "result": res}


def _densify(waypoints, substeps):
    """Linearly interpolate a waypoint list into a dense trajectory (M,3).

    substeps = number of samples per segment (>=1). 1 => waypoints only.
    """
    wp = np.asarray(waypoints, float)
    if wp.ndim != 2 or wp.shape[0] < 2 or substeps <= 1:
        return wp.copy() if wp.size else wp.reshape(0, 3)
    out = []
    for a, b in zip(wp[:-1], wp[1:]):
        ts = np.linspace(0.0, 1.0, int(substeps), endpoint=False)[:, None]
        out.append(a[None, :] * (1 - ts) + b[None, :] * ts)
    out.append(wp[-1][None, :])
    return np.vstack(out)


def _pos_key(pos, ndigits=3):
    return tuple(np.round(np.asarray(pos, float), ndigits))


def _translate(p):
    """4x4 pure-translation matrix (devices are a point at the local origin
    positioned via this transform, so the gizmo handle and the sphere coincide)."""
    M = np.eye(4)
    M[:3, 3] = np.asarray(p, float)
    return M


# =============================================================================
#  Background solver
# =============================================================================
class _Solver:
    """Position-keyed, cached solver. Two modes:

    mode="sync"   : request() runs the engine inline (blocks the caller). Best
                    for fast engines — the render loop pauses during the solve,
                    so it runs uncontended at full speed (no GIL fight).
    mode="thread" : request() hands off to a worker thread (latest-wins) so the
                    UI stays live. Good for slow solves, but a Python-bound
                    engine will run slower here because the main-thread render
                    loop contends for the GIL the whole time it computes.
    """

    def __init__(self, engine_fn, mesh, engine_kwargs, auto_boresight,
                 mode="sync"):
        self._engine_fn = engine_fn
        self._mesh = mesh
        self._kwargs = engine_kwargs
        self._auto = auto_boresight
        self._mode = mode

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._pending = None           # (pos_tx, pos_rx) latest wins
        self._out = None               # dict: pos_tx,pos_rx,entry,err
        self._busy = False
        self._cache = {}
        self._thread = None

        if mode == "thread":
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def request(self, pos_tx, pos_rx):
        pos_tx = np.asarray(pos_tx, float)
        pos_rx = np.asarray(pos_rx, float)
        if self._mode == "sync":
            out = self._solve_cached(pos_tx, pos_rx)   # blocks here (uncontended)
            with self._lock:
                self._out = out
            return
        with self._lock:
            self._pending = (pos_tx, pos_rx)
            self._busy = True
        self._wake.set()

    def poll(self):
        """Return a finished result dict once, or None. Non-blocking."""
        with self._lock:
            out, self._out = self._out, None
            return out

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def shutdown(self):
        self._stop.set()
        self._wake.set()

    def _solve_cached(self, pos_tx, pos_rx):
        """Run (or cache-hit) one solve and return the result dict. Shared by
        both modes; the engine call here releases the GIL only for its NumPy
        parts, so this is fastest when nothing else is running concurrently."""
        key = (_pos_key(pos_tx), _pos_key(pos_rx))
        entry = self._cache.get(key)
        if entry is not None:
            print(f"[live] cache hit TX={np.round(pos_tx, 1).tolist()}", flush=True)
            return {"pos_tx": pos_tx, "pos_rx": pos_rx, "entry": entry, "err": None}
        t0 = time.time()
        print(f"[live] solving  TX={np.round(pos_tx, 1).tolist()} "
              f"RX={np.round(pos_rx, 1).tolist()} ...", flush=True)
        try:
            entry = _solve(self._engine_fn, self._mesh, pos_tx, pos_rx,
                           self._kwargs, self._auto)
            self._cache[key] = entry
            n = len([p for p in entry["result"].get("nlos_paths", [])
                     if int(p.get("bounces", 0)) > 0])
            print(f"[live] done in {time.time() - t0:.1f}s — {n} NLoS paths",
                  flush=True)
            return {"pos_tx": pos_tx, "pos_rx": pos_rx, "entry": entry, "err": None}
        except Exception as e:                      # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
            print(f"[live] solve FAILED in {time.time() - t0:.1f}s: {err}",
                  flush=True)
            return {"pos_tx": pos_tx, "pos_rx": pos_rx, "entry": None, "err": err}

    def _run(self):
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.1):
                continue
            self._wake.clear()
            with self._lock:
                req, self._pending = self._pending, None
            if req is None:
                continue
            out = self._solve_cached(*req)
            with self._lock:
                self._out = out
                self._busy = self._pending is not None   # newer request queued?
                if self._pending is not None:
                    self._wake.set()


# =============================================================================
#  GUI
# =============================================================================
class _LiveState:
    def __init__(self, pos_tx, pos_rx, r_influence, tx_path):
        self.tx_pos = np.asarray(pos_tx, float)
        self.rx_pos = np.asarray(pos_rx, float)
        self.r_influence = r_influence

        self.cur_entry = None
        self.last_err = None
        self.global_bounces = set()
        self.refl_cmap = _bounce_color_map([])

        # display
        self.show_rays = True
        self.show_los = True
        self.show_sphere = r_influence is not None
        self.show_boresight = False
        self.show_traj = True
        self.ray_radius = 0.0025
        self.bounce_lo = 1
        self.bounce_hi = 8

        # gizmo / move
        self.gizmo = True
        self.active = "TX"                  # which device the gizmo controls
        self.tx_field = list(map(float, self.tx_pos))
        self.rx_field = list(map(float, self.rx_pos))
        self._cand = self.tx_pos.copy()
        self._cand_prev = self.tx_pos.copy()
        self._stable = 0
        self._giz_prev = self.tx_pos.copy()  # last active-gizmo world position
        self._giz_cfg = (False, False)      # (tx_gizmo_on, rx_gizmo_on) applied

        # animation
        self.tx_path = (np.asarray(tx_path, float)
                        if tx_path is not None and len(tx_path) else
                        np.zeros((0, 3)))
        self.substeps = 1
        self.play = False
        self.play_idx = 0
        self.play_period = 4
        self._tick = 0
        self._traj = self.tx_path.copy()

        self.cn_names = set()          # dynamic curve networks
        self.pc_names = set()          # dynamic point clouds


def launch_live(mesh, pos_tx, pos_rx, engine_fn, engine_kwargs=None,
                auto_boresight=True, r_influence=None, tx_path=None,
                blocking=True,
                background=(0.0, 0.0, 0.0), title="Lunar RT — Live Single Link"):
    """Open the interactive single-link viewer.

    mesh         : trimesh.Trimesh (uses .vertices/.faces)
    pos_tx/pos_rx: initial positions [m]
    engine_fn    : your solver, called engine_fn(mesh=, pos_tx=, pos_rx=, **kwargs)
    engine_kwargs: dict passed to the engine (r_influence, frequenza, num_rays, ...)
    auto_boresight: re-aim TX->RX and RX->TX each solve (like run_campaign_2)
    r_influence  : capture-sphere radius for display; falls back to engine_kwargs
    tx_path      : optional (N,3) waypoints for the Play animation
    """
    import polyscope as ps
    import polyscope.imgui as psim

    if r_influence is None and engine_kwargs:
        r_influence = engine_kwargs.get("r_influence", None)

    st = _LiveState(pos_tx, pos_rx, r_influence, tx_path)
    solver = _Solver(engine_fn, mesh, dict(engine_kwargs or {}), auto_boresight)

    # ---- static scene --------------------------------------------------------
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")
    ps.set_transparency_mode("pretty")
    if hasattr(ps, "set_background_color"):
        ps.set_background_color(background)

    ps.register_surface_mesh(
        "Lunar Mesh", np.asarray(mesh.vertices, float), np.asarray(mesh.faces),
        color=(0.42, 0.42, 0.42), edge_width=0.0, transparency=0.55,
        back_face_policy="identical",   # terrain is one-sided; render both faces
    )

    # Device markers: a single point at the LOCAL ORIGIN, positioned purely via
    # the structure transform's translation. This guarantees the gizmo handle
    # and the rendered sphere are co-located, even far from the world origin.
    _ORIGIN = np.zeros((1, 3))
    tx_pc = ps.register_point_cloud("TX", _ORIGIN,
                                    color=(1.0, 0.0, 0.0), point_render_mode="sphere")
    rx_pc = ps.register_point_cloud("RX", _ORIGIN,
                                    color=(0.0, 0.35, 1.0), point_render_mode="sphere")
    tx_pc.set_transform(_translate(st.tx_pos))
    rx_pc.set_transform(_translate(st.rx_pos))
    tx_pc.set_transform_gizmo_enabled(st.gizmo and st.active == "TX")
    rx_pc.set_transform_gizmo_enabled(st.gizmo and st.active == "RX")
    st._giz_cfg = (st.gizmo and st.active == "TX", st.gizmo and st.active == "RX")

    sph_pc = None
    if st.r_influence:
        # sphere has no gizmo, so we position it directly by its point location
        sph_pc = ps.register_point_cloud("RX capture sphere", st.rx_pos[None, :],
                                         color=_CYAN, point_render_mode="sphere")
        sph_pc.set_radius(float(st.r_influence), relative=False)
        sph_pc.set_transparency(0.18)

    # ---- dynamic geometry helpers -------------------------------------------
    def _clear_dynamic():
        for nm in st.cn_names:
            if ps.has_curve_network(nm):
                ps.remove_curve_network(nm)
        for nm in st.pc_names:
            if ps.has_point_cloud(nm):
                ps.remove_point_cloud(nm)
        st.cn_names.clear()
        st.pc_names.clear()

    def _place_markers():
        """Position TX/RX (via transform) and the sphere to committed positions.

        Called only when a position is COMMITTED (or boresight/sphere toggled) —
        never on every solve result, so an incoming solve can't disturb a drag.
        Setting a pure-translation transform also discards any gizmo rotation.
        """
        tx_pc.set_transform(_translate(st.tx_pos))
        rx_pc.set_transform(_translate(st.rx_pos))
        if sph_pc is not None:
            sph_pc.update_point_positions(st.rx_pos[None, :])
            sph_pc.set_enabled(st.show_sphere)

        tx_pc.remove_all_quantities()
        rx_pc.remove_all_quantities()
        if st.show_boresight:
            d = st.rx_pos - st.tx_pos
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                tx_pc.add_vector_quantity("boresight", (d / n)[None, :],
                                          enabled=True, color=(0.86, 0.08, 0.24))
                rx_pc.add_vector_quantity("boresight", (-d / n)[None, :],
                                          enabled=True, color=(0.25, 0.41, 0.88))

    def _rebuild_dynamic():
        """Rebuild rays / LoS / trajectory. Does NOT touch device markers."""
        _clear_dynamic()

        # trajectory
        if st.show_traj and len(st._traj) >= 1:
            wp = ps.register_point_cloud("TX waypoints", st._traj,
                                         color=(1.0, 0.6, 0.0))
            wp.set_radius(0.004, relative=True)
            st.pc_names.add("TX waypoints")
            if len(st._traj) >= 2:
                e = np.column_stack([np.arange(len(st._traj) - 1),
                                     np.arange(1, len(st._traj))]).astype(np.int64)
                tj = ps.register_curve_network("TX trajectory", st._traj, e,
                                               color=(1.0, 0.6, 0.0))
                tj.set_radius(0.0015, relative=True)
                st.cn_names.add("TX trajectory")

        entry = st.cur_entry
        if entry is None:
            return

        if st.show_los:
            los = _los_to_network(entry)
            if los is not None:
                nodes, edges, color, _blk = los
                cn = ps.register_curve_network("Line of Sight", nodes, edges,
                                               color=color)
                cn.set_radius(st.ray_radius * 1.6, relative=True)
                st.cn_names.add("Line of Sight")

        if st.show_rays:
            groups = _rays_to_networks(entry, st.refl_cmap,
                                       bounce_lo=st.bounce_lo, bounce_hi=st.bounce_hi)
            for key, (nodes, edges, color) in groups.items():
                nm = f"rays [{key}]"
                cn = ps.register_curve_network(nm, nodes, edges, color=color)
                cn.set_radius(st.ray_radius, relative=True)
                st.cn_names.add(nm)

    def _ingest_result(out):
        """Take a finished solve from the worker into state."""
        st.last_err = out["err"]
        if out["entry"] is not None:
            st.cur_entry = out["entry"]
            # grow stable colormap if new bounce orders appeared
            bs = {int(p.get("bounces", 0))
                  for p in out["entry"]["result"].get("nlos_paths", [])
                  if int(p.get("bounces", 0)) > 0}
            if not bs.issubset(st.global_bounces):
                st.global_bounces |= bs
                st.refl_cmap = _bounce_color_map(st.global_bounces)

    def _commit(tx_pos, rx_pos):
        """Adopt new positions: place markers now (instant visual), then solve."""
        st.tx_pos = np.asarray(tx_pos, float)
        st.rx_pos = np.asarray(rx_pos, float)
        st.tx_field = list(map(float, st.tx_pos))
        st.rx_field = list(map(float, st.rx_pos))
        st._stable = 0
        # gizmo translation is now the ABSOLUTE position; seed tracker with it
        st._giz_prev = (st.tx_pos if st.active == "TX" else st.rx_pos).copy()
        _place_markers()                 # repositions transforms immediately
        solver.request(st.tx_pos, st.rx_pos)

    # ---- per-frame callback --------------------------------------------------
    DEBOUNCE = 6      # frames of a still gizmo before auto-solving
    EPS = 1e-6

    def _apply_gizmo():
        """Show the gizmo only on the active device (idempotent)."""
        want = (st.gizmo and st.active == "TX", st.gizmo and st.active == "RX")
        if want != st._giz_cfg:
            tx_pc.set_transform_gizmo_enabled(want[0])
            rx_pc.set_transform_gizmo_enabled(want[1])
            st._giz_cfg = want

    def _set_active(name, live_tx, live_rx, drifted):
        """Switch which device the gizmo controls, baking any pending drag."""
        if name == st.active:
            return
        if st.gizmo and drifted:
            _commit(live_tx, live_rx)
        st.active = name
        st._stable = 0
        st._giz_prev = (st.tx_pos if name == "TX" else st.rx_pos).copy()
        _apply_gizmo()

    def callback():
        # 1) drain finished solves -> rebuild rays only (markers already placed)
        out = solver.poll()
        if out is not None:
            _ingest_result(out)
            _rebuild_dynamic()

        # 2) read the ACTIVE gizmo only. The marker is a point at the origin
        #    positioned by the transform, so the transform translation IS the
        #    device's world position and the gizmo handle sits exactly on it.
        live_tx, live_rx = st.tx_pos.copy(), st.rx_pos.copy()
        drifted = False
        if st.gizmo:
            pc = tx_pc if st.active == "TX" else rx_pc
            t = np.asarray(pc.get_transform(), float)[:3, 3]   # absolute position
            committed = st.tx_pos if st.active == "TX" else st.rx_pos
            if st.active == "TX":
                live_tx = t
            else:
                live_rx = t
            moved = float(np.linalg.norm(t - st._giz_prev))
            st._giz_prev = t
            st._stable = 0 if moved > EPS else st._stable + 1
            drifted = float(np.linalg.norm(t - committed)) > 1e-4

            # clicking a TX/RX sphere in the 3D view selects it -> gizmo follows
            if ps.have_selection():
                sel = ps.get_selection()
                nm = getattr(sel, "structure_name", None)
                if nm in ("TX", "RX"):
                    _set_active(nm, live_tx, live_rx, drifted)

            if (not st.play and not solver.busy
                    and st._stable == DEBOUNCE and drifted):
                _commit(live_tx, live_rx)

        # 3) panel
        psim.TextColored((0.4627, 0.7255, 0.0, 1.0), title)
        psim.Separator()

        cg, st.gizmo = psim.Checkbox("Enable move gizmo", st.gizmo)
        if cg:
            if not st.gizmo and drifted:          # bake pending drag on disable
                _commit(live_tx, live_rx)
            _apply_gizmo()
        psim.Text("  select a device (or click its sphere in the view):")
        if psim.Button("Control TX"):
            _set_active("TX", live_tx, live_rx, drifted)
        psim.SameLine()
        if psim.Button("Control RX"):
            _set_active("RX", live_tx, live_rx, drifted)
        psim.SameLine()
        psim.Text(f"-> {st.active}")
        if st.gizmo:
            psim.Text(f"  TX [{live_tx[0]:.1f}, {live_tx[1]:.1f}, {live_tx[2]:.1f}]"
                      f"   RX [{live_rx[0]:.1f}, {live_rx[1]:.1f}, {live_rx[2]:.1f}]")
            if psim.Button("Solve now"):
                _commit(live_tx, live_rx)

        # numeric TX
        ctx = False
        for i, ax in enumerate("XYZ"):
            c, st.tx_field[i] = psim.DragFloat(f"TX {ax}", st.tx_field[i], 0.25)
            ctx |= c
        # numeric RX
        crx = False
        for i, ax in enumerate("XYZ"):
            c, st.rx_field[i] = psim.DragFloat(f"RX {ax}", st.rx_field[i], 0.25)
            crx |= c
        if (ctx or crx) and not solver.busy:
            _commit(np.asarray(st.tx_field, float), np.asarray(st.rx_field, float))


        if solver.busy:
            psim.TextColored((1.0, 0.55, 0.0, 1.0), "  solving...")
        elif st.last_err:
            psim.TextColored((1.0, 0.3, 0.3, 1.0), f"  solve error: {st.last_err}")

        # 4) animation
        psim.Separator()
        psim.Text(f"Trajectory: {len(st.tx_path)} waypoints "
                  f"({len(st._traj)} steps @ x{st.substeps})")
        if psim.Button("Add current TX"):
            st.tx_path = np.vstack([st.tx_path, st.tx_pos[None, :]])
            st._traj = _densify(st.tx_path, st.substeps)
            _rebuild_dynamic()
        psim.SameLine()
        if psim.Button("Clear path"):
            st.tx_path = np.zeros((0, 3))
            st._traj = st.tx_path.copy()
            st.play = False
            _rebuild_dynamic()
        cs, st.substeps = psim.SliderInt("Interp substeps", st.substeps, 1, 40)
        if cs:
            st._traj = _densify(st.tx_path, st.substeps)
            _rebuild_dynamic()
        cp, st.play = psim.Checkbox("Play (sweep TX along path)", st.play)
        if cp and st.play:
            st._traj = _densify(st.tx_path, st.substeps)
            st.play_idx = 0
        _, st.play_period = psim.SliderInt("Min frames / step", st.play_period, 1, 30)

        # advance animation only when the previous solve is done
        if st.play and len(st._traj) >= 1:
            st._tick += 1
            if not solver.busy and st._tick >= st.play_period:
                st._tick = 0
                if st.play_idx >= len(st._traj):
                    st.play_idx = 0                  # loop
                _commit(st._traj[st.play_idx], st.rx_pos)
                st.play_idx += 1

        # 5) display options + readout
        psim.Separator()
        c1, st.show_rays = psim.Checkbox("Rays", st.show_rays)
        psim.SameLine(); c2, st.show_los = psim.Checkbox("LoS", st.show_los)
        c3, st.show_sphere = psim.Checkbox("Capture sphere", st.show_sphere)
        psim.SameLine(); c4, st.show_boresight = psim.Checkbox("Boresight", st.show_boresight)
        c5, st.show_traj = psim.Checkbox("Show trajectory", st.show_traj)
        cr, st.ray_radius = psim.SliderFloat("Ray thickness", st.ray_radius, 0.0005, 0.02)
        cl, st.bounce_lo = psim.SliderInt("Min bounces", st.bounce_lo, 1, 12)
        ch, st.bounce_hi = psim.SliderInt("Max bounces", st.bounce_hi, 1, 12)
        if st.bounce_hi < st.bounce_lo:
            st.bounce_hi = st.bounce_lo
        if c3 or c4:                                   # sphere / boresight live on markers
            _place_markers()
        if any([c1, c2, c5, cr, cl, ch]):              # rays / los / traj / filters
            _rebuild_dynamic()

        if st.cur_entry is not None:
            s = _stats(st.cur_entry)
            blk = "BLOCKED" if s["los_blocked"] else "clear"
            psim.Separator()
            psim.Text(f"TX [{st.tx_pos[0]:.1f}, {st.tx_pos[1]:.1f}, {st.tx_pos[2]:.1f}]"
                      f"   link {s['link_dist_m']:.1f} m  (LoS {blk})")
            psim.Text(f"NLoS paths: {s['n_nlos']}   "
                      f"excess delay mean {s['excess_mean_ns']:.1f} / "
                      f"max {s['excess_max_ns']:.1f} ns")
            psim.Text(f"RMS delay spread: {s['rms_ds_ns']:.2f} ns")

    # ---- initial solve + run -------------------------------------------------
    solver.request(st.tx_pos, st.rx_pos)
    ps.set_user_callback(callback)

    if blocking:
        # Standalone / script use: show() runs the native loop and blocks until
        # the window is closed. DO NOT use this path in a Jupyter cell — it will
        # hang the kernel. Use blocking=False with `%gui polyscope` instead.
        try:
            ps.show()
        finally:
            ps.clear_user_callback()
            solver.shutdown()
        return None

    # Non-blocking (Jupyter + `interactive_polyscope` + `%gui polyscope`):
    # show() returns immediately and the window updates alongside the notebook.
    # We must NOT tear down here, or the callback/worker would die instantly.
    ps.show()

    def stop():
        ps.clear_user_callback()
        solver.shutdown()

    handles = {"state": st, "solver": solver, "stop": stop}
    _LIVE_HANDLES.append(handles)        # keep references alive
    return handles