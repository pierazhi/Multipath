"""
polyscope_gui.py
================
Standalone Polyscope + Dear ImGui viewer for the lunar propagation ray tracer.

This is the same architecture Sionna RT's GUI uses: Polyscope renders the 3D
scene (mesh, rays, devices) in its own OpenGL window, and its bundled Dear ImGui
draws the control panel (sliders, checkboxes, readouts) on top.

It consumes the output of `run_campaign_2(...)` directly: a list of dicts, each
  {"tx_idx", "rx_idx", "pos_tx", "pos_rx", "result"}
where result carries the full geometry (`nlos_paths[*]["points"]`, `los_segment`,
`los_blocked`). IMPORTANT: results must be FULL, not slimmed — `slim_result_for_stats`
drops "points" and this viewer needs it.

Scrubbing the TX (and RX) sliders swaps the multipath geometry in real time. No
solver runs in the loop, so it stays smooth; precompute the sweep into
`campaign_results` first.

Usage
-----
    from polyscope_gui import launch_gui
    launch_gui(mesh, campaign_results, r_influence=2.0)

Color convention matches viz.py:
  pure reflection -> per-bounce HSV color,  Keller diffraction -> lime,
  reflection+diffraction (mixed) -> deepskyblue,  LoS clear -> cyan / blocked -> yellow.
"""
from __future__ import annotations

import colorsys
import numpy as np

C_LIGHT = 299_792_458.0

# ---- fixed category colors (match viz.py) ------------------------------------
_LIME = (0.20, 1.00, 0.20)          # Keller diffraction
_DEEPSKYBLUE = (0.00, 0.749, 1.00)  # mixed reflection + diffraction
_CYAN = (0.00, 1.00, 1.00)          # LoS clear
_YELLOW = (1.00, 1.00, 0.00)        # LoS blocked


# =============================================================================
#  Pure data adapters  (no polyscope dependency -> unit-testable headless)
# =============================================================================
def _bounce_color_map(bounce_counts):
    """Map each distinct bounce count to a distinct HSV color (rgb floats 0..1).

    Uses the GLOBAL set of bounce counts so colors stay stable while scrubbing.
    Mirrors viz._bounce_color_map but returns float tuples instead of CSS rgb().
    """
    counts = sorted({int(b) for b in bounce_counts})
    n = max(len(counts), 1)
    return {b: colorsys.hsv_to_rgb(i / n, 0.65, 1.0) for i, b in enumerate(counts)}


def _category(ray):
    """Return (kind, bounces) for an NLoS ray, or None for LoS / degenerate.

    kind in {"refl", "keller", "mixed"} — same split as viz.py.
    """
    b = int(ray.get("bounces", 0))
    if b <= 0:
        return None
    diffracted = bool(ray.get("has_diffracted", False))
    has_refl = any(
        bb.get("tipo", "reflection") == "reflection"
        for bb in ray.get("dettaglio_rimbalzi", [])
    )
    if diffracted and has_refl:
        return ("mixed", b)
    if diffracted:
        return ("keller", b)
    return ("refl", b)


def _group_color(kind, b, refl_cmap):
    if kind == "mixed":
        return _DEEPSKYBLUE
    if kind == "keller":
        return _LIME
    return refl_cmap.get(b, (1.0, 1.0, 1.0))


def _rays_to_networks(entry, refl_cmap, bounce_lo=1, bounce_hi=10_000):
    """Build, per visual group, a single curve-network's node/edge arrays.

    Returns {group_key: (nodes (M,3) float, edges (E,2) int, color rgb)} so each
    group is ONE polyscope structure (constant per-frame structure count is not
    required by polyscope, but one-network-per-group keeps the legend clean and
    toggling cheap).
    """
    paths = entry["result"].get("nlos_paths", [])
    buckets = {}  # group_key -> ([pts arrays], [edge arrays], offset, color)

    for ray in paths:
        cat = _category(ray)
        if cat is None:
            continue
        kind, b = cat
        if b < bounce_lo or b > bounce_hi:
            continue
        pts = np.asarray(ray["points"], dtype=float)
        if pts.ndim != 2 or pts.shape[0] < 2:
            continue

        key = f"{kind}_{b}"
        if key not in buckets:
            buckets[key] = ([], [], 0, _group_color(kind, b, refl_cmap))
        node_chunks, edge_chunks, offset, color = buckets[key]

        m = pts.shape[0]
        seg = np.column_stack([np.arange(m - 1), np.arange(1, m)]) + offset
        node_chunks.append(pts)
        edge_chunks.append(seg)
        buckets[key] = (node_chunks, edge_chunks, offset + m, color)

    out = {}
    for key, (node_chunks, edge_chunks, _off, color) in buckets.items():
        nodes = np.vstack(node_chunks)
        edges = np.vstack(edge_chunks).astype(np.int64)
        out[key] = (nodes, edges, color)
    return out


def _los_to_network(entry):
    """Return (nodes (N,3), edges (E,2) int, color, blocked) or None."""
    res = entry["result"]
    seg = res.get("los_segment", None)
    if seg is None:
        return None
    nodes = np.asarray(seg, dtype=float)
    if nodes.ndim != 2 or nodes.shape[0] < 2:
        return None
    m = nodes.shape[0]
    edges = np.column_stack([np.arange(m - 1), np.arange(1, m)]).astype(np.int64)
    blocked = bool(res.get("los_blocked", False))
    return nodes, edges, (_YELLOW if blocked else _CYAN), blocked


def _stats(entry):
    """Per-link summary numbers for the info panel."""
    res = entry["result"]
    pos_tx = np.asarray(entry["pos_tx"], float)
    pos_rx = np.asarray(entry["pos_rx"], float)
    paths = res.get("nlos_paths", [])

    nlos = [p for p in paths if int(p.get("bounces", 0)) > 0]
    delays = np.array([float(p["ritardo_assoluto_ns"]) for p in nlos])
    powers = np.array([abs(complex(p["campo_complesso"])) ** 2 for p in nlos])

    tau_los = float(np.linalg.norm(pos_rx - pos_tx)) / C_LIGHT * 1e9
    excess = delays - tau_los if len(delays) else np.array([])

    rms_ds = float("nan")
    if len(delays) > 1 and powers.sum() > 0:
        w = powers / powers.sum()
        tau_mean = float(np.dot(w, delays))
        rms_ds = float(np.sqrt(max(np.dot(w, delays ** 2) - tau_mean ** 2, 0.0)))

    return {
        "n_paths_total": len(paths),
        "n_nlos": len(nlos),
        "los_blocked": bool(res.get("los_blocked", False)),
        "tau_los_ns": tau_los,
        "excess_mean_ns": float(excess.mean()) if len(excess) else float("nan"),
        "excess_max_ns": float(excess.max()) if len(excess) else float("nan"),
        "rms_ds_ns": rms_ds,
        "link_dist_m": float(np.linalg.norm(pos_rx - pos_tx)),
    }


def _infer_r_influence(campaign_results, fallback=None):
    """Best-effort capture radius from stored per-path 'r_influence_used'."""
    for e in campaign_results:
        for p in e["result"].get("nlos_paths", []):
            r = p.get("r_influence_used", None)
            if r is not None and np.isfinite(r) and r > 0:
                return float(r)
    return fallback


def _organize(campaign_results):
    """Return (tx_ids, rx_ids, lookup) indexing the campaign.

    lookup maps (i_tx, i_rx) -> entry, where i_* are positions into tx_ids/rx_ids.
    Falls back to a flat one-slider layout if tx_idx/rx_idx are absent.
    """
    have_idx = all(("tx_idx" in e and "rx_idx" in e) for e in campaign_results)
    if have_idx:
        tx_ids = sorted({int(e["tx_idx"]) for e in campaign_results})
        rx_ids = sorted({int(e["rx_idx"]) for e in campaign_results})
        tx_pos = {v: i for i, v in enumerate(tx_ids)}
        rx_pos = {v: i for i, v in enumerate(rx_ids)}
        lookup = {}
        for e in campaign_results:
            lookup[(tx_pos[int(e["tx_idx"])], rx_pos[int(e["rx_idx"])])] = e
        return tx_ids, rx_ids, lookup

    # flat fallback: each entry is its own "TX", single RX column
    lookup = {(i, 0): e for i, e in enumerate(campaign_results)}
    return list(range(len(campaign_results))), [0], lookup


# =============================================================================
#  GUI  (polyscope is imported lazily so the adapters above stay importable)
# =============================================================================
class _State:
    def __init__(self, tx_ids, rx_ids, lookup, refl_cmap, r_influence, global_bmax):
        self.tx_ids = tx_ids
        self.rx_ids = rx_ids
        self.lookup = lookup
        self.refl_cmap = refl_cmap
        self.r_influence = r_influence
        self.global_bmax = global_bmax

        self.i_tx = 0
        self.i_rx = 0
        self.show_rays = True
        self.show_los = True
        self.show_sphere = r_influence is not None
        self.show_boresight = False
        self.ray_radius = 0.0025          # relative to scene extent
        self.bounce_lo = 1
        self.bounce_hi = max(global_bmax, 1)

        self.play = False
        self.play_period = 8              # frames between auto-advances
        self._tick = 0

        self.dirty = True
        self.cn_names = set()             # dynamic curve networks
        self.pc_names = set()             # dynamic point clouds

    def entry(self):
        return self.lookup.get((self.i_tx, self.i_rx), None)


def launch_gui(mesh, campaign_results, r_influence=None,
               background=(0.0, 0.0, 0.0), title="Lunar RT — Propagation Viewer"):
    """Open the standalone Polyscope+ImGui viewer.

    Parameters
    ----------
    mesh : trimesh.Trimesh   (uses .vertices (V,3) and .faces (F,3))
    campaign_results : list of full run_campaign_2 entries (with "points")
    r_influence : float | None   RX capture-sphere radius [m]; inferred if None
    """
    import polyscope as ps
    import polyscope.imgui as psim

    if not campaign_results:
        raise ValueError("campaign_results is empty.")

    r_influence = _infer_r_influence(campaign_results, fallback=r_influence)

    # global bounce range -> stable reflection colormap across all frames
    all_bounces = [
        int(p.get("bounces", 0))
        for e in campaign_results
        for p in e["result"].get("nlos_paths", [])
        if int(p.get("bounces", 0)) > 0
    ]
    refl_cmap = _bounce_color_map(all_bounces)
    global_bmax = max(all_bounces) if all_bounces else 1

    tx_ids, rx_ids, lookup = _organize(campaign_results)
    st = _State(tx_ids, rx_ids, lookup, refl_cmap, r_influence, global_bmax)

    # ---- one-time scene setup ------------------------------------------------
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")          # the moon surface *is* the ground
    ps.set_transparency_mode("pretty")        # needed for the capture sphere
    ps.set_background_color(background)

    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces)
    ps.register_surface_mesh(
        "Lunar Mesh", verts, faces,
        color=(0.42, 0.42, 0.42), edge_width=0.0, transparency=0.55,
        back_face_policy="cull",
    )

    def _clear_dynamic():
        for nm in st.cn_names:
            if ps.has_curve_network(nm):
                ps.remove_curve_network(nm)
        for nm in st.pc_names:
            if ps.has_point_cloud(nm):
                ps.remove_point_cloud(nm)
        st.cn_names.clear()
        st.pc_names.clear()

    def _rebuild():
        _clear_dynamic()
        entry = st.entry()
        if entry is None:
            return
        pos_tx = np.asarray(entry["pos_tx"], float)
        pos_rx = np.asarray(entry["pos_rx"], float)

        # --- TX / RX device markers ---
        tx_pc = ps.register_point_cloud("TX", pos_tx[None, :],
                                        color=(1.0, 0.0, 0.0), point_render_mode="sphere")
        rx_pc = ps.register_point_cloud("RX", pos_rx[None, :],
                                        color=(0.0, 0.35, 1.0), point_render_mode="sphere")
        st.pc_names.update({"TX", "RX"})

        if st.show_boresight:
            d = pos_rx - pos_tx
            n = float(np.linalg.norm(d))
            if n > 1e-9:
                tx_pc.add_vector_quantity("boresight", (d / n)[None, :],
                                          enabled=True, color=(0.86, 0.08, 0.24))
                rx_pc.add_vector_quantity("boresight", (-d / n)[None, :],
                                          enabled=True, color=(0.25, 0.41, 0.88))

        # --- RX capture sphere (single sphere-rendered point, absolute radius) ---
        if st.show_sphere and st.r_influence:
            sph = ps.register_point_cloud("RX capture sphere", pos_rx[None, :],
                                          color=(0.0, 1.0, 1.0),
                                          point_render_mode="sphere")
            sph.set_radius(float(st.r_influence), relative=False)
            sph.set_transparency(0.18)
            st.pc_names.add("RX capture sphere")

        # --- Line of sight ---
        if st.show_los:
            los = _los_to_network(entry)
            if los is not None:
                nodes, edges, color, blocked = los
                cn = ps.register_curve_network("Line of Sight", nodes, edges,
                                               color=color)
                cn.set_radius(st.ray_radius * 1.6, relative=True)
                st.cn_names.add("Line of Sight")

        # --- Multipath rays, grouped by category+bounce ---
        if st.show_rays:
            groups = _rays_to_networks(entry, st.refl_cmap,
                                       bounce_lo=st.bounce_lo, bounce_hi=st.bounce_hi)
            for key, (nodes, edges, color) in groups.items():
                nm = f"rays [{key}]"
                cn = ps.register_curve_network(nm, nodes, edges, color=color)
                cn.set_radius(st.ray_radius, relative=True)
                st.cn_names.add(nm)

        if hasattr(ps, "reset_camera_to_home_view") and st._tick == 0:
            ps.reset_camera_to_home_view()

    # ---- per-frame ImGui control panel --------------------------------------
    def callback():
        psim.TextColored((0.4627, 0.7255, 0.0, 1.0), title)
        psim.Separator()

        # TX / RX navigation
        if len(st.tx_ids) > 1:
            changed, st.i_tx = psim.SliderInt(
                f"TX  (idx {st.tx_ids[st.i_tx]})", st.i_tx, 0, len(st.tx_ids) - 1)
            st.dirty |= changed
        if len(st.rx_ids) > 1:
            changed, st.i_rx = psim.SliderInt(
                f"RX  (idx {st.rx_ids[st.i_rx]})", st.i_rx, 0, len(st.rx_ids) - 1)
            st.dirty |= changed

        # transport controls
        changed, st.play = psim.Checkbox("Play (auto-sweep TX)", st.play)
        psim.SameLine()
        if psim.Button("Prev"):
            st.i_tx = (st.i_tx - 1) % len(st.tx_ids); st.dirty = True
        psim.SameLine()
        if psim.Button("Next"):
            st.i_tx = (st.i_tx + 1) % len(st.tx_ids); st.dirty = True
        _, st.play_period = psim.SliderInt("Sweep period (frames)", st.play_period, 1, 60)

        psim.Separator()

        # display toggles
        c1, st.show_rays = psim.Checkbox("Rays", st.show_rays); st.dirty |= c1
        psim.SameLine()
        c2, st.show_los = psim.Checkbox("LoS", st.show_los); st.dirty |= c2
        c3, st.show_sphere = psim.Checkbox("Capture sphere", st.show_sphere); st.dirty |= c3
        psim.SameLine()
        c4, st.show_boresight = psim.Checkbox("Boresight", st.show_boresight); st.dirty |= c4

        cr, st.ray_radius = psim.SliderFloat("Ray thickness", st.ray_radius,
                                             0.0005, 0.02)
        st.dirty |= cr

        if st.global_bmax > 1:
            cl, st.bounce_lo = psim.SliderInt("Min bounces", st.bounce_lo,
                                              1, st.global_bmax)
            ch, st.bounce_hi = psim.SliderInt("Max bounces", st.bounce_hi,
                                              1, st.global_bmax)
            if st.bounce_hi < st.bounce_lo:
                st.bounce_hi = st.bounce_lo
            st.dirty |= (cl or ch)

        # info readout
        psim.Separator()
        entry = st.entry()
        if entry is not None:
            s = _stats(entry)
            blk = "BLOCKED" if s["los_blocked"] else "clear"
            psim.Text(f"Link distance : {s['link_dist_m']:.1f} m   (LoS {blk})")
            psim.Text(f"Paths total   : {s['n_paths_total']}   |   NLoS: {s['n_nlos']}")
            psim.Text(f"LoS delay     : {s['tau_los_ns']:.1f} ns")
            psim.Text(f"Excess delay  : mean {s['excess_mean_ns']:.1f}  "
                      f"max {s['excess_max_ns']:.1f} ns")
            psim.Text(f"RMS delay spr : {s['rms_ds_ns']:.2f} ns")

        # auto-sweep
        if st.play and len(st.tx_ids) > 1:
            st._tick += 1
            if st._tick >= st.play_period:
                st._tick = 0
                st.i_tx = (st.i_tx + 1) % len(st.tx_ids)
                st.dirty = True

        if st.dirty:
            _rebuild()
            st.dirty = False

    _rebuild()
    st._tick = 1  # don't re-home the camera on subsequent rebuilds
    ps.set_user_callback(callback)
    ps.show()
    ps.clear_user_callback()