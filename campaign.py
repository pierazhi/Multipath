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
import plotly.io as pio
import pickle
pio.renderers.default = 'browser'

"""
campaign_plots.py
-----------------
Reproduce Figs 4, 5, 6 from the paper given campaign_results from run_campaign().
No files are saved — everything is plotted inline.
"""

def _extract_arrays(campaign_results, pdp_relative_delay=False, use_local_aoa=True):
    """
    Extract arrays for paper-style campaign statistics.

    Statistics PDFs:
        - NLoS only
        - delay is excess delay relative to each pair's LoS
        - AoA is NLoS only

    PDP:
        - includes both LoS and NLoS
        - LoS points can be plotted in red
        - NLoS points can be plotted in blue/black
        - delay can be absolute or relative, controlled by pdp_relative_delay
    """
    C_LIGHT = 299_792_458.0

    # NLoS-only statistics
    delay_nlos_rel_ns = []
    power_nlos_lin = []
    aoa_az_nlos = []
    aoa_el_nlos = []
    rms_ds_list = []

    # PDP arrays: LoS + NLoS kept separate
    pdp_los_delay_ns = []
    pdp_los_power_lin = []

    pdp_nlos_delay_ns = []
    pdp_nlos_power_lin = []

    for entry in campaign_results:
        paths = entry["result"].get("nlos_paths", [])
        if len(paths) == 0:
            continue

        pos_tx = np.asarray(entry["pos_tx"], dtype=float)
        pos_rx = np.asarray(entry["pos_rx"], dtype=float)

        tau_los_ref_ns = np.linalg.norm(pos_rx - pos_tx) / C_LIGHT * 1e9

        # Fallback for global-angle subtraction
        ref_az, ref_el = _los_reference_angles(pos_tx, pos_rx)

        pair_delays = []
        pair_powers = []

        for p in paths:
            bounces = int(p.get("bounces", 0))

            tau_abs_ns = float(p["ritardo_assoluto_ns"])
            tau_rel_ns = tau_abs_ns - tau_los_ref_ns

            # Clean tiny numerical roundoff
            if tau_rel_ns < 0.0 and abs(tau_rel_ns) < 1e-6:
                tau_rel_ns = 0.0

            campo = complex(p["campo_complesso"])
            power_lin = abs(campo) ** 2

            # ── PDP arrays: include LoS + NLoS ─────────────────────
            tau_pdp = tau_rel_ns if pdp_relative_delay else tau_abs_ns

            if bounces == 0:
                pdp_los_delay_ns.append(tau_pdp)
                pdp_los_power_lin.append(power_lin)
                continue

            pdp_nlos_delay_ns.append(tau_pdp)
            pdp_nlos_power_lin.append(power_lin)

            # ── Statistical PDFs: NLoS only, relative delay ─────────
            delay_nlos_rel_ns.append(tau_rel_ns)
            power_nlos_lin.append(power_lin)

            if (
                use_local_aoa
                and "aoa_azimuth_local_deg" in p
                and "aoa_elevation_local_deg" in p
                and np.isfinite(float(p["aoa_azimuth_local_deg"]))
                and np.isfinite(float(p["aoa_elevation_local_deg"]))
            ):
                aoa_az_nlos.append(_wrap180(float(p["aoa_azimuth_local_deg"])))
                aoa_el_nlos.append(float(p["aoa_elevation_local_deg"]))
            else:
                aoa_az_nlos.append(_wrap180(float(p["aoa_azimuth_deg"]) - ref_az))
                aoa_el_nlos.append(float(p["aoa_elevation_deg"]) - ref_el)

            pair_delays.append(tau_rel_ns)
            pair_powers.append(power_lin)

        # RMS-DS per TX/RX pair, NLoS only, using relative delays
        if len(pair_delays) > 1:
            d = np.array(pair_delays, dtype=float)
            w = np.array(pair_powers, dtype=float)

            # Power weights, not power**2
            if np.sum(w) > 0.0:
                w /= np.sum(w)
                tau_mean = np.dot(w, d)
                rms_ds = np.sqrt(max(np.dot(w, d**2) - tau_mean**2, 0.0))
                rms_ds_list.append(float(rms_ds))

    return {
        "delay_nlos_rel_ns": np.asarray(delay_nlos_rel_ns),
        "power_nlos_lin": np.asarray(power_nlos_lin),
        "aoa_az_nlos": np.asarray(aoa_az_nlos),
        "aoa_el_nlos": np.asarray(aoa_el_nlos),
        "rms_ds": np.asarray(rms_ds_list),

        "pdp_los_delay_ns": np.asarray(pdp_los_delay_ns),
        "pdp_los_power_lin": np.asarray(pdp_los_power_lin),
        "pdp_nlos_delay_ns": np.asarray(pdp_nlos_delay_ns),
        "pdp_nlos_power_lin": np.asarray(pdp_nlos_power_lin),
    }

def plot_paper_figures(
    campaign_results,
    label="Terrain",
    pdp_relative_delay=False,
    use_local_aoa=True,
):
    arr = _extract_arrays(
        campaign_results,
        pdp_relative_delay=pdp_relative_delay,
        use_local_aoa=use_local_aoa,
    )

    delay_ns = arr["delay_nlos_rel_ns"]      # NLoS only, relative delay
    power_lin = arr["power_nlos_lin"]        # NLoS only
    aoa_az = arr["aoa_az_nlos"]              # NLoS only
    aoa_el = arr["aoa_el_nlos"]              # NLoS only
    rms_ds = arr["rms_ds"]                   # NLoS only, per pair

    pdp_los_delay = arr["pdp_los_delay_ns"]
    pdp_los_power = arr["pdp_los_power_lin"]

    pdp_nlos_delay = arr["pdp_nlos_delay_ns"]
    pdp_nlos_power = arr["pdp_nlos_power_lin"]

    if len(delay_ns) == 0:
        print("[plot_paper_figures] No NLoS paths available for statistics.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Channel Statistics — {label}", fontsize=14)

    # ── Fig 4(a): PDF of NLoS excess delay ─────────────────────────────
    ax = axes[0, 0]

    ax.hist(
        delay_ns,
        bins=80,
        density=True,
        color="steelblue",
        alpha=0.6,
        label=label,
    )

    d_pos = delay_ns[delay_ns > 1e-9]

    if len(d_pos) > 2:
        k, loc, theta = gamma_dist.fit(d_pos, floc=0)
        x = np.linspace(0, np.percentile(delay_ns, 99), 400)

        ax.plot(
            x,
            gamma_dist.pdf(x, k, loc, theta),
            "r-",
            lw=2,
            label=f"Gamma (k={k:.2f}, θ={theta:.0f})",
        )
    else:
        k, theta = np.nan, np.nan

    ax.set_xlabel("NLoS excess delay relative to LoS (ns)")
    ax.set_ylabel("PDF")
    ax.set_title("Fig 4(a) — PDF of NLoS Excess Delay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Fig 4(b): CDF of RMS-DS ────────────────────────────────────────
    ax = axes[0, 1]

    if len(rms_ds) > 0:
        rms_sorted = np.sort(rms_ds)
        cdf = np.arange(1, len(rms_sorted) + 1) / len(rms_sorted)

        ax.plot(
            rms_sorted,
            cdf,
            color="steelblue",
            lw=2,
            label=label,
        )

    ax.set_xlabel("RMS-DS (ns)")
    ax.set_ylabel("CDF")
    ax.set_title("Fig 4(b) — CDF of RMS-DS")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Fig 5: PDP with LoS + NLoS ─────────────────────────────────────
    ax = axes[0, 2]

    pdp_los_db = 10.0 * np.log10(
        np.where(pdp_los_power > 0.0, pdp_los_power, 1e-300)
    )
    pdp_nlos_db = 10.0 * np.log10(
        np.where(pdp_nlos_power > 0.0, pdp_nlos_power, 1e-300)
    )

    # NLoS points
    if len(pdp_nlos_delay) > 0:
        ax.scatter(
            pdp_nlos_delay,
            pdp_nlos_db,
            s=2,
            alpha=0.30,
            color="steelblue",
            label="NLoS",
        )

    # LoS points
    if len(pdp_los_delay) > 0:
        ax.scatter(
            pdp_los_delay,
            pdp_los_db,
            s=18,
            alpha=0.90,
            color="red",
            label="LoS",
            zorder=5,
        )

    # Exponential fit on NLoS only
    if len(pdp_nlos_delay) > 10:
        logP = np.log(np.clip(pdp_nlos_power, 1e-300, None))

        A = np.vstack([
            np.ones_like(pdp_nlos_delay),
            pdp_nlos_delay,
        ]).T

        coeffs, *_ = np.linalg.lstsq(A, logP, rcond=None)

        b = -coeffs[1]

        x_fit = np.linspace(
            np.min(pdp_nlos_delay),
            np.max(pdp_nlos_delay),
            300,
        )

        y_fit = 10.0 * (coeffs[0] - b * x_fit) / np.log(10.0)

        ax.plot(
            x_fit,
            y_fit,
            "r-",
            lw=2,
            label=f"Exp fit, NLoS (b={b*1e3:.4f}×10⁻³/ns)",
        )

    if pdp_relative_delay:
        ax.set_xlabel("Excess delay relative to LoS (ns)")
    else:
        ax.set_xlabel("Absolute delay (ns)")

    ax.set_ylabel("Power (dB)")
    ax.set_title("Fig 5 — Power Delay Profile")
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(True, alpha=0.3)

    # ── Fig 6(a): PDF of NLoS AAoA ─────────────────────────────────────
    ax = axes[1, 0]

    ax.hist(
        aoa_az,
        bins=80,
        density=True,
        color="steelblue",
        alpha=0.6,
        label=label,
    )

    if len(aoa_az) > 2:
        mu_az, b_az = laplace.fit(aoa_az)
        x_az = np.linspace(aoa_az.min(), aoa_az.max(), 400)

        ax.plot(
            x_az,
            laplace.pdf(x_az, mu_az, b_az),
            "r-",
            lw=2,
            label=f"Laplace (μ={mu_az:.2f}, σ={b_az:.2f})",
        )
    else:
        mu_az, b_az = np.nan, np.nan

    ax.set_xlabel("NLoS azimuth AoA offset Δα (°)")
    ax.set_ylabel("PDF")
    ax.set_title("Fig 6(a) — PDF of NLoS AAoA")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Fig 6(b): PDF of NLoS EAoA ─────────────────────────────────────
    ax = axes[1, 1]

    ax.hist(
        aoa_el,
        bins=80,
        density=True,
        color="steelblue",
        alpha=0.6,
        label=label,
    )

    if len(aoa_el) > 2:
        mu_el, b_el = laplace.fit(aoa_el)
        x_el = np.linspace(aoa_el.min(), aoa_el.max(), 400)

        ax.plot(
            x_el,
            laplace.pdf(x_el, mu_el, b_el),
            "r-",
            lw=2,
            label=f"Laplace (μ={mu_el:.2f}, σ={b_el:.2f})",
        )
    else:
        mu_el, b_el = np.nan, np.nan

    ax.set_xlabel("NLoS elevation AoA offset Δβ (°)")
    ax.set_ylabel("PDF")
    ax.set_title("Fig 6(b) — PDF of NLoS EAoA")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Summary table ─────────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")

    table_data = [
        ["Parameter", "Value"],
        ["Pairs simulated", str(len(campaign_results))],
        ["NLoS paths for PDFs", str(len(delay_ns))],
        ["PDP LoS paths", str(len(pdp_los_delay))],
        ["PDP NLoS paths", str(len(pdp_nlos_delay))],
        ["Mean NLoS excess delay (ns)", f"{delay_ns.mean():.1f}" if len(delay_ns) else "—"],
        ["Mean RMS-DS (ns)", f"{rms_ds.mean():.1f}" if len(rms_ds) else "—"],
        ["Gamma k", f"{k:.3f}" if np.isfinite(k) else "—"],
        ["Gamma θ", f"{theta:.1f}" if np.isfinite(theta) else "—"],
        ["AAoA μ / σ", f"{mu_az:.3f} / {b_az:.3f}" if np.isfinite(mu_az) else "—"],
        ["EAoA μ / σ", f"{mu_el:.3f} / {b_el:.3f}" if np.isfinite(mu_el) else "—"],
    ]

    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    ax.set_title("Fitted Parameters", pad=12)

    plt.tight_layout()
    plt.show()

def _los_reference_angles(pos_tx, pos_rx):
    """Came-from AoA (az, el) at the RX for the direct TX->RX path, in degrees."""
    d = np.asarray(pos_tx, float) - np.asarray(pos_rx, float)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return 0.0, 0.0
    return (np.degrees(np.arctan2(d[1], d[0])),
            np.degrees(np.arcsin(np.clip(d[2] / n, -1.0, 1.0))))


def _wrap180(a):
    """Wrap an angle (deg) to (-180, 180]."""
    return (a + 180.0) % 360.0 - 180.0



def slim_result_for_stats(result):
    """Drop everything except the per-path fields used by campaign_plots.py.
    Keeps memory bounded on large grids; output is plot-compatible."""
    slim_paths = [{
        "campo_complesso":   complex(p["campo_complesso"]),
        "ritardo_assoluto_ns": float(p["ritardo_assoluto_ns"]),
        "aoa_azimuth_deg":   float(p["aoa_azimuth_deg"]),
        "aoa_elevation_deg": float(p["aoa_elevation_deg"]),
        "bounces":           int(p["bounces"]),
    } for p in result.get("nlos_paths", [])]
    return {"nlos_paths": slim_paths}


def run_campaign_2(
    mesh,
    tx_positions,            # (N_tx, 3)
    rx_positions,            # (N_rx, 3)
    engine_fn,
    engine_kwargs=None,      # everything else: r_influence, isotropic, frequenza, num_rays, ...
    auto_boresight=True,     # recompute boresight per pair (TX<->RX). False -> use engine_kwargs
    up_tx=(0.0, 0.0, 1.0),
    up_rx=(0.0, 0.0, 1.0),
    extract_fn=None,         # e.g. slim_result_for_stats, to reduce stored data
    progress=True,
):
    """
    Run engine_fn for every (tx, rx) pair.
 
    Returns a list of dicts, each with keys:
        "tx_idx", "rx_idx", "pos_tx", "pos_rx", "result"
    where "result" is the engine output (or extract_fn(result) if given).
    """
    engine_kwargs = dict(engine_kwargs or {})
    engine_kwargs.setdefault("verbose", False)          # quiet by default for grids
    engine_kwargs.setdefault("up_tx", np.asarray(up_tx, dtype=float))
    engine_kwargs.setdefault("up_rx", np.asarray(up_rx, dtype=float))
    engine_kwargs.setdefault("r_influence", None)       # None -> adaptive radius
 
    if not auto_boresight and ("boresight_tx" not in engine_kwargs
                               or "boresight_rx" not in engine_kwargs):
        raise ValueError("auto_boresight=False requires boresight_tx and "
                         "boresight_rx in engine_kwargs.")
 
    tx_positions = np.asarray(tx_positions, dtype=float)
    rx_positions = np.asarray(rx_positions, dtype=float)
 
    pairs = list(itertools.product(range(len(tx_positions)),
                                   range(len(rx_positions))))
    total = len(pairs)
    all_results = []
 
    for i, (tx_idx, rx_idx) in enumerate(pairs):
        pos_tx = tx_positions[tx_idx]
        pos_rx = rx_positions[rx_idx]
 
        kwargs = dict(engine_kwargs)
        if auto_boresight:
            d = pos_rx - pos_tx
            n = np.linalg.norm(d)
            if n < 1e-9:
                continue                                # coincident TX/RX -> skip
            kwargs["boresight_tx"] =  d / n             # TX looks at RX
            kwargs["boresight_rx"] = -d / n             # RX looks at TX
 
        if progress:
            print(f"[CAMPAIGN] {i+1}/{total} — TX {tx_idx} -> RX {rx_idx}")
 
        result = engine_fn(mesh=mesh, pos_tx=pos_tx, pos_rx=pos_rx, **kwargs)
        if extract_fn is not None:
            result = extract_fn(result)
 
        all_results.append({
            "tx_idx": tx_idx,
            "rx_idx": rx_idx,
            "pos_tx": pos_tx,
            "pos_rx": pos_rx,
            "result": result,
        })
 
    return all_results
