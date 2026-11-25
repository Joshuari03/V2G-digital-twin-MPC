"""Entry point for the V2G frequency and voltage control workflow."""

"""
main.py
-------
Central simulation loop for the V2G smart grid study.

Scenarios
---------
Each scenario is a combination of:
  - EV penetration  : 20 / 50 / 80 % of fleet connected
  - Control strategy: G2V (uncontrolled)
                      RB-V2G  (rule-based / heuristic V2G, ev_model.simulate_v2g)
                      MPC-V2G (QP-MPC, cvxpy + OSQP, mpc_control.py)
  - PV              : enabled / disabled  (toggle PV_ENABLED in network_setup.py)

Results are saved as CSV in results/ and passed to plots.py.
"""

import argparse
import os

import numpy as np
import pandas as pd

from ev_model      import generate_fleet, simulate_g2v, simulate_v2g, N_EV, SIM_START_H, E_BAT
from network_setup import make_pf_runner, PV_ENABLED
from pv_model      import get_pv_profile, get_pv_surplus, align_to_sim_start
from mpc_control   import run_mpc, MPCConfig

import plots   # will call plots.py once results are ready

# ── Scenario definitions ──────────────────────────────────────────────────────
EV_PENETRATIONS = [0.2, 0.5, 0.8]    # fraction of full fleet active
STRATEGIES      = ['G2V', 'RB-V2G', 'MPC-V2G']
MPC_CFG         = MPCConfig()        # weights / target SoC: see mpc_control.MPCConfig
SIM_DATE        = '2024-06-21'

# Base load profile: realistic 24h residential curve [MW total network]
# Values represent the sum across all 32 load buses.
# Shape loosely follows a typical Chinese urban feeder:
#   - morning ramp 06-09, midday plateau, evening peak 18-21, night trough
_BASE_LOAD_TOTAL_MW = np.array([
    0.8, 0.7, 0.65, 0.6, 0.6, 0.65,    # 00-05
    0.8, 1.0, 1.3,  1.5, 1.6, 1.65,    # 06-11
    1.7, 1.65,1.6,  1.5, 1.6, 1.8,     # 12-17
    2.0, 2.1, 2.0,  1.8, 1.5, 1.1,     # 18-23
])   # shape (24,)  — index 0 = 00:00

def run_scenario(penetration: float,
                 strategy: str,
                 fleet_full: pd.DataFrame,
                 pv_sim: np.ndarray) -> dict:
    """
    Run one 24-hour scenario and return a result dict.

    penetration : 0.2 / 0.5 / 0.8
    strategy    : 'G2V' or 'V2G'
    fleet_full  : full 500-EV DataFrame from generate_fleet()
    pv_sim      : shape (24,) PV output aligned to SIM_START_H
    """
    # Scale fleet to penetration level
    n_active = int(N_EV * penetration)
    fleet    = fleet_full.iloc[:n_active].copy().reset_index(drop=True)

    # ── Forecast inputs (aligned to the simulation axis, index 0 = SIM_START_H) ──
    base_load_sim = np.array([
        _BASE_LOAD_TOTAL_MW[(SIM_START_H + s) % 24] * 1000   # → kW
        for s in range(24)
    ])
    pv_surplus = get_pv_surplus(pv_sim, base_load_sim)

    # One AC power-flow "runner": run(slot, ev_total_kw) -> (V[33], loading[37])
    pf = make_pf_runner(_BASE_LOAD_TOTAL_MW, pv_sim, SIM_START_H, PV_ENABLED)

    def _run_power_flow(ev_profile_kw: np.ndarray) -> tuple:
        """Open-loop PF across 24 slots for a given EV profile."""
        voltages_24h = np.zeros((24, 33))
        loading_24h  = np.zeros((24, 37))
        for slot in range(24):
            voltages_24h[slot], loading_24h[slot] = pf(slot, ev_profile_kw[slot])
            if np.isnan(voltages_24h[slot]).any():
                print(f"  [WARN] Power flow did not converge at slot {slot} "
                      f"({(SIM_START_H + slot) % 24:02d}:00) — filled with NaN")
        return voltages_24h, loading_24h, np.asarray(ev_profile_kw, float).copy()

    extra, info = {}, {}
    if strategy == 'G2V':
        ev_total_kw, soc_profile, info = simulate_g2v(fleet)
        voltages_24h, loading_24h, ev_kw_24h = _run_power_flow(ev_total_kw)

    elif strategy == 'RB-V2G':
        # Pass 1: run G2V to get a realistic voltage profile
        ev_g2v_kw, _, _ = simulate_g2v(fleet.copy())
        voltages_ref, _, _ = _run_power_flow(ev_g2v_kw)
        v_profile = np.nanmean(voltages_ref, axis=1)
        if np.isnan(v_profile).all():
            v_profile = np.ones(24)

        # Pass 2: run rule-based V2G with the measured voltage profile
        ev_total_kw, soc_profile, info = simulate_v2g(fleet.copy(), v_profile, pv_surplus)
        voltages_24h, loading_24h, ev_kw_24h = _run_power_flow(ev_total_kw)

    elif strategy == 'MPC-V2G':
        # Closed loop: the AC power flow is called inside the control loop
        # (re-centring of the LinDistFlow model at every step).
        out = run_mpc(fleet.copy(), pf,
                      base_kw_hod=_BASE_LOAD_TOTAL_MW * 1000.0,
                      pv_kw_hod=np.roll(pv_sim, SIM_START_H),
                      cfg=MPC_CFG, sim_start_h=SIM_START_H)
        ev_kw_24h, soc_profile = out['ev_kw'], out['soc_profile']
        info = {'unserved_kwh': out['unserved_kwh']}
        voltages_24h, loading_24h = out['voltages'], out['line_loading']
        extra = {
            'mpc_solve_mean_s' : float(out['mpc_solve_s'].mean()),
            'mpc_solve_max_s'  : float(out['mpc_solve_s'].max()),
            'mpc_slack_max_pu' : out['mpc_slack_max'],
            'mpc_n_fail'       : int(sum(st not in ('optimal', 'optimal_inaccurate')
                                         for st in out['mpc_status'])),
            'lin_model_mae_pu' : out['lin_model_mae_pu'],
        }
    else:
        raise ValueError(f"Unknown strategy {strategy!r}")

    return {
        'penetration' : penetration,
        'strategy'    : strategy,
        'pv_enabled'  : PV_ENABLED,
        'n_ev'        : n_active,
        'ev_kw'       : ev_kw_24h,           # (24,)
        'net_load_kw' : base_load_sim + ev_kw_24h - (pv_sim if PV_ENABLED else 0.0),   # (24,)
        'voltages'    : voltages_24h,         # (24, 33)
        'line_loading': loading_24h,          # (24, 37)
        'soc_profile' : soc_profile,          # (24,)
        'unserved_kwh': info.get('unserved_kwh', 0.0),
        **extra,
    }


def compute_metrics(result: dict) -> dict:
    """
    Compute scalar summary metrics from a scenario result.

    Returns a flat dict suitable for a summary DataFrame row.
    """
    V   = result['voltages']                   # (24, 33)
    ev  = result['ev_kw']                      # (24,)

    # Mean absolute voltage deviation from 1.0 pu (all buses, all hours)
    volt_dev_mean = float(np.nanmean(np.abs(V - 1.0)))

    # Minimum voltage across all buses and hours
    volt_min = float(np.nanmin(V))

    # Number of (slot, bus) pairs with voltage outside [0.95, 1.05]
    violations = int(np.sum((V < 0.95) | (V > 1.05)))

    # EV peak load [kW]
    ev_peak = float(ev.max())

    # EV load factor: avg / peak (higher = flatter = better for G2V)
    ev_load_factor = float(ev.mean() / ev_peak) if ev_peak > 0 else 0.0

    net = result['net_load_kw']                # base + EV - PV  [kW]
    out = {
        'penetration'    : result['penetration'],
        'strategy'       : result['strategy'],
        'pv_enabled'     : result['pv_enabled'],
        'volt_dev_mean'  : volt_dev_mean,
        'volt_min_pu'    : volt_min,
        'volt_max_pu'    : float(np.nanmax(V)),
        'volt_violations': violations,
        'ev_peak_kw'     : ev_peak,
        'ev_load_factor' : ev_load_factor,
        'net_peak_kw'    : float(net.max()),
        'net_load_std_kw': float(net.std()),
        'ev_charge_kwh'  : float(np.maximum(ev, 0).sum()),
        'ev_inject_kwh'  : float(-np.minimum(ev, 0).sum()),
        'soc_end_mean'   : float(result['soc_profile'][-1]),
        'unserved_trip_kwh': float(result['unserved_kwh']),
    }
    for key in ('mpc_solve_mean_s', 'mpc_solve_max_s', 'mpc_slack_max_pu',
                'mpc_n_fail', 'lin_model_mae_pu'):
        out[key] = result.get(key, np.nan)
    return out


# ── Main entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pens", type=float, nargs="*", default=EV_PENETRATIONS,
                    help="EV penetration levels, e.g. --pens 0.2 0.5")
    ap.add_argument("--strategies", nargs="*", default=STRATEGIES, choices=STRATEGIES)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print(f"  V2G Smart Grid Simulation (G2V vs rule-based V2G vs QP-MPC)")
    print(f"  Date: {SIM_DATE}  |  PV: {'ON' if PV_ENABLED else 'OFF'}")
    print("=" * 60)

    # Generate fleet once — shared across all scenarios
    fleet_full = generate_fleet(seed=42)

    # PV profile aligned to simulation time axis
    pv_raw = get_pv_profile(date=SIM_DATE)
    pv_sim = align_to_sim_start(pv_raw, sim_start_h=SIM_START_H)

    all_results = []
    all_metrics = []

    total = len(args.pens) * len(args.strategies)
    done  = 0

    for pen in args.pens:
        for strat in args.strategies:
            done += 1
            label = f"EV={int(pen*100)}%  strategy={strat}  PV={'ON' if PV_ENABLED else 'OFF'}"
            print(f"\n[{done}/{total}] Running: {label} ...", flush=True)

            result  = run_scenario(pen, strat, fleet_full, pv_sim)
            metrics = compute_metrics(result)

            all_results.append(result)
            all_metrics.append(metrics)

            print(f"       volt_dev={metrics['volt_dev_mean']:.4f} pu  "
                  f"vmin={metrics['volt_min_pu']:.4f}  "
                  f"violations={metrics['volt_violations']}  "
                  f"ev_peak={metrics['ev_peak_kw']:.0f} kW  "
                  f"net_peak={metrics['net_peak_kw']:.0f} kW")

    # ── Save metrics summary ──────────────────────────────────────────────────
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv("results/metrics_summary_2500kW_min40soc.csv", index=False)
    print("\n\nMetrics saved → results/metrics_summary_2500kW_min40soc.csv")
    print(df_metrics.to_string(index=False))

    # ── Save per-scenario voltage timeseries ──────────────────────────────────
    for r in all_results:
        fname = (f"results/voltages_"
                 f"EV{int(r['penetration']*100)}_"
                 f"{r['strategy']}_"
                 f"PV{'ON' if r['pv_enabled'] else 'OFF'}_2500kW_min40soc.csv")
        df_v = pd.DataFrame(r['voltages'], columns=[f'bus_{i}' for i in range(33)])
        df_v.insert(0, 'slot', range(24))
        df_v.insert(1, 'hour', [(SIM_START_H + s) % 24 for s in range(24)])
        df_v.to_csv(fname, index=False)

    print("\nVoltage CSVs saved → results/")

    # ── Generate plots ────────────────────────────────────────────────────────
    if not args.no_plots:
        print("\nGenerating plots ...")
        plots.plot_all(all_results, pv_sim, _BASE_LOAD_TOTAL_MW, SIM_START_H)
        print("Done. Check results/ for all output files.")
