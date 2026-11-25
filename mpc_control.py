"""
mpc_control.py
--------------
QP-based Model Predictive Control (MPC) for the V2G study.

Pieces
------
1. LinDistFlowModel : quasi-steady-state voltage-sensitivity "digital twin" of the
                      radial IEEE 33-bus feeder
                          V_{k+j} = V_k^AC + S_P dP + S_Q dQ
                      The model is re-centred at EVERY control step on a fresh exact AC
                      power-flow measurement V_k^AC from pandapower.
2. V2GMPC           : the cvxpy/OSQP QP (per-EV charge/discharge, soft voltage corridor).
                      Built once, solved every step through cvxpy Parameters (DPP).
3. run_mpc          : closed-loop receding-horizon simulation (plant = the same fleet
                      model used by G2V / rule-based V2G, AC power flow = ground truth).

Conventions
-----------
* Powers in kW, positive = draw from the grid (load convention); discharge is p_dis >= 0.
* Slot k <-> hour (SIM_START_H + k) % 24.  Horizon Np = 24 h with dt = 1 h.
  Forecasts (base load, PV, plug-in windows) are periodic with period 24 h, so the horizon
  naturally spans the next morning's departure.
* Plant timing per slot (identical to ev_model.simulate_v2g):
    (1) commute-energy drops on plug-out transitions, (2) charge / discharge.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandapower.networks as pn

from ev_model import (E_BAT, P_CHARGE, P_DISCHARGE, SOC_MIN, SOC_MAX,
                      ETA_CH, ETA_DIS, SIM_START_H,
                      FleetSchedule, apply_commute_drops, _precharge_to_sim_start)
from network_setup import nominal_load_weights, LOAD_BUSES


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class MPCConfig:
    horizon: int = 24              # Np  [steps]
    dt_h: float = 1.0              # step length [h]

    soc_dep_target: float = 0.90   # SoC required at home departure (hard constraint)
    v_ref: float = 1.0
    v_min: float = 0.95            # soft corridor
    v_max: float = 1.05

    # objective weights (all terms are normalised, see V2GMPC._build)
    w_v: float = 1.0e3             # ||V - Vref||^2         [pu^2, summed over buses & steps]
    w_load: float = 1.0e2          # net-load variance      [MW^2, summed over steps]
    w_deg: float = 0.1             # sum P_dis^2            [(P/P_DISCHARGE)^2]
    w_ch: float = 0.01             # tiny sum P_ch^2        (spreads charging, strict convexity)
    w_slack_l2: float = 1.0e6      # slack quadratic penalty
    w_slack_l1: float = 1.0e4      # slack linear (exact-penalty) term

    osqp_eps: float = 1.0e-4
    osqp_max_iter: int = 20000


# ══════════════════════════════════════════════════════════════════════════════
# 1. LinDistFlow sensitivity model
# ══════════════════════════════════════════════════════════════════════════════
class LinDistFlowModel:
    """
    LinDistFlow on a radial feeder.  For bus i, load change dP_m (kW) at bus m:

        d(V_i^2) = -2 * sum_{l in path(i) ∩ path(m)} (R_l dP_m + X_l dQ_m)   [pu]
        dV_i    ~= d(V_i^2) / 2           (V ~ 1 pu)

    so  S_P[i, m] = -R_common(i, m),  S_Q[i, m] = -X_common(i, m)   [pu / kW].
    Only the *sensitivities* are constant; the operating point V^AC is re-measured
    at each MPC step (see run_mpc).
    """

    def __init__(self, net=None):
        net = net if net is not None else pn.case33bw()
        self.n_bus = len(net.bus)
        root = int(net.ext_grid.bus.iloc[0])
        lines = net.line[net.line.in_service]

        z_base = float(net.bus.vn_kv.iloc[root]) ** 2 / float(net.sn_mva)    # [ohm]
        r_pu = (lines.r_ohm_per_km * lines.length_km / lines.parallel).values / z_base
        x_pu = (lines.x_ohm_per_km * lines.length_km / lines.parallel).values / z_base

        # path incidence A[bus, line] = 1 if line lies on the path root -> bus
        A = np.zeros((self.n_bus, len(lines)))
        adj = {b: [] for b in range(self.n_bus)}
        for pos, (fb, tb) in enumerate(zip(lines.from_bus.values, lines.to_bus.values)):
            adj[int(fb)].append((int(tb), pos))
            adj[int(tb)].append((int(fb), pos))
        seen, queue = {root}, deque([root])
        while queue:
            cur = queue.popleft()
            for nxt, pos in adj[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    A[nxt] = A[cur]
                    A[nxt, pos] = 1.0
                    queue.append(nxt)
        if len(seen) != self.n_bus:
            raise ValueError("Network is not a connected radial feeder.")

        kw_to_pu = 1.0 / (1000.0 * float(net.sn_mva))
        self.S_P = -(A * r_pu) @ A.T * kw_to_pu     # (n_bus, n_bus)  pu per kW of load at bus m
        self.S_Q = -(A * x_pu) @ A.T * kw_to_pu     # pu per kvar
        self.load_bus = net.load.bus.values.astype(int)

    def aggregate_sensitivity(self, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Sensitivity (n_bus,) of every bus voltage to ONE kW of *total* fleet power that is
        split over the load buses with `weights` (same split used by the plant).
        """
        s_p = self.S_P[:, self.load_bus] @ weights
        s_q = self.S_Q[:, self.load_bus] @ weights
        return s_p, s_q


# ══════════════════════════════════════════════════════════════════════════════
# 2. The QP
# ══════════════════════════════════════════════════════════════════════════════
class V2GMPC:
    """
    Decision variables (per EV i, step j):  p_ch >= 0,  p_dis >= 0,  y = SoC at end of slot j.
    Aggregate fleet power  P_ev,j = sum_i (p_ch - p_dis).

    SoC dynamics (x = SoC after commute drops, "drop" = commute fraction at plug-out):
        y_j = y_{j-1} - drop_j + dt/E * (eta_ch p_ch - p_dis / eta_dis),   y_{-1} = soc_now
    Constraints
        0 <= p_ch <= P_CHARGE * avail,  0 <= p_dis <= P_DISCHARGE * avail   (avail = plugged in)
        lb_j <= y_j <= SOC_MAX     (lb = 40 % floor incl. coming commute drop,
                                    and target SoC on the step before home departure)
        V_j >= v_min - eps_lo,  V_j <= v_max + eps_hi,  eps >= 0              (soft)
    Grid model
        V_j = V_AC + s_P (c_dv_j + P_ev,j) + s_Q dQ_j    (re-centred each step via V_AC, c_dv)
    Objective
        w_v ||V - 1||^2 + w_load Var(net load)/1e6 + w_deg sum (p_dis/Pd)^2 + w_ch sum (p_ch/Pc)^2
        + slack penalties
    """

    def __init__(self, n_ev: int, s_p: np.ndarray, s_q: np.ndarray, cfg: MPCConfig):
        self.n, self.cfg = n_ev, cfg
        self.s_p, self.s_q = s_p, s_q
        self.nb = len(s_p)
        self._build()

    def _build(self):
        n, Np, nb, cfg = self.n, self.cfg.horizon, self.nb, self.cfg
        dt = cfg.dt_h

        # variables
        self.p_ch = cp.Variable((n, Np), nonneg=True)
        self.p_dis = cp.Variable((n, Np), nonneg=True)
        self.y = cp.Variable((n, Np))
        self.p_ev = cp.Variable(Np)
        self.eps_lo = cp.Variable((nb, Np), nonneg=True)
        self.eps_hi = cp.Variable((nb, Np), nonneg=True)

        # parameters (updated every step -> problem compiled once, DPP)
        self.soc0 = cp.Parameter(n)
        self.avail = cp.Parameter((n, Np), nonneg=True)
        self.drop = cp.Parameter((n, Np), nonneg=True)
        self.lb = cp.Parameter((n, Np))
        self.v_ac = cp.Parameter(nb)            # exact AC measurement  (re-centring)
        self.c_dv = cp.Parameter(Np)            # net-load change vs. measured operating point [kW]
        self.dq = cp.Parameter(Np)              # reactive change vs. operating point [kvar]
        self.c_load = cp.Parameter(Np)          # base - PV forecast [kW]

        # grid digital twin: V (nb x Np)
        s_p, s_q = self.s_p.reshape(-1, 1), self.s_q.reshape(-1, 1)
        v_pred = (cp.reshape(self.v_ac, (nb, 1), order="F") @ np.ones((1, Np))
                  + s_p @ cp.reshape(self.c_dv + self.p_ev, (1, Np), order="F")
                  + s_q @ cp.reshape(self.dq, (1, Np), order="F"))
        self.v_pred = v_pred

        # battery model
        y_prev = cp.hstack([cp.reshape(self.soc0, (n, 1), order="F"), self.y[:, :-1]])
        d_energy = (dt / E_BAT) * (ETA_CH * self.p_ch - self.p_dis / ETA_DIS)

        cons = [
            self.y == y_prev - self.drop + d_energy,
            self.y <= SOC_MAX,
            self.y >= self.lb,
            self.p_ch <= P_CHARGE * self.avail,
            self.p_dis <= P_DISCHARGE * self.avail,
            self.p_ev == cp.sum(self.p_ch, axis=0) - cp.sum(self.p_dis, axis=0),
            v_pred >= cfg.v_min - self.eps_lo,
            v_pred <= cfg.v_max + self.eps_hi,
        ]

        net_load = self.c_load + self.p_ev                                  # kW
        load_var = cp.sum_squares((net_load - cp.sum(net_load) / Np) / 1000.0)  # MW^2

        obj = (cfg.w_v * cp.sum_squares(v_pred - cfg.v_ref)
               + cfg.w_load * load_var
               + cfg.w_deg * cp.sum_squares(self.p_dis / P_DISCHARGE)
               + cfg.w_ch * cp.sum_squares(self.p_ch / P_CHARGE)
               + cfg.w_slack_l2 * (cp.sum_squares(self.eps_lo) + cp.sum_squares(self.eps_hi))
               + cfg.w_slack_l1 * (cp.sum(self.eps_lo) + cp.sum(self.eps_hi)))

        self.prob = cp.Problem(cp.Minimize(obj), cons)
        assert self.prob.is_dcp(dpp=True), "MPC problem is not DPP-compliant"

    def solve(self, soc0, avail, drop, lb, v_ac, c_dv, dq, c_load):
        self.soc0.value = soc0
        self.avail.value = avail
        self.drop.value = drop
        self.lb.value = lb
        self.v_ac.value = v_ac
        self.c_dv.value = c_dv
        self.dq.value = dq
        self.c_load.value = c_load

        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True,
                            eps_abs=self.cfg.osqp_eps, eps_rel=self.cfg.osqp_eps,
                            max_iter=self.cfg.osqp_max_iter, polish=False)
            status = self.prob.status
        except cp.error.SolverError:
            status = "solver_error"
        dt = time.perf_counter() - t0

        ok = status in ("optimal", "optimal_inaccurate") and self.p_ch.value is not None
        out = {"status": status, "solve_s": dt, "ok": ok}
        if ok:
            out["p_ch0"] = np.asarray(self.p_ch.value[:, 0])
            out["p_dis0"] = np.asarray(self.p_dis.value[:, 0])
            out["v_pred0"] = v_ac + self.s_p * (c_dv[0] + float(self.p_ev.value[0])) + self.s_q * dq[0]
            out["slack_max"] = float(max(self.eps_lo.value.max(), self.eps_hi.value.max()))
        return out


# ══════════════════════════════════════════════════════════════════════════════
# 3. Plant + closed loop
# ══════════════════════════════════════════════════════════════════════════════
class _FleetPlant:
    """Vectorised fleet dynamics; uses the SAME FleetSchedule / commute logic as G2V and RB-V2G."""

    def __init__(self, fleet, sim_start_h: int):
        fleet = fleet.reset_index(drop=True)
        pre = _precharge_to_sim_start(fleet)
        self.sch = FleetSchedule(pre)
        self.n = self.sch.n
        self.soc = pre["soc_current"].to_numpy(float).copy()
        self.leg = self.sch.leg
        self.home, self.work, self.avail = self.sch.home, self.sch.work, self.sch.avail
        self.unserved_kwh = 0.0

    def start_slot(self, h: int):
        """Apply commute-energy drops on plug-out transitions at the start of hour h."""
        self.soc, lost = apply_commute_drops(self.soc, self.sch, h)
        self.unserved_kwh += lost

    def apply(self, p_ch, p_dis, dt_h):
        self.soc = np.clip(self.soc + dt_h / E_BAT * (ETA_CH * p_ch - p_dis / ETA_DIS),
                           SOC_MIN, SOC_MAX)


def _horizon_inputs(plant: _FleetPlant, h0: int, cfg: MPCConfig):
    """Periodic 24 h forecasts of plug-in state, commute drops and SoC bounds for the QP."""
    Np, n = cfg.horizon, plant.n
    hrs = (h0 + np.arange(Np + 1)) % 24
    prv = (hrs - 1) % 24

    dep_home = plant.home[prv] & ~plant.home[hrs]                           # (Np+1, n)
    dep_work = plant.work[prv] & ~plant.work[hrs]
    drop_full = plant.leg[None, :] * (dep_home.astype(float) + dep_work.astype(float))

    avail = plant.avail[hrs[:Np]].T.astype(float)                           # (n, Np)
    drop = drop_full[:Np].T.copy()                                          # (n, Np)
    drop[:, 0] = 0.0                                                        # slot-0 drop already applied

    # lower bound on y_j (end of slot j, i.e. just before transition j+1)
    nxt_drop = drop_full[1:Np + 1].T                                        # (n, Np)
    nxt_dep = dep_home[1:Np + 1].T
    lb = np.maximum(SOC_MIN, SOC_MIN + nxt_drop)                            # still >= 40 % after the drop
    lb = np.maximum(lb, np.where(nxt_dep, cfg.soc_dep_target, 0.0))         # departure target

    # feasibility guard: never ask for more than max-charging can reach
    prev, ymax = plant.soc.copy(), np.empty((n, Np))
    for j in range(Np):
        ymax[:, j] = np.minimum(SOC_MAX, prev - drop[:, j]
                                + avail[:, j] * P_CHARGE * ETA_CH * cfg.dt_h / E_BAT)
        prev = ymax[:, j]
    lb = np.minimum(lb, ymax - 1e-4)
    return avail, drop, lb, hrs[:Np]


def _fallback_action(plant: _FleetPlant, h: int):
    """Uncontrolled charging for one step if the QP fails (should be rare)."""
    ok = plant.avail[h] & (plant.soc < SOC_MAX - 1e-6)
    return np.where(ok, P_CHARGE, 0.0), np.zeros(plant.n)


def run_mpc(fleet, pf_runner, base_kw_hod, pv_kw_hod, cfg: MPCConfig | None = None,
            sim_start_h: int = SIM_START_H, verbose: bool = False) -> dict:
    """
    Closed-loop receding-horizon V2G control over one simulated day (24 steps).

    pf_runner(slot, ev_kw) -> (V[33], loading[37])   exact AC power flow (pandapower)
    base_kw_hod / pv_kw_hod : (24,) total base load / PV [kW] by hour of day (forecast = truth)

    Each step k:
      1. plant applies commute drops;
      2. AC power flow at the last applied EV set-point  ->  V_k^AC   (re-centring);
      3. QP over Np steps, first move applied;
      4. AC power flow with the applied set-point  ->  ground-truth voltages.
    """
    cfg = cfg or MPCConfig()
    plant = _FleetPlant(fleet, sim_start_h)
    n = plant.n

    model = LinDistFlowModel()
    s_p, s_q = model.aggregate_sensitivity(nominal_load_weights())
    ctrl = V2GMPC(n, s_p, s_q, cfg)

    base_kw_hod = np.asarray(base_kw_hod, float)
    pv_kw_hod = np.asarray(pv_kw_hod, float)

    nb = model.n_bus
    ev_kw = np.zeros(24)
    soc_profile = np.zeros(24)
    volts, loads = np.zeros((24, nb)), None
    v_pred = np.full((24, nb), np.nan)
    status, solve_s, slack_max = [], [], []
    ev_prev, v_last = 0.0, np.ones(nb)

    for k in range(24):
        h = (sim_start_h + k) % 24
        plant.start_slot(h)

        # --- (2) exact AC measurement, EV set-point still at its previous value
        v_ac, _ = pf_runner(k, ev_prev)
        if np.isnan(v_ac).any():
            v_ac = v_last
        v_last = v_ac

        # --- forecasts over the horizon (periodic)
        avail, drop, lb, hrs = _horizon_inputs(plant, h, cfg)
        c_load = base_kw_hod[hrs] - pv_kw_hod[hrs]
        c_dv = c_load - (base_kw_hod[h] - pv_kw_hod[h]) - ev_prev
        dq = np.zeros(cfg.horizon)           # EVs at unity pf, base Q constant -> no reactive change

        # --- (3) QP
        sol = ctrl.solve(plant.soc, avail, drop, lb, v_ac, c_dv, dq, c_load)
        status.append(sol["status"])
        solve_s.append(sol["solve_s"])
        if sol["ok"]:
            p_ch, p_dis = sol["p_ch0"], sol["p_dis0"]
            slack_max.append(sol["slack_max"])
            v_pred[k] = sol["v_pred0"]
        else:
            print(f"  [WARN] MPC step {k} ({h:02d}:00): QP status '{sol['status']}' -> uncontrolled fallback")
            p_ch, p_dis = _fallback_action(plant, h)

        # clean numerical noise, respect plug-in state and bounds, never charge+discharge together
        on = plant.avail[h]
        p_ch = np.clip(p_ch, 0.0, P_CHARGE) * on
        p_dis = np.clip(p_dis, 0.0, P_DISCHARGE) * on
        both = np.minimum(p_ch, p_dis)
        p_ch, p_dis = p_ch - both, p_dis - both
        p_ch[p_ch < 1e-3] = 0.0
        p_dis[p_dis < 1e-3] = 0.0

        plant.apply(p_ch, p_dis, cfg.dt_h)
        ev_kw[k] = p_ch.sum() - p_dis.sum()
        soc_profile[k] = plant.soc.mean()

        # --- (4) ground truth
        volts[k], ld = pf_runner(k, ev_kw[k])
        if loads is None:
            loads = np.zeros((24, len(ld)))
        loads[k] = ld
        ev_prev = ev_kw[k]

        if verbose:
            print(f"  k={k:2d} {h:02d}:00  EV={ev_kw[k]:8.1f} kW  SoC={soc_profile[k]:.3f}  "
                  f"{sol['status']}  {sol['solve_s']:.2f}s")

    err = np.abs(v_pred - volts)
    return {
        "ev_kw": ev_kw,
        "soc_profile": soc_profile,
        "voltages": volts,
        "line_loading": loads,
        "v_pred": v_pred,
        "mpc_status": status,
        "mpc_solve_s": np.array(solve_s),
        "mpc_slack_max": float(max(slack_max)) if slack_max else 0.0,
        "lin_model_mae_pu": float(np.nanmean(err)),
        "final_soc": plant.soc.copy(),
        "unserved_kwh": plant.unserved_kwh,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Self-test:  python mpc_control.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import numpy as _np
    from ev_model import generate_fleet
    from network_setup import make_pf_runner
    from pv_model import get_pv_profile, align_to_sim_start

    BASE_MW = _np.array([0.8, 0.7, 0.65, 0.6, 0.6, 0.65, 0.8, 1.0, 1.3, 1.5, 1.6, 1.65,
                         1.7, 1.65, 1.6, 1.5, 1.6, 1.8, 2.0, 2.1, 2.0, 1.8, 1.5, 1.1])
    pv_raw = get_pv_profile()
    pv_sim = align_to_sim_start(pv_raw, SIM_START_H)
    pf = make_pf_runner(BASE_MW, pv_sim, SIM_START_H)

    # (a) LinDistFlow vs. AC: perturb the fleet power around an operating point
    model = LinDistFlowModel()
    s_p, _ = model.aggregate_sensitivity(nominal_load_weights())
    v0, _ = pf(12, 0.0)
    print("LinDistFlow vs AC (slot 12, perturbation of total fleet power):")
    for dp in (-500, 500, 1000, 2000):
        v_ac, _ = pf(12, dp)
        v_lin = v0 + s_p * dp
        print(f"  dP={dp:6d} kW   max |err| = {_np.abs(v_ac - v_lin).max():.4f} pu   "
              f"(AC dV_min = {(v_ac - v0).min():+.4f} pu)")

    # (b) small closed-loop run
    fleet = generate_fleet(42).iloc[:60].reset_index(drop=True)
    res = run_mpc(fleet, pf, BASE_MW, pv_raw, verbose=True)
    print("statuses:", set(res["mpc_status"]), "| lin-model MAE [pu]:", round(res["lin_model_mae_pu"], 5))
