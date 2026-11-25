"""
check_consistency.py
--------------------
Sanity checks that G2V, rule-based V2G and MPC-V2G see exactly the same fleet
(schedule, commute energy, SoC bookkeeping).

    python check_consistency.py
"""

import numpy as np

import ev_model as ev
import mpc_control as mpc
from ev_model import FleetSchedule, apply_commute_drops, generate_fleet, SIM_START_H, E_BAT, SOC_MIN

ok_all = True


def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'OK ' if cond else 'FAIL'}] {name} {detail}")


fleet = generate_fleet(42)
n = len(fleet)

# ── 1. Schedule integrity ────────────────────────────────────────────────────
print("1. Schedule")
sch = FleetSchedule(fleet)
check("never plugged at home and at work at once", not (sch.home & sch.work).any())

# sequence of states over the day, starting at 00:00
n_out_home = sum(int(sch.drop_masks(h)[0].sum()) for h in range(24))
n_out_work = sum(int(sch.drop_masks(h)[1].sum()) for h in range(24))
check("exactly one home plug-out and one work plug-out per EV per day",
      n_out_home == n and n_out_work == n, f"({n_out_home}, {n_out_work} for {n} EVs)")

away = ~sch.avail                       # (24, n) unplugged slots
dep = fleet.departure_h.values.astype(int)
ws, we, arr = (fleet[c].values.astype(int) for c in ("work_start_h", "work_end_h", "arrival_h"))
travel1 = np.array([away[dep[i], i] and not away[(dep[i] - 1) % 24, i] and away[ws[i] - 1, i] for i in range(n)])
check("travel slot between departure and work arrival = 1 h", np.all(ws - dep == 1) and travel1.all())
check("travel slot between work exit and home arrival = 1 h", np.all(arr - we == 1))
check("work exit window 17..19", we.min() >= 17 and we.max() <= 19, f"(min {we.min()}, max {we.max()})")
check("home departure window 6..9 (so work arrival 7..10)", dep.min() >= 6 and dep.max() <= 9)
check("home arrival 18..20", arr.min() >= 18 and arr.max() <= 20, f"(min {arr.min()}, max {arr.max()})")
check("unplugged slots per EV = 2 travel + 0 other (= 2 h/day)", np.all(away.sum(axis=0) == 2),
      f"(min {away.sum(0).min()}, max {away.sum(0).max()})")

# ── 2. Zero-power drive test: the three simulators must give identical SoC ────
print("2. Zero-power run (all strategies must give identical mean SoC every slot)")


def _full(fl):
    fl = fl.copy()
    fl["soc_current"] = 0.95
    return fl


saved = (ev.P_CHARGE, ev.P_DISCHARGE, ev._precharge_to_sim_start, mpc._precharge_to_sim_start)
ev.P_CHARGE = ev.P_DISCHARGE = 0.0
ev._precharge_to_sim_start = mpc._precharge_to_sim_start = _full
try:
    p1, s_g2v, i1 = ev.simulate_g2v(fleet)
    p2, s_rb, i2 = ev.simulate_v2g(fleet, np.ones(24), np.zeros(24))
    plant = mpc._FleetPlant(fleet, SIM_START_H)
    s_mpc = np.zeros(24)
    z = np.zeros(n)
    soc_log = []
    for k in range(24):
        plant.start_slot((SIM_START_H + k) % 24)
        plant.apply(z, z, 1.0)
        s_mpc[k] = plant.soc.mean()
        soc_log.append(plant.soc.copy())
finally:
    ev.P_CHARGE, ev.P_DISCHARGE, ev._precharge_to_sim_start, mpc._precharge_to_sim_start = saved

check("G2V == RB-V2G", np.allclose(s_g2v, s_rb, atol=1e-12), f"(max diff {np.abs(s_g2v - s_rb).max():.1e})")
check("G2V == MPC plant", np.allclose(s_g2v, s_mpc, atol=1e-12), f"(max diff {np.abs(s_g2v - s_mpc).max():.1e})")
check("no trip energy lost to the SoC floor in the pure-driving run",
      i1["unserved_kwh"] == 0 and i2["unserved_kwh"] == 0 and plant.unserved_kwh == 0)
day_drop = (0.95 - np.array(soc_log[-1])) * E_BAT
check("each EV spends exactly 2 commute legs per day",
      np.allclose(day_drop, 2 * fleet.commute_kwh_leg.values, atol=1e-9))

# ── 3. MPC internal model predicts the plant's commute drops ────────────────
print("3. MPC prediction of commute drops vs plant")
cfg = mpc.MPCConfig()
err = 0.0
plant = mpc._FleetPlant(_full(fleet), SIM_START_H)
traj = np.zeros((24, n))
for k in range(24):
    plant.start_slot((SIM_START_H + k) % 24)
    traj[k] = plant.soc
plant2 = mpc._FleetPlant(_full(fleet), SIM_START_H)
for k in range(24):
    h = (SIM_START_H + k) % 24
    plant2.start_slot(h)
    _, drop, _, _ = mpc._horizon_inputs(plant2, h, cfg)
    for j in range(1, 24 - k):
        pred = plant2.soc - drop[:, 1:j + 1].sum(axis=1)
        err = max(err, np.abs(pred - traj[k + j]).max())
check("predicted SoC (soc_k - cumulative drops) == plant SoC for the rest of the day", err < 1e-12,
      f"(max err {err:.1e})")

print("\nALL CHECKS PASSED" if ok_all else "\nSOME CHECKS FAILED")
