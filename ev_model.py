"""EV fleet model and state of charge logic."""

import numpy as np
import pandas as pd

# Physical parameters
N_EV         = 500       # number of EVs in the fleet
E_BAT        = 60.0      # battery capacity [kWh]
P_CHARGE     = 7.4       # AC wallbox charge power [kW] (Mode 2)
P_DISCHARGE  = 5.0       # V2G discharge power [kW] (slightly lower)
SOC_MIN      = 0.40      # minimum SoC to protect battery
SOC_MAX      = 0.95      # maximum SoC (avoid 100 %)
EFFICIENCY   = 0.92      # round-trip charge/discharge efficiency
ETA_CH       = EFFICIENCY ** 0.5   # one-way charge efficiency   (~0.959)
ETA_DIS      = EFFICIENCY ** 0.5   # one-way discharge efficiency (~0.959)
SIM_START_H  = 6         # simulation starts at 06:00
DT_H         = 1.0       # slot length [h]

# ── Mobility schedule (integer hours, shared by ALL strategies) ──────────────
#   home departure  : ~N(8, 1) rounded, clipped to [6, 9]
#   travel home->work: TRAVEL_H  (EV unplugged, 1 commute leg of energy)
#   work window     : [work_start, work_end) with work_start = departure + TRAVEL_H
#   work exit       : uniform integer in WORK_END_RANGE (17..19)
#   travel work->home: TRAVEL_H  (EV unplugged, 1 commute leg of energy)
#   home arrival    : work_end + TRAVEL_H  -> plugged in again at home
TRAVEL_H       = 1
DEP_RANGE      = (6, 9)       # inclusive
WORK_END_RANGE = (17, 19)     # inclusive

V_HIGH      = 1.04   # overvoltage threshold
V_LOW       = 0.96   # undervoltage threshold
V_DEADBAND  = 0.005  # hysteresis: do not switch modes near thresholds


def generate_fleet(seed=42):
    """
    Create a DataFrame of 500 EVs with a consistent daily mobility schedule.
    Columns: arrival_h (home plug-in), departure_h (home plug-out), work_start_h,
             work_end_h, soc_arrival, soc_current, commute_kwh_leg.
    """
    rng = np.random.default_rng(seed)

    departure   = np.clip(np.rint(rng.normal(loc=8.0, scale=1.0, size=N_EV)),
                          DEP_RANGE[0], DEP_RANGE[1]).astype(int)
    work_end    = rng.integers(WORK_END_RANGE[0], WORK_END_RANGE[1] + 1, size=N_EV)
    soc_arrival = rng.uniform(low=SOC_MIN, high=0.65, size=N_EV)

    work_start = departure + TRAVEL_H
    arrival    = work_end + TRAVEL_H

    commute_kwh_total = (SOC_MAX - soc_arrival) * E_BAT
    commute_kwh_leg = commute_kwh_total / 2.0

    fleet = pd.DataFrame({
        "arrival_h": arrival.astype(float),
        "departure_h": departure.astype(float),
        "work_start_h": work_start.astype(float),
        "work_end_h": work_end.astype(float),
        "soc_arrival": soc_arrival,
        "soc_current": soc_arrival.copy(),
        "commute_kwh_leg": commute_kwh_leg,
    })
    return fleet


class FleetSchedule:
    """
    Plug-in schedule shared by G2V, rule-based V2G and MPC (single source of truth).

    home[h, i] / work[h, i] : EV i is plugged in at home / at work during slot h (hour of day)
    A commute leg of energy (leg[i], SoC fraction) is consumed at every plug-out
    transition (home->away at departure, work->away at work exit); transitions are
    evaluated between hour h-1 and h, so the very first slot of the simulation is
    handled correctly too (an EV leaving at 06:00 pays its leg at slot 0).
    """

    def __init__(self, fleet: pd.DataFrame):
        arr = fleet["arrival_h"].to_numpy(float)[None, :]
        dep = fleet["departure_h"].to_numpy(float)[None, :]
        ws = fleet["work_start_h"].to_numpy(float)[None, :]
        we = fleet["work_end_h"].to_numpy(float)[None, :]
        hrs = np.arange(24)[:, None]

        self.n = fleet.shape[0]
        self.home = (hrs >= arr) | (hrs < dep)
        self.work = (hrs >= ws) & (hrs < we)
        if (self.home & self.work).any():
            raise ValueError("Inconsistent schedule: EV plugged at home and at work simultaneously")
        self.avail = self.home | self.work
        self.leg = fleet["commute_kwh_leg"].to_numpy(float) / E_BAT

    def drop_masks(self, h: int):
        """(plug-out from home, plug-out from work) masks at the start of slot h."""
        p = (h - 1) % 24
        return self.home[p] & ~self.home[h], self.work[p] & ~self.work[h]


def apply_commute_drops(soc: np.ndarray, sch: FleetSchedule, h: int):
    """
    Apply commute energy at slot h. Returns (new_soc, unserved_kwh), where unserved_kwh is
    the energy that could not be drawn because SoC hit SOC_MIN (the clip would otherwise
    create energy out of nothing; it is reported as a metric).
    """
    soc = soc.copy()
    lost = 0.0
    for gone in sch.drop_masks(h):
        new = soc - sch.leg * gone
        clipped = np.maximum(new, SOC_MIN)
        lost += float(((clipped - new) * E_BAT).sum())
        soc = clipped
    return soc, lost


def _precharge_to_sim_start(fleet: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute SoC at SIM_START_H (06:00), simulating overnight charging
    from arrival_h until 06:00 next morning.

    After this, soc_current is near SOC_MAX for most EVs.
    The simulation then correctly shows near-zero load in the morning.
    """
    fleet = fleet.copy()
    for i in range(len(fleet)):
        arr = fleet.at[i, 'arrival_h']
        soc = fleet.at[i, 'soc_arrival']

        hours_charging = int((24 - arr) + SIM_START_H)

        for _ in range(hours_charging):
            if soc >= SOC_MAX:
                break
            energy_needed  = (SOC_MAX - soc) * E_BAT
            energy_charged = min(P_CHARGE * ETA_CH, energy_needed)
            soc += energy_charged / E_BAT

        fleet.at[i, 'soc_current'] = np.clip(soc, SOC_MIN, SOC_MAX)

    return fleet


def simulate_g2v(fleet: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Uncontrolled G2V: every EV charges at full power whenever it is plugged in, at home
    or at work (no control, no V2G).

    Returns (total_power_kW[24], mean_soc[24], info) with info['unserved_kwh'].
    """
    fleet = _precharge_to_sim_start(fleet.reset_index(drop=True))
    sch = FleetSchedule(fleet)
    soc = fleet["soc_current"].to_numpy(float).copy()

    total_power = np.zeros(24)
    soc_profile = np.zeros(24)
    unserved = 0.0

    for slot in range(24):
        h = (SIM_START_H + slot) % 24
        soc, lost = apply_commute_drops(soc, sch, h)
        unserved += lost

        need = np.maximum(SOC_MAX - soc, 0.0) * E_BAT                    # kWh to full
        stored = np.where(sch.avail[h], np.minimum(P_CHARGE * ETA_CH * DT_H, need), 0.0)
        soc = soc + stored / E_BAT
        total_power[slot] = float((stored / (ETA_CH * DT_H)).sum())      # grid draw [kW]
        soc_profile[slot] = float(soc.mean())

    return total_power, soc_profile, {"unserved_kwh": unserved}


def simulate_v2g(
    fleet: pd.DataFrame,
    v_pu_profile: np.ndarray,
    pv_surplus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Rule-based (heuristic) V2G: each EV charges/discharges based on bus voltage and PV surplus.

    v_pu_profile : array (24,) avg. voltage in p.u.
    pv_surplus   : array (24,) excess PV over base load [kW]
    Returns (net fleet power [kW] (24,), mean_soc (24,), info): positive = draw, negative = injection.
    """
    fleet = _precharge_to_sim_start(fleet.reset_index(drop=True))
    sch = FleetSchedule(fleet)
    soc = fleet["soc_current"].to_numpy(float).copy()
    n = sch.n

    total_power = np.zeros(24)
    soc_profile = np.zeros(24)
    prev_mode = np.array(["idle"] * n, dtype=object)
    unserved = 0.0

    for slot in range(24):
        h = (SIM_START_H + slot) % 24
        v = v_pu_profile[slot]
        ps = pv_surplus[slot]

        soc, lost = apply_commute_drops(soc, sch, h)
        unserved += lost

        home, work = sch.home[h], sch.work[h]
        n_connected = int((home | work).sum())

        for i in range(n):
            if not (home[i] or work[i]):
                continue

            s = soc[i]
            headroom_kw = max(SOC_MAX - s, 0.0) * E_BAT / (ETA_CH * DT_H)   # max useful charge power

            if work[i]:
                # PV-driven charging only during work hours.
                if ps > 0 and s < SOC_MAX:
                    p = min(P_CHARGE * 0.6, ps / max(n_connected, 1), headroom_kw)
                    soc[i] += (p * ETA_CH * DT_H) / E_BAT
                    prev_mode[i] = "charge"
                else:
                    p = 0.0
                    prev_mode[i] = "idle"
            else:
                if prev_mode[i] == "charge" and v >= V_HIGH - V_DEADBAND and s < SOC_MAX:
                    p = min(P_CHARGE, headroom_kw)
                    soc[i] += (p * ETA_CH * DT_H) / E_BAT
                    mode = "charge"
                elif prev_mode[i] == "discharge" and v <= V_LOW + V_DEADBAND and s > SOC_MIN + 0.1:
                    p = -P_DISCHARGE
                    soc[i] -= P_DISCHARGE * DT_H / (E_BAT * ETA_DIS)
                    mode = "discharge"
                elif v > V_HIGH and s < SOC_MAX:
                    p = min(P_CHARGE, headroom_kw)
                    soc[i] += (p * ETA_CH * DT_H) / E_BAT
                    mode = "charge"
                elif v < V_LOW and s > SOC_MIN + 0.1:
                    p = -P_DISCHARGE
                    soc[i] -= P_DISCHARGE * DT_H / (E_BAT * ETA_DIS)
                    mode = "discharge"
                elif ps > 0 and s < SOC_MAX:
                    p = min(P_CHARGE * 0.6, ps / max(n_connected, 1), headroom_kw)
                    soc[i] += (p * ETA_CH * DT_H) / E_BAT
                    mode = "charge"
                elif s < SOC_MAX:
                    p = min(P_CHARGE * 0.4, headroom_kw)
                    soc[i] += (p * ETA_CH * DT_H) / E_BAT
                    mode = "charge"
                else:
                    p = 0.0
                    mode = "idle"

                prev_mode[i] = mode

            soc[i] = float(np.clip(soc[i], SOC_MIN, SOC_MAX))
            total_power[slot] += p

        soc_profile[slot] = float(soc.mean())

    return total_power, soc_profile, {"unserved_kwh": unserved}


if __name__ == "__main__":
    fleet = generate_fleet(seed=42)
    print(fleet.head(10))
    print(f"\nMean arrival SoC:  {fleet['soc_arrival'].mean():.2%}")
    print(f"Mean home arrival: {fleet['arrival_h'].mean():.1f}h")

    power_g2v, _, info = simulate_g2v(fleet)

    peak_slot = int(power_g2v.argmax())
    peak_hour = (SIM_START_H + peak_slot) % 24
    print(f"\nG2V peak load: {power_g2v.max():.1f} kW at {peak_hour:02d}:00")
    print()
    for slot in range(24):
        h   = (SIM_START_H + slot) % 24
        bar = "█" * int(power_g2v[slot] / 100)
        print(f"  {h:02d}:00  {power_g2v[slot]:6.1f} kW  {bar}")
