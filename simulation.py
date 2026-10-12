"""
simulation.py
=============
Discrete-event simulation (DES) of a small four-station factory whose parts
are moved by Automated Guided Vehicles (AGVs).

Layout (one straight route, positions in metres):

    Raw Material Store (RM) ---- Machine 1 (M1) ---- Machine 2 (M2) ---- Finished Goods (FG)
           0 m                       100 m               200 m                300 m

Every component follows RM -> M1 -> M2 -> FG.

Units used everywhere inside the model:
    time      -> minutes
    distance  -> metres
    speed     -> metres per second in the parameters, converted to m/min once

How the model is organised
--------------------------
* ``Params``            : all input assumptions in one place.
* ``Factory``           : builds the SimPy objects and contains the processes:
    - ``arrivals``      : generates components (the only random input).
    - ``component``     : the life of one component (transport -> machine -> ...).
    - ``dispatch``      : pairs waiting transport jobs with idle AGVs using
                          the chosen policy ("FIFO" or "NEAREST").
    - ``_agv_trip``     : one AGV executing one transport job.
* ``compute_metrics``   : turns the event records into the four KPIs,
                          counting only the measurement window.
* ``check_invariants``  : sanity checks run after every replication.
* ``run_replication``   : runs one (policy, fleet size, replication) case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import simpy

# ---------------------------------------------------------------------------
# 1. Layout and parameters (ILLUSTRATIVE ASSUMPTIONS, not measured data)
# ---------------------------------------------------------------------------

# Station positions along the single straight route, in metres.
STATIONS = {"RM": 0.0, "M1": 100.0, "M2": 200.0, "FG": 300.0}

# The fixed manufacturing route every component must follow.
ROUTE = ["RM", "M1", "M2", "FG"]

# The order of events we expect to see in each component's history.
EXPECTED_EVENTS = [
    ("arrive", "RM"),
    ("request", "RM"),      # transport request RM -> M1
    ("delivered", "M1"),
    ("start", "M1"),
    ("finish", "M1"),
    ("request", "M1"),      # transport request M1 -> M2
    ("delivered", "M2"),
    ("start", "M2"),
    ("finish", "M2"),
    ("request", "M2"),      # transport request M2 -> FG
    ("delivered", "FG"),
    ("complete", "FG"),
]

POLICIES = ("FIFO", "NEAREST")

# Small tolerance used when comparing floating-point times.
EPS = 1e-9


@dataclass
class Params:
    """All model inputs. Change values here (or in main.py) to run what-if studies."""

    agv_speed_mps: float = 0.8            # AGV speed, metres per second
    mean_interarrival_min: float = 4.0    # mean of exponential interarrival time
    m1_process_min: float = 7.0           # fixed processing time at Machine 1
    m2_process_min: float = 9.0           # fixed processing time at Machine 2
    pickup_min: float = 0.25              # loading time per transport job
    dropoff_min: float = 0.25             # unloading time per transport job
    sim_time_min: float = 540.0           # total simulated time
    warmup_min: float = 60.0              # excluded from statistics
    fleet_sizes: tuple = (1, 2, 3, 4, 5)
    n_replications: int = 20
    base_seed: int = 42

    @property
    def measure_min(self) -> float:
        """Length of the measurement window in minutes (540 - 60 = 480)."""
        return self.sim_time_min - self.warmup_min

    @property
    def speed_m_per_min(self) -> float:
        """Convert speed once: m/s * 60 s/min = m/min (0.8 m/s -> 48 m/min)."""
        return self.agv_speed_mps * 60.0


class SimulationError(Exception):
    """Raised when a model invariant is violated (a bug, not a result)."""


def travel_time_min(distance_m: float, speed_mps: float) -> float:
    """Travel time in MINUTES for a distance in METRES at a speed in M/S.

    t [min] = d [m] / (v [m/s] * 60 [s/min])

    Example: 100 m at 0.8 m/s -> 100 / 48 = 2.083 min (= 125 s).
    """
    if distance_m < 0:
        raise ValueError(f"Distance must be non-negative, got {distance_m}")
    if speed_mps <= 0:
        raise ValueError(f"Speed must be positive, got {speed_mps}")
    return distance_m / (speed_mps * 60.0)


def overlap(start: float, end: float, lo: float, hi: float) -> float:
    """Length of the part of interval [start, end] that lies inside [lo, hi]."""
    return max(0.0, min(end, hi) - max(start, lo))


# ---------------------------------------------------------------------------
# 2. Simple record types
# ---------------------------------------------------------------------------

@dataclass
class TransportJob:
    """One request to move one component from one station to the next."""

    job_id: int
    component_id: int
    leg: int                 # 1 = RM->M1, 2 = M1->M2, 3 = M2->FG
    pickup: str              # station name
    dest: str                # station name
    created: float           # time the request was made
    done: simpy.Event        # succeeds when the component has been dropped off
    # Filled in when an AGV is assigned (all times are known at assignment
    # because speed and handling times are deterministic):
    assigned: float | None = None
    agv_id: int | None = None
    empty_dist_m: float = 0.0
    loaded_dist_m: float = 0.0
    t_at_pickup: float | None = None     # end of empty travel
    t_loaded_start: float | None = None  # end of pickup, start of loaded travel
    t_at_dest: float | None = None       # end of loaded travel
    t_done: float | None = None          # end of drop-off (planned)
    finished: float | None = None        # end of drop-off (actually observed)


@dataclass
class AGV:
    agv_id: int
    position_m: float
    idle_since: float
    load: int | None = None   # component id currently on board, or None


@dataclass
class ComponentRecord:
    component_id: int
    arrival: float
    completed: float | None = None
    history: list = field(default_factory=list)  # (time, event, station)

    def log(self, t: float, event: str, station: str) -> None:
        self.history.append((t, event, station))


# ---------------------------------------------------------------------------
# 3. The factory model
# ---------------------------------------------------------------------------

class Factory:
    def __init__(self, env: simpy.Environment, params: Params, n_agvs: int,
                 policy: str, rng: np.random.Generator):
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy {policy!r}; choose from {POLICIES}")
        if n_agvs < 1:
            raise ValueError("Need at least one AGV")

        self.env = env
        self.p = params
        self.policy = policy
        self.rng = rng

        # Each machine can process one component at a time.
        self.machines = {
            "M1": simpy.Resource(env, capacity=1),
            "M2": simpy.Resource(env, capacity=1),
        }
        self.process_time = {"M1": params.m1_process_min, "M2": params.m2_process_min}

        # All AGVs start idle at the Raw Material Store at time 0.
        self.agvs = [AGV(agv_id=i, position_m=STATIONS["RM"], idle_since=0.0)
                     for i in range(n_agvs)]
        self.idle_agvs: list[AGV] = list(self.agvs)

        self.waiting_jobs: list[TransportJob] = []   # the dispatch queue
        self.jobs: list[TransportJob] = []           # every job ever created
        self.components: dict[int, ComponentRecord] = {}
        # (start, end, component_id) for every machining operation that started
        self.machine_log: dict[str, list] = {"M1": [], "M2": []}

    # ----- arrival process (the only random input) -------------------------
    def arrivals(self):
        """Create components with exponentially distributed interarrival times."""
        cid = 0
        while True:
            gap = self.rng.exponential(self.p.mean_interarrival_min)
            yield self.env.timeout(gap)
            self.env.process(self.component(cid))
            cid += 1

    # ----- life of one component ------------------------------------------
    def component(self, cid: int):
        rec = ComponentRecord(component_id=cid, arrival=self.env.now)
        self.components[cid] = rec
        rec.log(self.env.now, "arrive", "RM")

        yield from self.transport(rec, "RM", "M1", leg=1)
        yield from self.machine(rec, "M1")
        yield from self.transport(rec, "M1", "M2", leg=2)
        yield from self.machine(rec, "M2")
        yield from self.transport(rec, "M2", "FG", leg=3)

        # Only now - after drop-off at FG - does the component count as completed.
        rec.completed = self.env.now
        rec.log(self.env.now, "complete", "FG")

    def transport(self, rec: ComponentRecord, frm: str, to: str, leg: int):
        """Post a transport job and wait until an AGV has delivered the component."""
        job = TransportJob(job_id=len(self.jobs), component_id=rec.component_id,
                           leg=leg, pickup=frm, dest=to, created=self.env.now,
                           done=self.env.event())
        self.jobs.append(job)
        self.waiting_jobs.append(job)
        rec.log(self.env.now, "request", frm)
        self.dispatch()          # maybe an AGV is free right now
        yield job.done           # component waits in the station buffer
        rec.log(self.env.now, "delivered", to)

    def machine(self, rec: ComponentRecord, name: str):
        """Wait for the machine (unlimited input buffer), then process."""
        with self.machines[name].request() as req:
            yield req
            start = self.env.now
            end = start + self.process_time[name]
            self.machine_log[name].append((start, end, rec.component_id))
            rec.log(start, "start", name)
            yield self.env.timeout(self.process_time[name])
            rec.log(self.env.now, "finish", name)
        # Leaving the 'with' block releases the machine. The component now
        # sits in the machine's output buffer until an AGV collects it.

    # ----- dispatching ----------------------------------------------------
    def dispatch(self):
        """Assign idle AGVs to waiting jobs while both exist.

        Called whenever a new job is posted or an AGV becomes idle, so no AGV
        stays idle while a job is waiting.
        """
        while self.waiting_jobs and self.idle_agvs:
            if self.policy == "FIFO":
                agv, job = self._select_fifo()
            else:
                agv, job = self._select_nearest()
            self.waiting_jobs.remove(job)
            self.idle_agvs.remove(agv)
            self._assign(agv, job)

    def _select_fifo(self):
        """Strategy A: oldest job first; AGV that has been idle longest."""
        job = min(self.waiting_jobs, key=lambda j: (j.created, j.job_id))
        agv = min(self.idle_agvs, key=lambda a: (a.idle_since, a.agv_id))
        return agv, job

    def _select_nearest(self):
        """Strategy B: the (AGV, job) pair with the shortest empty travel.

        Ties are broken by job creation time, then AGV id
        (then job id, so the choice is always unique and reproducible).
        """
        best_key, best_pair = None, None
        for agv in self.idle_agvs:
            for job in self.waiting_jobs:
                empty = abs(agv.position_m - STATIONS[job.pickup])
                key = (empty, job.created, agv.agv_id, job.job_id)
                if best_key is None or key < best_key:
                    best_key, best_pair = key, (agv, job)
        return best_pair

    def _assign(self, agv: AGV, job: TransportJob):
        """Record the assignment and start the AGV's trip."""
        now = self.env.now
        speed = self.p.agv_speed_mps
        job.assigned = now
        job.agv_id = agv.agv_id
        job.empty_dist_m = abs(agv.position_m - STATIONS[job.pickup])
        job.loaded_dist_m = abs(STATIONS[job.dest] - STATIONS[job.pickup])

        t_empty = travel_time_min(job.empty_dist_m, speed)
        t_loaded = travel_time_min(job.loaded_dist_m, speed)

        # Planned timeline of this job (deterministic once assigned).
        job.t_at_pickup = now + t_empty
        job.t_loaded_start = job.t_at_pickup + self.p.pickup_min
        job.t_at_dest = job.t_loaded_start + t_loaded
        job.t_done = job.t_at_dest + self.p.dropoff_min

        self.env.process(self._agv_trip(agv, job, t_empty, t_loaded))

    def _agv_trip(self, agv: AGV, job: TransportJob, t_empty: float, t_loaded: float):
        # 1) drive empty to the pickup station
        yield self.env.timeout(t_empty)
        agv.position_m = STATIONS[job.pickup]

        # 2) pick up - the AGV must be empty (one component at a time)
        if agv.load is not None:
            raise SimulationError(f"AGV {agv.agv_id} already carries component {agv.load}")
        agv.load = job.component_id
        yield self.env.timeout(self.p.pickup_min)

        # 3) drive loaded to the destination
        yield self.env.timeout(t_loaded)
        agv.position_m = STATIONS[job.dest]

        # 4) drop off; the AGV then waits at this station until dispatched again
        yield self.env.timeout(self.p.dropoff_min)
        agv.load = None
        job.finished = self.env.now
        agv.idle_since = self.env.now
        self.idle_agvs.append(agv)
        job.done.succeed()        # wake up the waiting component
        self.dispatch()           # this AGV may take a waiting job immediately


# ---------------------------------------------------------------------------
# 4. Metrics (measurement window only)
# ---------------------------------------------------------------------------

def compute_metrics(f: Factory) -> dict:
    """Compute the KPIs for one replication.

    Measurement window W = [warm-up, end) = [60, 540) min, length 480 min.
    Boundary rules:
      * throughput      - a component counts if it reached FG inside W.
      * waiting time    - jobs CREATED inside W and ASSIGNED before the end.
                          Jobs still unassigned at the end are right-censored;
                          they are counted separately as 'backlog'.
      * distances       - only the part of each driving interval inside W
                          (constant speed, so distance = speed x time-in-W).
      * utilisation     - only the part of each busy interval inside W.
    """
    p = f.p
    lo, hi = p.warmup_min, p.sim_time_min
    window_min = hi - lo
    window_h = window_min / 60.0
    v = p.speed_m_per_min
    n_agvs = len(f.agvs)

    # --- throughput -------------------------------------------------------
    completed_in_window = [c for c in f.components.values()
                           if c.completed is not None and lo <= c.completed < hi]
    throughput = len(completed_in_window) / window_h

    # --- transport waiting time ------------------------------------------
    created_in_window = [j for j in f.jobs if lo <= j.created < hi]
    assigned = [j for j in created_in_window if j.assigned is not None]
    waits = [j.assigned - j.created for j in assigned]
    mean_wait = float(np.mean(waits)) if waits else math.nan
    # Lower bound that also includes censored jobs (their wait is at least hi - created)
    waits_lb = [(j.assigned if j.assigned is not None else hi) - j.created
                for j in created_in_window]
    mean_wait_lb = float(np.mean(waits_lb)) if waits_lb else math.nan

    leg_waits = {}
    for leg in (1, 2, 3):
        w = [j.assigned - j.created for j in assigned if j.leg == leg]
        leg_waits[leg] = float(np.mean(w)) if w else math.nan

    # --- distances and AGV utilisation ----------------------------------
    started = [j for j in f.jobs if j.assigned is not None]
    empty_m = sum(v * overlap(j.assigned, j.t_at_pickup, lo, hi) for j in started)
    loaded_m = sum(v * overlap(j.t_loaded_start, j.t_at_dest, lo, hi) for j in started)
    busy_min = sum(overlap(j.assigned, j.t_done, lo, hi) for j in started)
    utilisation = busy_min / (n_agvs * window_min)

    # --- diagnostics (help explain the results) -------------------------
    m_util = {name: sum(overlap(s, e, lo, hi) for s, e, _ in log) / window_min
              for name, log in f.machine_log.items()}
    n_arrived = len(f.components)
    n_completed_total = sum(1 for c in f.components.values() if c.completed is not None)

    return {
        "throughput_per_h": throughput,
        "completed_in_window": len(completed_in_window),
        "mean_transport_wait_min": mean_wait,
        "mean_transport_wait_lower_bound_min": mean_wait_lb,
        "wait_leg1_RM_M1_min": leg_waits[1],
        "wait_leg2_M1_M2_min": leg_waits[2],
        "wait_leg3_M2_FG_min": leg_waits[3],
        "jobs_created_in_window": len(created_in_window),
        "jobs_unassigned_at_end": len(f.waiting_jobs),
        "empty_travel_km": empty_m / 1000.0,
        "loaded_travel_km": loaded_m / 1000.0,
        # empty metres driven per loaded metre (lower = less wasted driving)
        "empty_to_loaded_ratio": (empty_m / loaded_m) if loaded_m > 0 else math.nan,
        "agv_utilisation": utilisation,
        "m1_utilisation": m_util["M1"],
        "m2_utilisation": m_util["M2"],
        "components_arrived": n_arrived,
        "wip_at_end": n_arrived - n_completed_total,
        "m1_queue_at_end": len(f.machines["M1"].queue),
        "m2_queue_at_end": len(f.machines["M2"].queue),
    }


# ---------------------------------------------------------------------------
# 5. Invariant checks (run after every replication)
# ---------------------------------------------------------------------------

def check_invariants(f: Factory, metrics: dict | None = None) -> list[str]:
    """Return a list of problems found. An empty list means all checks passed."""
    problems: list[str] = []
    p = f.p

    # (a) travel/handling times are non-negative and in the right order;
    #     planned and observed finish times agree.
    for j in f.jobs:
        if j.assigned is None:
            continue
        if j.assigned < j.created - EPS:
            problems.append(f"job {j.job_id} assigned before it was created")
        if not (j.assigned <= j.t_at_pickup <= j.t_loaded_start <= j.t_at_dest <= j.t_done):
            problems.append(f"job {j.job_id} has a negative or out-of-order duration")
        if j.empty_dist_m < 0 or j.loaded_dist_m < 0:
            problems.append(f"job {j.job_id} has a negative distance")
        if j.finished is not None and abs(j.finished - j.t_done) > 1e-6:
            problems.append(f"job {j.job_id}: observed finish differs from planned finish")
        expected_loaded = abs(STATIONS[j.dest] - STATIONS[j.pickup])
        if abs(j.loaded_dist_m - expected_loaded) > EPS:
            problems.append(f"job {j.job_id}: wrong loaded distance")

    # (b) each component follows the route, in order, with non-decreasing times
    for c in f.components.values():
        events = [(e, s) for _, e, s in c.history]
        if events != EXPECTED_EVENTS[:len(events)]:
            problems.append(f"component {c.component_id} broke the route: {events}")
        times = [t for t, _, _ in c.history]
        if any(t2 < t1 - EPS for t1, t2 in zip(times, times[1:])):
            problems.append(f"component {c.component_id} has time going backwards")
        if c.completed is not None and events != EXPECTED_EVENTS:
            problems.append(f"component {c.component_id} completed without the full route")
    legs_by_comp: dict[int, list] = {}
    for j in f.jobs:
        legs_by_comp.setdefault(j.component_id, []).append((j.leg, j.pickup, j.dest))
    expected_legs = [(1, "RM", "M1"), (2, "M1", "M2"), (3, "M2", "FG")]
    for cid, legs in legs_by_comp.items():
        if legs != expected_legs[:len(legs)]:
            problems.append(f"component {cid} has wrong transport legs {legs}")

    # (c) each AGV does one job at a time (busy intervals never overlap)
    for agv in f.agvs:
        mine = sorted((j for j in f.jobs if j.agv_id == agv.agv_id), key=lambda j: j.assigned)
        for a, b in zip(mine, mine[1:]):
            if b.assigned < a.t_done - EPS:
                problems.append(f"AGV {agv.agv_id} took job {b.job_id} before finishing {a.job_id}")

    # (d) each machine processes one component at a time
    for name, log in f.machine_log.items():
        ordered = sorted(log)
        for (s1, e1, _), (s2, _, _) in zip(ordered, ordered[1:]):
            if s2 < e1 - EPS:
                problems.append(f"{name} processed two components at once at t={s2:.3f}")

    # (e) completions match finished final-leg jobs
    leg3_done = sum(1 for j in f.jobs if j.leg == 3 and j.finished is not None)
    n_completed = sum(1 for c in f.components.values() if c.completed is not None)
    if leg3_done != n_completed:
        problems.append("completed components do not match finished FG deliveries")

    # (f) metric ranges
    if metrics is not None:
        for key in ("agv_utilisation", "m1_utilisation", "m2_utilisation"):
            if not (-EPS <= metrics[key] <= 1 + EPS):
                problems.append(f"{key} = {metrics[key]} is outside [0, 1]")
        # Machine 2 can finish at most floor(window / 9) + 1 parts inside the window.
        max_completions = math.floor(p.measure_min / p.m2_process_min) + 1
        if metrics["completed_in_window"] > max_completions:
            problems.append("more completions than Machine 2 can physically produce")
        for key in ("empty_travel_km", "loaded_travel_km"):
            if metrics[key] < 0:
                problems.append(f"{key} is negative")
        # Distance cannot exceed (fleet size x speed x window).
        max_km = len(f.agvs) * p.speed_m_per_min * p.measure_min / 1000.0
        if metrics["empty_travel_km"] + metrics["loaded_travel_km"] > max_km + 1e-6:
            problems.append("AGVs travelled further than physically possible")

    return problems


# ---------------------------------------------------------------------------
# 6. Run one replication
# ---------------------------------------------------------------------------

def seed_for(params: Params, rep_index: int) -> int:
    """Replication r uses seed base_seed + r for its arrival stream.

    The same seed is used for every policy and fleet size, so all cases
    with the same r see exactly the same component arrivals
    (common random numbers).
    """
    return params.base_seed + rep_index


def run_replication(params: Params, n_agvs: int, policy: str, rep_index: int,
                    check: bool = True):
    """Run one simulation and return (metrics dict, Factory object)."""
    seed = seed_for(params, rep_index)
    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    factory = Factory(env, params, n_agvs, policy, rng)
    env.process(factory.arrivals())
    env.run(until=params.sim_time_min)

    metrics = compute_metrics(factory)
    if check:
        problems = check_invariants(factory, metrics)
        if problems:
            raise SimulationError(
                f"{policy}, {n_agvs} AGVs, rep {rep_index}: " + "; ".join(problems[:5]))

    row = {"policy": policy, "n_agvs": n_agvs, "replication": rep_index, "seed": seed}
    row.update(metrics)
    return row, factory


def analytic_capacity(params: Params) -> dict:
    """Simple hand-calculation checks used to sanity-check the simulation."""
    arrival_rate = 60.0 / params.mean_interarrival_min          # parts per hour
    m1_cap = 60.0 / params.m1_process_min
    m2_cap = 60.0 / params.m2_process_min
    leg_min = travel_time_min(100.0, params.agv_speed_mps) + params.pickup_min + params.dropoff_min
    return {
        "arrival_rate_per_h": arrival_rate,
        "m1_capacity_per_h": m1_cap,
        "m2_capacity_per_h": m2_cap,
        "bottleneck": "M2" if m2_cap < m1_cap else "M1",
        "bottleneck_capacity_per_h": min(m1_cap, m2_cap),
        "m2_load_ratio": arrival_rate / m2_cap,
        "loaded_leg_time_min": leg_min,
    }
