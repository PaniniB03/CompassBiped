import os
import math
import numpy as np
import mujoco as mj

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Config
# -------------------------
XML_PATH = "bipedCompass3.xml"
GAMMA = 0.01
DT = 5e-5
NSTEPS = 8
TMAX_PER_STEP = 30.0

# Post-impact fixed point (dimensionless)
THETA0_FP  = 0.20734774
THETAD0_FP = -0.20613085

# Event detection thresholds
EPS_LIFTOFF    = 1e-4     # (bigger than 1e-6 helps avoid chatter)
XSW_MIN        = 1e-4
PHI_MIN        = 1e-3
PHI_ROOT_TOL   = 8e-2     # tolerance on |wrap(phi - 2 theta)|
VZ_MIN_DOWN    = 0.0      # require downward crossing (vz_est < -VZ_MIN_DOWN)

# Forward convention:
# accept if FORWARD_SIGN * xsw_local > XSW_MIN
FORWARD_SIGN = 1.0

USE_DIMENSIONLESS_GRAVITY = True

PRINT_REJECTED = True
MAX_REJECTED_PRINTS = 60


# -------------------------
# Angle wrapping
# -------------------------
def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def wrap_diff(a: float) -> float:
    return wrap_to_pi(a)


# -------------------------
# Analytic impact map (beta=0)
# -------------------------
def impact_map_from_theta(theta_m: float, thetad_m: float):
    c = math.cos(2.0 * theta_m)
    theta_p  = -theta_m
    thetad_p = c * thetad_m
    phi_p    = -2.0 * theta_m
    phid_p   = c * (1.0 - c) * thetad_m
    return theta_p, thetad_p, phi_p, phid_p

def full_state_from_reduced(theta0: float, thetad0: float):
    phi0  = 2.0 * theta0
    phid0 = (1.0 - math.cos(2.0 * theta0)) * thetad0
    return theta0, thetad0, phi0, phid0


# -------------------------
# MuJoCo helpers
# -------------------------
def get_qpos_adr(model, joint_name: str) -> int:
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"Joint '{joint_name}' not found.")
    return int(model.jnt_qposadr[jid])

def get_dof_adr(model, joint_name: str) -> int:
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"Joint '{joint_name}' not found.")
    return int(model.jnt_dofadr[jid])

def try_get_joint_addrs(model, joint_name: str):
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None, None
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])

def get_site_id(model, site_name: str) -> int:
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise RuntimeError(f"Site '{site_name}' not found.")
    return int(sid)

def site_vel_world(model, data, site_id: int):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp @ data.qvel


# -------------------------
# Load model
# -------------------------
model = mj.MjModel.from_xml_path(XML_PATH)
data = mj.MjData(model)

model.opt.timestep = DT

if USE_DIMENSIONLESS_GRAVITY:
    model.opt.gravity[:] = np.array([math.sin(GAMMA), 0.0, -math.cos(GAMMA)], dtype=float)

# joints
THETA_QADR = get_qpos_adr(model, "theta")
PHI_QADR   = get_qpos_adr(model, "phi")
THETA_VADR = get_dof_adr(model, "theta")
PHI_VADR   = get_dof_adr(model, "phi")

BASEX_QADR, BASEX_VADR = try_get_joint_addrs(model, "base_x")
HAS_BASEX = (BASEX_QADR is not None)

SWING_SITE = get_site_id(model, "swing_foot_site")

# Treat base_x as a *coordinate shift* (kinematic), not a physical DOF:
FREEZE_BASEX_DURING_SWING = True


# -------------------------
# Init at post-impact FP
# -------------------------
theta0, thetad0, phi0, phid0 = full_state_from_reduced(THETA0_FP, THETAD0_FP)

data.qpos[THETA_QADR] = theta0
data.qpos[PHI_QADR]   = phi0
data.qvel[THETA_VADR] = thetad0
data.qvel[PHI_VADR]   = phid0

base_x_world = 0.0
if HAS_BASEX:
    data.qpos[BASEX_QADR] = base_x_world
    data.qvel[BASEX_VADR] = 0.0

mj.mj_forward(model, data)

z0  = float(data.site_xpos[SWING_SITE][2])
vz0 = float(site_vel_world(model, data, SWING_SITE)[2])
print(f"[init] swing_site_z={z0:+.6e}, swing_site_vz={vz0:+.6e} (want z~0 and vz>0)")


# -------------------------
# Logging
# -------------------------
t_log = []
theta_log = []
phi_log = []

z_site_log = []
x_world_log = []
x_local_log = []

z_pred_log = []
x_pred_log = []

root_err_log = []

cross_t = []
cross_kind = []  # "accept" / "reject"


def current_base_x():
    return base_x_world if HAS_BASEX else 0.0


def save_plots():
    if len(t_log) == 0:
        print("[plot] no samples logged.")
        return

    # 1) z agreement
    plt.figure()
    plt.title("Swing-foot height: MuJoCo site_z vs kinematic prediction")
    plt.plot(t_log, z_site_log, label="site_z")
    plt.plot(t_log, z_pred_log, "--", label="z_pred = cos(theta)-cos(theta-phi)")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("time")
    plt.ylabel("z")
    plt.grid(True)
    plt.legend()
    plt.savefig("debug_z_site_vs_pred.png", dpi=200)
    plt.close()

    # 2) z error
    errz = np.array(z_site_log) - np.array(z_pred_log)
    plt.figure()
    plt.title("Kinematic error: site_z - z_pred")
    plt.plot(t_log, errz)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("time")
    plt.ylabel("error")
    plt.grid(True)
    plt.savefig("debug_z_error.png", dpi=200)
    plt.close()

    # 3) x agreement (LOCAL!)
    plt.figure()
    plt.title("Swing-foot x (LOCAL): (site_x - base_x) vs kinematic prediction")
    plt.plot(t_log, x_local_log, label="x_local = site_x - base_x")
    plt.plot(t_log, x_pred_log, "--", label="x_pred = sin(theta)-sin(theta-phi)")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("time")
    plt.ylabel("x_local")
    plt.grid(True)
    plt.legend()
    plt.savefig("debug_x_local_vs_pred.png", dpi=200)
    plt.close()

    # 4) root err and z
    plt.figure()
    plt.title("Heelstrike diagnostics: |wrap(phi-2theta)| and z_site")
    plt.plot(t_log, root_err_log, label="|wrap(phi-2theta)|")
    plt.plot(t_log, z_site_log, "--", label="z_site")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("time")
    plt.grid(True)
    plt.legend()
    plt.savefig("debug_rooterr_and_z.png", dpi=200)
    plt.close()

    # 5) crossing scatter
    if len(cross_t) > 0:
        plt.figure()
        plt.title("Ground-crossing candidates (accept vs reject)")
        y = [1 if k == "accept" else 0 for k in cross_kind]
        plt.scatter(cross_t, y)
        plt.yticks([0, 1], ["reject", "accept"])
        plt.xlabel("time")
        plt.grid(True)
        plt.savefig("debug_crossings_accept_reject.png", dpi=200)
        plt.close()

    max_z_err = float(np.max(np.abs(errz)))
    print("\n[DEBUG SUMMARY]")
    print(f"  logs = {len(t_log)} samples")
    print(f"  max |site_z - z_pred| = {max_z_err:.3e}")
    print("  Saved: debug_z_site_vs_pred.png, debug_z_error.png, debug_x_local_vs_pred.png, debug_rooterr_and_z.png, debug_crossings_accept_reject.png")


# -------------------------
# Main loop
# -------------------------
overall_min_z = float("inf")
rejected_prints = 0

try:
    for step in range(NSTEPS):
        step_min_z = float("inf")
        armed = False
        hit = False

        # prev sample
        qpos_prev = data.qpos.copy()
        qvel_prev = data.qvel.copy()
        z_prev = float(data.site_xpos[SWING_SITE][2])
        xw_prev = float(data.site_xpos[SWING_SITE][0])
        t_prev = float(data.time)

        nmax = int(TMAX_PER_STEP / DT)
        for k in range(nmax):
            # save previous
            qpos_prev = data.qpos.copy()
            qvel_prev = data.qvel.copy()
            z_prev = float(data.site_xpos[SWING_SITE][2])
            xw_prev = float(data.site_xpos[SWING_SITE][0])
            t_prev = float(data.time)

            # step dynamics
            mj.mj_step(model, data)

            # wrap theta/phi to avoid huge angles ruining trig precision
            theta_raw = float(data.qpos[THETA_QADR])
            phi_raw   = float(data.qpos[PHI_QADR])
            theta_w = wrap_to_pi(theta_raw)
            phi_w   = wrap_to_pi(phi_raw)
            if (theta_w != theta_raw) or (phi_w != phi_raw):
                data.qpos[THETA_QADR] = theta_w
                data.qpos[PHI_QADR]   = phi_w

            # freeze base_x during swing (treat it as coordinate frame)
            if HAS_BASEX and FREEZE_BASEX_DURING_SWING:
                data.qpos[BASEX_QADR] = base_x_world
                data.qvel[BASEX_VADR] = 0.0

            mj.mj_forward(model, data)

            # state
            t = float(data.time)
            theta = float(data.qpos[THETA_QADR])
            phi   = float(data.qpos[PHI_QADR])

            # site world
            z_site = float(data.site_xpos[SWING_SITE][2])
            x_world = float(data.site_xpos[SWING_SITE][0])

            # local x (relative to stance foot frame)
            bx = current_base_x()
            x_local = x_world - bx

            # predicted local kinematics from angles (unit legs)
            z_pred = math.cos(theta) - math.cos(theta - phi)
            x_pred = math.sin(theta) - math.sin(theta - phi)

            root_err = abs(wrap_diff(phi - 2.0 * theta))

            # log
            t_log.append(t)
            theta_log.append(theta)
            phi_log.append(phi)
            z_site_log.append(z_site)
            x_world_log.append(x_world)
            x_local_log.append(x_local)
            z_pred_log.append(z_pred)
            x_pred_log.append(x_pred)
            root_err_log.append(root_err)

            # crossing bookkeeping
            z_cur = z_site
            xw_cur = x_world
            t_cur = t

            step_min_z = min(step_min_z, z_cur)
            overall_min_z = min(overall_min_z, z_cur)

            if not armed and (z_cur > EPS_LIFTOFF):
                armed = True

            crossed_down = armed and (z_prev > 0.0) and (z_cur <= 0.0)
            if not crossed_down:
                continue

            # interpolate event time
            alpha = z_prev / (z_prev - z_cur + 1e-12)
            alpha = min(1.0, max(0.0, alpha))

            theta_m  = (1 - alpha) * float(qpos_prev[THETA_QADR]) + alpha * float(data.qpos[THETA_QADR])
            phi_m    = (1 - alpha) * float(qpos_prev[PHI_QADR])   + alpha * float(data.qpos[PHI_QADR])
            thetad_m = (1 - alpha) * float(qvel_prev[THETA_VADR]) + alpha * float(data.qvel[THETA_VADR])
            t_evt    = (1 - alpha) * t_prev + alpha * t_cur

            # local x at event
            x_local_prev = xw_prev - bx
            x_local_cur  = xw_cur - bx
            xsw_local_evt = (1 - alpha) * x_local_prev + alpha * x_local_cur

            # diagnostics
            theta_m = wrap_to_pi(theta_m)
            phi_m   = wrap_to_pi(phi_m)

            vz_est = (z_cur - z_prev) / DT
            root_err_evt = abs(wrap_diff(phi_m - 2.0 * theta_m))
            xsw_fwd = FORWARD_SIGN * xsw_local_evt

            accept = (
                (xsw_fwd > XSW_MIN) and
                (abs(phi_m) > PHI_MIN) and
                (root_err_evt < PHI_ROOT_TOL) and
                (vz_est < -VZ_MIN_DOWN)
            )

            cross_t.append(t_evt)
            cross_kind.append("accept" if accept else "reject")

            if not accept:
                if PRINT_REJECTED and (rejected_prints < MAX_REJECTED_PRINTS):
                    print(
                        f"[rejected] step={step} t={t_evt:.6f}  "
                        f"xsw_local={xsw_local_evt:+.3e} xsw_fwd={xsw_fwd:+.3e}  "
                        f"theta={theta_m:+.6f} phi={phi_m:+.6f}  "
                        f"|phi-2theta|={root_err_evt:.3e}  vz_est={vz_est:+.3e}"
                    )
                    rejected_prints += 1
                continue

            # Apply analytic impact
            theta_p, thetad_p, phi_p, phid_p = impact_map_from_theta(theta_m, thetad_m)

            # Coordinate shift: new stance foot becomes the old swing foot
            if HAS_BASEX:
                base_x_world += xsw_local_evt  # (forward sign already handled in accept test)

            data.time = t_evt
            if HAS_BASEX:
                data.qpos[BASEX_QADR] = base_x_world
                data.qvel[BASEX_VADR] = 0.0

            data.qpos[THETA_QADR] = theta_p
            data.qpos[PHI_QADR]   = phi_p
            data.qvel[THETA_VADR] = thetad_p
            data.qvel[PHI_VADR]   = phid_p

            mj.mj_forward(model, data)

            hit = True
            print(
                f"[heelstrike] step={step} t={t_evt:.6f}  "
                f"xsw_local={xsw_local_evt:+.6f} xsw_fwd={xsw_fwd:+.6f}  "
                f"theta-={theta_m:+.6f} phi-={phi_m:+.6f} root_err={root_err_evt:.3e}"
            )
            break

        if not hit:
            raise RuntimeError(
                f"No heelstrike in step {step}. min(site_z) this step = {step_min_z:.3e} "
                f"(overall min so far {overall_min_z:.3e})."
            )

    print("Done.")

finally:
    save_plots()
