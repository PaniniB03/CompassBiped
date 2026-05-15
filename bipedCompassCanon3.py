import math
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw
import matplotlib.pyplot as plt
import time

'''
Time is dimensionless: τ = t * sqrt(g/ℓ). (Can treat it as “seconds” if only care about shapes.)
Model here is the classic point-foot simplest walker (not the curved-foot Wisse/Schwab)
bipedCompassCanon.py can find the necessary alternate ICs based on gamma
'''
L = 1.0      # meters (MJCF leg length)
g = 9.81     # m/s^2
# ---- time conversion between dimensionless tau and seconds ----
TAU_TO_SEC = math.sqrt(L / g)      # t [s] = tau * sqrt(L/g)
SEC_TO_TAU = 1.0 / TAU_TO_SEC      # tau = t[s] / sqrt(L/g)

XML_PATH = "bipedCompass1.xml"

GAMMA = 0.01 #slope
NSTEPS = 15
DT = 5e-5 #integration step (dimensionless time)
# DT_sec = DT * math.sqrt(L/g)
TMAX_PER_STEP = 30.0

# Fixed point for GAMMA=0.01 (simplest walking model)
# solve for a fixed point (theta, thetadot) for a given GAMMA
# Uncomment if want to change GAMMA and auto-find ICs.
"""
from scipy.optimize import root

def poincare_map(reduced, gamma):
    theta0, thetad0 = reduced
    x0 = full_state_from_reduced(theta0, thetad0)
    _, _, _, x_plus, _ = simulate_one_step(x0, gamma)
    return np.array([x_plus[0], x_plus[1]])

def find_fixed_point(gamma, guess=(0.2, -0.2)):
    def F(v):
        v_next = poincare_map(v, gamma)
        return v_next - v
    sol = root(F, np.array(guess, dtype=float), tol=1e-12)
    if not sol.success:
        raise RuntimeError(f"root find failed: {sol.message}")
    return sol.x[0], sol.x[1]
"""
# THETA0_FP, THETAD0_FP = find_fixed_point(GAMMA, guess=(0.2, -0.2))
THETA0_FP  = 0.20734774 
THETAD0_FP = -0.20613085




# Toggle: B cross-check (MuJoCo swing_foot_site world z vs analytic z_swing)
DO_B_CHECK = True

# Option 2 toggle: translate-forward (world x offset via base_x joint)
DO_TRANSLATE_FORWARD = True
FORWARD_SIGN = 1.0 # if step_length prints negative, keep -1.0 so forward motion is +x; else set +1.0

# Minimal rendering
RENDER = True
FPS = 60.0


# -------------------- KEYFRAME CAPTURE --------------------
CAPTURE_KEYFRAMES = True
N_KEYFRAMES = 4
KEYFRAME_START = 0.05 #can skip initial transient
KEYFRAME_END = 4.0

keyframe_times = np.linspace(KEYFRAME_START, KEYFRAME_END, N_KEYFRAMES)
keyframe_imgs = []
keyframe_ts = []
_next_k = 0



DO_REJECTED_CROSSING_PRINT = True
PRINT_MAX_REJECTED = 20
DO_ACCEPTED_CROSSING_PRINT = True
PRINT_MAX_ACCEPTED = 20
accepted_events = []   # will store dicts: tau, theta, phi, phi-2theta, xsw, zsw, step_idx, kind
rejected_events = []   # store rejected too (optional, but useful for LaTeX table)

plt.rcParams.update({
    'axes.titlesize': 24,     # Title size
    'axes.titleweight':  "bold",
    'axes.labelsize': 22,     # X and Y label size
    'xtick.labelsize': 22,    # X-axis tick size
    'ytick.labelsize': 22,    # Y-axis tick size
    'legend.fontsize': 20,    # Legend font size
    'legend.loc':'upper right',
    'legend.facecolor':'white',
    'legend.framealpha': 1
})


# -------------------------
# Analytic dynamics (dimensionless)
# x = [theta, thetad, phi, phid]
# theta: stance-leg angle wrt slope normal
# phi: inter-leg angle
# -------------------------
def f(x, gamma): #stance dynamics rewritten
    theta, thetad, phi, phid = x #state
    thetadd = math.sin(theta - gamma) #theta double dot
    # From simplest walking model:   thetadd - phidd + thetad^2 sin(phi) - cos(theta-gamma) sin(phi) = 0
    phidd = thetadd + thetad * thetad * math.sin(phi) - math.cos(theta - gamma) * math.sin(phi) #phi double dot
    return np.array([thetad, thetadd, phid, phidd], dtype=float)

#forward integration of the 2 ODEs
def rk4_step(x, dt, gamma):
    k1 = f(x, gamma)
    k2 = f(x + 0.5 * dt * k1, gamma)
    k3 = f(x + 0.5 * dt * k2, gamma)
    k4 = f(x + dt * k3, gamma)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# Kinematics in slope-normal frame (ground is z=0)
# stance foot at (0,0) within each step
# hip: (sin(theta), cos(theta))
# swing foot: hip - (sin(theta-phi), cos(theta-phi))
def swing_foot_height_analytic(x):
    theta, _, phi, _ = x
    return math.cos(theta) - math.cos(theta - phi)

# Impact map (plastic collision + leg relabel) for simplest model (beta=0)
# As in the simplest-walker derivation:
# theta+  = -theta-
# thetad+ = cos(2theta-) * thetad-
# phi+    = -2theta-
# phid+   = cos(2theta-) * (1 - cos(2theta-)) * thetad-
def impact_map(x_minus):
    theta, thetad, _, _ = x_minus
    c = math.cos(2.0 * theta)
    return np.array([-theta, c * thetad, -2.0 * theta, c * (1.0 - c) * thetad], dtype=float)


# On the Poincaré section parameterization right after impact. The state is fully determined by (theta, thetadot):
# phi = 2theta, phidot = (1 - cos(2theta)) * thetadot
def full_state_from_reduced(theta0, thetad0):
    phi0 = 2.0 * theta0
    phid0 = (1.0 - math.cos(2.0 * theta0)) * thetad0
    return np.array([theta0, thetad0, phi0, phid0], dtype=float)


"""Swing-foot x position (local stance frame) at heelstrike."""
def step_length_local(x_minus):
    theta_m = x_minus[0]
    phi_m   = x_minus[2]
    return math.sin(theta_m) - math.sin(theta_m - phi_m)

# -------------------------
# MuJoCo helpers
# -------------------------
def get_qpos_adr(model, joint_name: str) -> int:         #-------get joints
    jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"Joint '{joint_name}' not found.")
    return int(model.jnt_qposadr[jid])

# def try_get_qpos_adr(model, joint_name: str):
#     jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_name)
#     if jid < 0:
#         return None
#     return int(model.jnt_qposadr[jid])

def get_site_id(model, site_name: str) -> int:      #-------get sites
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise RuntimeError(f"Site '{site_name}' not found.")
    return int(sid)

def get_geom_id(model, geom_name: str) -> int:      #-------get geoms
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise RuntimeError(f"Geom '{geom_name}' not found.")
    return int(gid)

def set_state_in_mujoco(model, data, theta, phi, base_x, t, BASEX_QADR, THETA_QADR, PHI_QADR):  #-------set states qpos
    # Only need qpos for kinematic visualization
    if BASEX_QADR is not None:
        data.qpos[BASEX_QADR] = base_x
    data.qpos[THETA_QADR] = theta
    data.qpos[PHI_QADR]   = phi
    data.time = t
    mj.mj_forward(model, data)


# -------------------------
# Load model + window
# -------------------------
model = mj.MjModel.from_xml_path(XML_PATH)
data = mj.MjData(model)

# required joints
THETA_QADR = get_qpos_adr(model, "theta")
PHI_QADR   = get_qpos_adr(model, "phi")

# only for translate-forward
# BASEX_QADR = None
# if DO_TRANSLATE_FORWARD:
#     BASEX_QADR = try_get_qpos_adr(model, "base_x")
#     if BASEX_QADR is None:
#         raise RuntimeError("DO_TRANSLATE_FORWARD=True but joint 'base_x' not found.")
BASEX_QADR = get_qpos_adr(model, "base_x")
if DO_TRANSLATE_FORWARD and (BASEX_QADR is None):
    raise RuntimeError("DO_TRANSLATE_FORWARD=True but joint 'base_x' not found in XML. Use the translate-forward MJCF.")

# get sites and geoms for tracking
SWING_SITE = get_site_id(model, "swing_foot_site")
HIP_SITE    = get_site_id(model, "hip_site") #not needed
SWING_FOOT_GEOM = get_geom_id(model, "swing_foot_geom")
TRUE_FOOT_RADIUS = float(model.geom_size[SWING_FOOT_GEOM][0])

# local-frame offset sanity check (both are in hip body frame in XML)
# (This is optional; just helps debug MJCF consistency.)
site_local = np.array(model.site_pos[SWING_SITE])
geom_local = np.array(model.geom_pos[SWING_FOOT_GEOM])
nominal_site_center_dist = float(np.linalg.norm(site_local - geom_local))

print(f"[model check] swing site–center nominal dist = {nominal_site_center_dist:.6f} (expected ~{TRUE_FOOT_RADIUS:.6f})")

# -------------------------
# GLFW setup
# -------------------------
if RENDER:
    if not glfw.init():
        raise RuntimeError("GLFW init failed.")
    window = glfw.create_window(1200, 900, "Hybrid simplest-walker: MuJoCo Viz", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mj.MjvCamera()
    opt = mj.MjvOption()
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)
    # time.sleep(1)
    
    # THIS IS FOR FOOT VIEW CAMERA
    # cam.distance = 2.0
    # cam.lookat = np.array([0.0, 0.0, 0.1])
    # cam.elevation = 0
    # cam.azimuth = 90
    cam.distance = 1.5
    cam.lookat = np.array([0.0, 0.0, 0.5])
    cam.elevation = -10
    cam.azimuth = 100

# -------------------------
# Hybrid simulation (A) with cross-check (B) + translate-forward
# -------------------------
x = full_state_from_reduced(THETA0_FP, THETAD0_FP)

# global logs
T = []
T_new = []
theta_log, phi_log = [], []
zsw_log = []
xhip_log, zhip_log = [], []

# MuJoCo logs
site_z_log = []
geom_bottom_z_log = []
site_center_dist_err_max = 0.0
min_site_z = float("inf")
min_geom_bottom_z = float("inf")
max_abs_site_err = 0.0
max_abs_site_minus_geom_bottom = 0.0

poincare_pts = []
rejected_count = 0
crossing_log = []

t_global = 0.0
base_x_world = 0.0 # this is put into joint "base_x" when translate-forward is enabled
next_render_t = 0.0
eps_liftoff = 1e-4


"""
    Integrate until the swing foot hits the ground (z_swing -> 0 from above),
    but ignore the trivial root at t=0 by:
      - waiting until swing-foot height rises above eps (liftoff gate)
      - requiring stance leg has passed vertical: theta < 0 at impact
"""

for step in range(NSTEPS):
    enabled = False
    t_local = 0.0
    z = swing_foot_height_analytic(x)

    nmax = int(TMAX_PER_STEP / DT) #DT
    for _ in range(nmax):
        # Set MuJoCo kinematics
        bx = base_x_world if DO_TRANSLATE_FORWARD else 0.0
        set_state_in_mujoco(model, data, x[0], x[2], bx, t_global + t_local,
                            BASEX_QADR, THETA_QADR, PHI_QADR) #model, data, theta0, phi0, base_x_world, event step +dt


        # --- MuJoCo geometry checks ---
        site_pos = data.site_xpos[SWING_SITE].copy()
        geom_center = data.geom_xpos[SWING_FOOT_GEOM].copy()

        site_z = float(site_pos[2])
        geom_center_z = float(geom_center[2])
        # geom_bottom_z = geom_center_z - TRUE_FOOT_RADIUS
        R = data.geom_xmat[SWING_FOOT_GEOM].reshape(3, 3)   # local->world rotation
        geom_point_local_negz_z = geom_center_z - TRUE_FOOT_RADIUS * R[2, 2]
        geom_bottom_z = geom_point_local_negz_z


        site_z_log.append(site_z)
        geom_bottom_z_log.append(geom_bottom_z)

        min_site_z = min(min_site_z, site_z)
        min_geom_bottom_z = min(min_geom_bottom_z, geom_bottom_z)

        max_abs_site_minus_geom_bottom = max(max_abs_site_minus_geom_bottom, abs(site_z - geom_bottom_z))

        # site–center distance should stay ~radius (rigid)
        dist_sc = float(np.linalg.norm(site_pos - geom_center))
        site_center_dist_err_max = max(site_center_dist_err_max, abs(dist_sc - nominal_site_center_dist))


        


        # --- Analytic logs ---
        theta, thetad, phi, phid = x
        T.append(t_global + t_local)
        # T_sec = np.array(T) * math.sqrt(L/g)
        # T_new.append(T_sec)
        theta_log.append(theta)
        phi_log.append(phi)
        zsw = swing_foot_height_analytic(x)
        zsw_log.append(zsw)

        #hip position
        xhip_log.append(base_x_world + math.sin(theta))
        zhip_log.append(math.cos(theta))

        # B cross-check: analytic point-foot swing height vs MuJoCo site world z - should match
        if DO_B_CHECK:
            err = abs(site_z - zsw)
            max_abs_site_err = max(max_abs_site_err, err)

        # Render at ~FPS
        if RENDER and (t_global + t_local) >= next_render_t:
            if DO_TRANSLATE_FORWARD:
                hip_loc = data.site_xpos[HIP_SITE].copy()
                cam.lookat[0] = hip_loc[0]
            vw, vh = glfw.get_framebuffer_size(window)
            viewport = mj.MjrRect(0, 0, vw, vh)
            mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
            mj.mjr_render(viewport, scene, context)
            

            if CAPTURE_KEYFRAMES and _next_k < len(keyframe_times) and data.time >= keyframe_times[_next_k]:
                rgb = np.empty((vh, vw, 3), dtype=np.uint8)
                depth = np.empty((vh, vw), dtype=np.float32)
                mj.mjr_readPixels(rgb, depth, viewport, context) # reads active buffer pixels :contentReference[oaicite:2]{index=2}
                rgb = np.flipud(rgb) # OpenGL origin is bottom-left
                margin = int(vw * 0.30) 
                rgb = rgb[:, margin : vw - margin, :]
                keyframe_imgs.append(rgb.copy())
                keyframe_ts.append(data.time)
                _next_k += 1

            glfw.swap_buffers(window)
            glfw.poll_events()
            if glfw.window_should_close(window):
                break
            next_render_t += 1.0 / FPS

        if RENDER and glfw.window_should_close(window):
            break

        # Integrate one step (analytic)
        x_next = rk4_step(x, DT, GAMMA) #x,DT,GAMMA
        z_next = swing_foot_height_analytic(x_next)

        # enable heelstrike detection only after liftoff
        if not enabled:
            if z_next > eps_liftoff:
                enabled = True
        else:
            # rejected crossings: z hits 0 but theta not negative (wrong phase) - for swing foot decision
            # if DO_REJECTED_CROSSING_PRINT and (rejected_count < PRINT_MAX_REJECTED):
            #     if (z > 0.0) and (z_next <= 0.0) and not (x_next[0] < 0.0):
            #         theta_tmp, _, phi_tmp, _ = x_next
            #         xsw_tmp = math.sin(theta_tmp) - math.sin(theta_tmp - phi_tmp)
            #         print(f"[rejected crossing] t={t_global+t_local:.6f}  theta={theta_tmp:.6f}  phi={phi_tmp:.6f}  x_swing={xsw_tmp:.6f}")
            #         rejected_count += 1
            # --- candidate downward crossing (only meaningful after liftoff enabled) ---
            # if enabled and (z > 0.0) and (z_next <= 0.0):

            #     theta_tmp, _, phi_tmp, _ = x_next
            #     xsw_tmp = math.sin(theta_tmp) - math.sin(theta_tmp - phi_tmp)
            #     branch_err = phi_tmp - 2.0 * theta_tmp   # ~0 means on the desired branch

            #     # phase guard (current criterion)
            #     phase_ok = (theta_tmp < 0.0)

            #     crossing_log.append({
            #         "tau": t_global + t_local,
            #         "theta": theta_tmp,
            #         "phi": phi_tmp,
            #         "xsw": xsw_tmp,
            #         "phi_minus_2theta": branch_err,
            #         "phase_ok": phase_ok,
            #         "decision": "ACCEPT" if phase_ok else "REJECT"
            #     })

            #     # optional print (only rejected)
            #     if DO_REJECTED_CROSSING_PRINT and (not phase_ok) and (rejected_count < PRINT_MAX_REJECTED):
            #         print(f"[rejected crossing] t={t_global+t_local:.6f}  theta={theta_tmp:.6f}  phi={phi_tmp:.6f}  "
            #             f"(phi-2theta)={branch_err:.3e}  x_swing={xsw_tmp:.6f}")
            #         rejected_count += 1

            if DO_REJECTED_CROSSING_PRINT and (rejected_count < PRINT_MAX_REJECTED):
                if (z > 0.0) and (z_next <= 0.0) and not (x_next[0] < 0.0):
                    theta_tmp, _, phi_tmp, _ = x_next
                    xsw_tmp = math.sin(theta_tmp) - math.sin(theta_tmp - phi_tmp)
                    rec = dict(
                        kind="rejected",
                        step=step,
                        tau=t_global + t_local,
                        theta=theta_tmp,
                        phi=phi_tmp,
                        phi_minus_2theta=(phi_tmp - 2.0*theta_tmp),
                        xsw=xsw_tmp,
                        zsw=z_next
                    )
                    rejected_events.append(rec)
                    print(f"[rejected crossing] t={rec['tau']:.6f}  theta={rec['theta']:.6f}  phi={rec['phi']:.6f}  "
                        f"(phi-2theta)={rec['phi_minus_2theta']:+.3e}  x_swing={rec['xsw']:.6f}")
                    rejected_count += 1

            # accepted heelstrike
            # xsw_next = math.sin(x_next[0]) - math.sin(x_next[0] - x_next[2]) # (xsw_next > 0.0) and (xsw_next > XSW_MIN)
            # XSW_MIN = 1e-6

            # heelstrike: z crosses 0 downward and theta < 0 at impact
            if (z > 0.0) and (z_next <= 0.0) and (x_next[0] < 0.0):
                alpha = z / (z - z_next + 1e-12)
                x_minus = x + alpha * (x_next - x)
                t_event = t_local + alpha * DT

                x_plus = impact_map(x_minus)
                # --- ACCEPTED heelstrike event logging ---
                if DO_ACCEPTED_CROSSING_PRINT and (len(accepted_events) < PRINT_MAX_ACCEPTED):
                    theta_m, _, phi_m, _ = x_minus
                    xsw_m = math.sin(theta_m) - math.sin(theta_m - phi_m)
                    zsw_m = swing_foot_height_analytic(x_minus)   # should be ~0
                    rec = dict(
                        kind="accepted",
                        step=step,
                        tau=t_global + t_event,
                        theta=theta_m,
                        phi=phi_m,
                        phi_minus_2theta=(phi_m - 2.0*theta_m),
                        xsw=xsw_m,
                        zsw=zsw_m
                    )
                    accepted_events.append(rec)
                    print(f"[ACCEPTED heelstrike] t={rec['tau']:.6f}  theta={rec['theta']:.6f}  phi={rec['phi']:.6f}  "
                        f"(phi-2theta)={rec['phi_minus_2theta']:+.3e}  x_swing={rec['xsw']:.6f}")

                # Update translate-forward base offset at impact
                dx_local = step_length_local(x_minus)
                if DO_TRANSLATE_FORWARD:
                    base_x_world += FORWARD_SIGN * dx_local

                poincare_pts.append([x_plus[0], x_plus[1]]) # Poincaré sample (just after impact)

                if step == 1:
                    idx_2steps = len(T)
                
                # Advance to next step
                t_global += t_event
                x = x_plus
                break
        
        # Continue integration
        x = x_next
        z = z_next
        t_local += DT

    else:
        raise RuntimeError(f"No heelstrike found in step {step}. Try different ICs or smaller DT.")

    if RENDER and glfw.window_should_close(window):
        break

if RENDER:
    glfw.terminate()

    if CAPTURE_KEYFRAMES and len(keyframe_imgs) > 0:
    # Define captions (must match N_KEYFRAMES)
        captions = [
            "Start of swing", 
            "Before mid-swing", 
            "Post mid-swing\npre-impact",
            "Post-impact\nlegs relabeled", 
        ]

        n_imgs = len(keyframe_imgs)
        n_rows = 2
        n_cols = 2 #(n_imgs) // n_rows 
        
        # Create the figure grid. Increase the height slightly (from 4 to 5) to make room for text underneath
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7, 10))
        axes_flat = axes.flatten() # Flatten axes array to iterate easily if it's 2D

        for i in range(len(axes_flat)):
            if i < n_imgs:
                ax = axes_flat[i]
                ax.imshow(keyframe_imgs[i])
                
                # Title (on top)
                ax.set_title(rf"$\tau$={keyframe_ts[i]:.2f}", fontsize=22, fontweight='bold', pad=5)
                
                
                # Caption (underneath)
                # transform=ax.transAxes means (0.5, -0.1) is relative to the plot box
                # 0.5 is center, -0.1 is just below the bottom edge
                if i < len(captions):
                    ax.text(0.5, -0.02, captions[i], 
                            transform=ax.transAxes, 
                            ha='center', va='top', 
                            fontsize=22, fontweight='bold',
                            wrap=True)
            
            axes_flat[i].axis("off") # Hide the axis lines/ticks for all subplots (including empty ones)

        # Use rect to prevent the tight_layout from clipping the bottom captions
        fig.tight_layout(rect=[0, 0.02, 1, 1]) 
        fig.savefig("biped_keyframes_captioned.png", dpi=200)
        plt.close(fig)
        print("Saved: biped_keyframes_captioned.png")

poincare_pts = np.array(poincare_pts)

print(f"[B check] max |site_z - analytic| = {max_abs_site_err:.3e}")
print(f"[geom check] min swing-foot site_z = {min_site_z:.6e}")
print(f"[geom check] min swing-foot sphere-bottom z = {min_geom_bottom_z:.6e}")
print(f"[geom check] max |site_z - sphere_bottom_z| = {max_abs_site_minus_geom_bottom:.6e}")
print(f"[geom check] max site-center distance drift = {site_center_dist_err_max:.3e}")

# if want time in seconds for plots
T_sec = np.array(T) * TAU_TO_SEC




# -------------------------
# Plots
# -------------------------
# plt.figure()
# plt.title("Angles vs Time")
# plt.plot(T_sec, theta_log, label=r"$\theta$")
# plt.plot(T_sec, phi_log, "--", label=r"$\phi$")
# plt.xlabel("time (s)")
# plt.ylabel("Angle (rad)")
# plt.grid(True)
# plt.legend()

plt.figure(figsize=(14.53,8.03))
plt.title("Angles vs Time (dimensionless)")
plt.plot(T[:idx_2steps], theta_log[:idx_2steps], label=r"$\theta$")
plt.plot(T[:idx_2steps], phi_log[:idx_2steps], "--", label=r"$\phi$")
plt.xlabel(r"time ($\tau$)")
plt.ylabel("Angle (rad)")
plt.grid(True)
plt.legend()
plt.savefig("Figure_1_1.png", dpi='figure')


plt.figure(figsize=(14.53,8.03))
plt.title("Swing-foot Height vs Time (dimensionless)")
plt.plot(T[:idx_2steps], zsw_log[:idx_2steps])
plt.xlabel(r"time ($\tau$)")
plt.ylabel(r"height (/$\ell$)")
plt.grid(True)
plt.savefig("SwingHeightAnalytic.png", dpi='figure')

plt.figure(figsize=(14.53,8.03))
plt.title("Swing-foot Height: Analytic vs MuJoCo")
plt.plot(T[:idx_2steps], zsw_log[:idx_2steps], label="analytic point foot")
plt.plot(T[:idx_2steps], site_z_log[:idx_2steps], "--", label="MuJoCo swing-foot site point")
# plt.plot(T[:idx_2steps], geom_bottom_z_log[:idx_2steps] "-.", label="MuJoCo swing-foot geom bottom") #commented
plt.xlabel(r"time ($\tau$)")
plt.ylabel(r"height (/$\ell$)")
plt.grid(True)
plt.legend(fontsize=15, loc='upper left', bbox_to_anchor=(-0.005, 1))
plt.savefig("SwingFootCompare1.png", dpi='figure')

#commented
# plt.figure(figsize=(14.53,8.03))
# plt.title("Finite-radius sphere bottom height (diagnostic)")
# plt.plot(T, geom_bottom_z_log, label="sphere bottom z = center_z - r")
# plt.xlabel("time (scaled)")
# plt.ylabel("height")
# plt.grid(True)
# plt.legend()
# plt.savefig("figtest.png", dpi='figure')

plt.figure(figsize=(14.53,8.03))
plt.title("Hip Cartesian Position vs Dimensionless Time")
plt.plot(T[:idx_2steps], xhip_log[:idx_2steps], label=r"Hip $x$")
plt.plot(T[:idx_2steps], zhip_log[:idx_2steps], label=r"Hip $z$")
plt.xlabel(r"time ($\tau$)")
plt.ylabel(r"position (/$\ell$)")
plt.grid(True)
plt.legend(loc='lower left')
plt.savefig("hipPos1.png", dpi='figure')

# plt.figure(figsize=(10, 6))
# theta_base = 0.2061308
# theta_base2 = 0.2073477
# plt.title("Poincaré samples (post-impact) -plot4")
# plt.plot((poincare_pts[:, 0]-theta_base2)*1e8, (poincare_pts[:, 1]+theta_base)*1e8)
# plt.xlabel(r"$(\theta^+ - 0.207) \times 10^{8}$")
# plt.ylabel(r"$(\dot{\theta}^+ + 0.207) \times 10^{8}$")
# plt.tight_layout()
# plt.grid(True)
# plt.scatter([0.0], [0.0], marker="*", s=200, label="fixed point (reference)")
# plt.legend()

import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter



# -------------------------
# NEW: Two Poincaré alternatives
# -------------------------
if len(poincare_pts) >= 2:
    theta_plus = poincare_pts[:, 0]
    thetad_plus = poincare_pts[:, 1]

    '''

    # Option 1: Return map (theta_{k+1}^+ vs theta_k^+)
    formatter = plt.gca().xaxis.get_major_formatter()
    plt.figure(figsize=(10, 6))
    theta_base = 0.2073477
    plt.title("Return map: $\\theta_{k+1}^+$ vs $\\theta_k^+$")
    # plt.plot(theta_plus[:-1], theta_plus[1:], "o-")
    plt.plot((theta_plus[:-1] - theta_base) * 1e8, 
         (theta_plus[1:] - theta_base) * 1e8, "o-")
    # plt.xlabel(r"$\theta_k^+$")
    # plt.ylabel(r"$\theta_{k+1}^+$")
    plt.xlabel(rf"$(\theta_k^+ - \theta^\star)\times 10^{{{int(np.log10(10e8))}}}$")
    plt.ylabel(rf"$(\theta_{{k+1}}^+ - \theta^\star)\times 10^{{{int(np.log10(10e8))}}}$")
    # plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useOffset=False)
    # plt.tight_layout()
    plt.grid(True)
    # After plotting u = (theta_k - theta_base)*scale, v = (theta_{k+1} - theta_base)*scale

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    lo = min(xmin, ymin)
    hi = max(xmax, ymax)
    ax.plot([lo, hi], [lo, hi], "--", linewidth=2, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend()


    plt.figure(figsize=(10, 6))
    plt.title("Poincaré samples (post-impact)-plot6")
    plt.plot(theta_plus, thetad_plus, "-o", linewidth=2, markersize=6)
    plt.scatter([THETA0_FP], [THETAD0_FP], marker="*", s=250, label="fixed point $p^*$")
    plt.xlabel(r"$\theta^+$ (rad)")
    plt.ylabel(r"$\dot{\theta}^+$ (rad/s or 1/$\tau$)")
    plt.grid(True)
    plt.legend()


    # 2. Custom Formatter for "Small Coefficient" look
    def log_formatter(x, pos):
        exponent = int(np.floor(np.log10(x)))
        coeff = x / 10**exponent
        # If it's a clean power of 10, just show 10^exp
        if coeff <= 1.1: # handling float precision
            return f"$10^{{{exponent}}}$"
        # Otherwise, show coefficient in a smaller font (\small or \scriptsize)
        return rf"$\scriptstyle {coeff:.0f} \times \textstyle 10^{{{exponent}}}$"
    
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))
    # Optional: also label minor ticks if desired
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(log_formatter))
    '''


    # Option 2: Step-index convergence to the fixed point (use known FP)
    theta_fp = THETA0_FP
    thetad_fp = THETAD0_FP
    dist = np.sqrt((theta_plus - theta_fp)**2 + (thetad_plus - thetad_fp)**2)

    plt.figure(figsize=(14.53,8.03)) #originally 13,6
    plt.title("Step-Index Error Convergence")
    ax = plt.gca()
    plt.semilogy(np.arange(1,len(dist)+1), dist, "o-")
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs=(1.0,), numticks=12))
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=12))
    ax.yaxis.set_minor_formatter(ticker.LogFormatterSciNotation())
    plt.xticks(np.arange(1, len(dist)+1, 1))
    # plt.xlim(0, NSTEPS+1)
    plt.ylim(1e-9, 1e-7)
    plt.xlabel(r"step number ($k$)")
    # plt.yticks(fontsize=16)
    plt.ylabel(r"$\|p_k - p^\star\|$ (log scale)")
    plt.tight_layout()
    plt.grid(True, which="both")
    plt.savefig("ConvergencePlot1.png", dpi='figure')


    scale = 1e8
    theta_plus  = poincare_pts[:, 0]
    thetad_plus = poincare_pts[:, 1]

    dtheta  = (theta_plus  - THETA0_FP)  * scale
    dthetad = (thetad_plus - THETAD0_FP) * scale

    plt.figure(figsize=(14.53,8.03)) #originally 10,6
    plt.title("Poincaré Samples (post-impact)")
    plt.plot(dtheta, dthetad, "-o", linewidth=2, markersize=6)
    ax = plt.gca()

    # these arrows show the progression in samples
    # for i in range(len(dtheta) - 1):
    #     ax.annotate(
    #         "", 
    #         xy=(dtheta[i+1], dthetad[i+1]), 
    #         xytext=(dtheta[i], dthetad[i]),
    #         arrowprops=dict(arrowstyle="->", linewidth=1.5),
    #     )

    # Optional: label a few points
    label_idx = [0, len(dtheta)-1]  # 1st, 2nd, 3rd, last
    for i in label_idx:
        ax.text(dtheta[i], dthetad[i], f"step {i+1}", fontsize=22, fontweight="bold")
    plt.scatter([[0.0]], [0.0], marker="*", s=250, label="fixed point $p^\star$")
    plt.xlabel(r"$(\theta^+ - \theta^\star)\times 10^{8}$")
    plt.ylabel(r"$(\dot{\theta}^+ - \dot{\theta}^\star)\times 10^{8}$")
    plt.grid(True)
    plt.legend()
    plt.savefig("PoincareSamples3.png", dpi='figure')






from matplotlib.ticker import ScalarFormatter, LogLocator, NullFormatter, LogFormatterSciNotation

def sci_no_offset(ax):
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((0, 0))   # always scientific
    fmt.set_useOffset(False)      # kill the weird "+2.073e-1" offsets
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)

theta_plus  = poincare_pts[:, 0]
thetad_plus = poincare_pts[:, 1]

# --- delta coordinates around the fixed point (thesis-clean) ---
dtheta  = theta_plus  - THETA0_FP
dthetad = thetad_plus - THETAD0_FP

# Optional scaling to put numbers ~O(1) while staying honest
SCALE = 1e8
dtheta_s  = dtheta  * SCALE
dthetad_s = dthetad * SCALE

# (A) Poincaré samples in state space (post-impact)
# plt.figure(figsize=(10, 6))
# plt.title(r"Poincaré samples (post-impact) in $(\theta^+,\theta'^+)$")
# plt.plot(dtheta_s, dthetad_s, "-o", linewidth=2, markersize=6)
# plt.scatter([0.0], [0.0], marker="*", s=250, label=r"fixed point $p^\star$")
# plt.xlabel(rf"$(\theta^+ - \theta^\star)\times 10^{{{int(np.log10(SCALE))}}}$")
# plt.ylabel(rf"$(\theta'^+ - \theta'^\star)\times 10^{{{int(np.log10(SCALE))}}}$")
# plt.grid(True)
# plt.legend()

# (B) Return map in delta-coordinates: θ_{k+1}^+ vs θ_k^+
if len(theta_plus) >= 2:
    plt.figure(figsize=(14.53,8.03)) #originally 10,6
    plt.title(r"Return map: $\theta_{k+1}^+$ vs $\theta_k^+$")
    xk  = dtheta_s[:-1]
    xk1 = dtheta_s[1:]
    plt.plot(xk, xk1, "o-", linewidth=2)
    plt.xlabel(rf"$(\theta_k^+ - \theta^\star)\times 10^{{{int(np.log10(SCALE))}}}$")
    plt.ylabel(rf"$(\theta_{{k+1}}^+ - \theta^\star)\times 10^{{{int(np.log10(SCALE))}}}$")
    ax = plt.gca()
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--", linewidth=2, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig("ReturnMap1.png", dpi='figure')






















# -------------------------
# Thesis-ready Poincaré plots (save + no clipping + consistent notation)
# -------------------------
'''
poincare_pts = np.array(poincare_pts)
if len(poincare_pts) >= 2:
    theta_plus  = poincare_pts[:, 0]
    thetad_plus = poincare_pts[:, 1]

    # Fixed point (reference)
    theta_star  = THETA0_FP
    thetad_star = THETAD0_FP
    p_star = np.array([theta_star, thetad_star])

    # Centered coordinates (for legibility)
    SCALE = 1e8
    dtheta_s  = (theta_plus  - theta_star)  * SCALE
    dthetad_s = (thetad_plus - thetad_star) * SCALE
    pow10 = int(np.log10(SCALE))

    # (1) Poincaré samples (post-impact)
    plt.figure(figsize=(10, 6))
    plt.title("Poincaré samples (post-impact)")
    plt.plot(dtheta_s, dthetad_s, "-o", linewidth=2, markersize=6)
    plt.scatter([0.0], [0.0], marker="*", s=250, label=r"fixed point $p^\star$")
    plt.xlabel(rf"$(\theta^+ - \theta^\star)\times 10^{{{pow10}}}$")
    plt.ylabel(rf"$(\dot{{\theta}}^+ - \dot{{\theta}}^\star)\times 10^{{{pow10}}}$")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.tight_layout()
    # plt.savefig("figures/CompassGaitFigures/PoincareSamples1.png", dpi=200, bbox_inches="tight")

    # (2) Return map (theta_{k+1}^+ vs theta_k^+) in centered coordinates
    plt.figure(figsize=(10, 6))
    plt.title(r"Return map: $\theta_{k+1}^+$ vs $\theta_k^+$")
    xk  = dtheta_s[:-1]
    xk1 = dtheta_s[1:]
    plt.plot(xk, xk1, "o-", linewidth=2)

    ax = plt.gca()
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--", linewidth=2, label=r"$y=x$")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    plt.xlabel(rf"$(\theta_k^+ - \theta^\star)\times 10^{{{pow10}}}$")
    plt.ylabel(rf"$(\theta_{{k+1}}^+ - \theta^\star)\times 10^{{{pow10}}}$")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.tight_layout()
    # plt.savefig("figures/CompassGaitFigures/ReturnMap1.png", dpi=200, bbox_inches="tight")

    # (3) Step-index convergence: ||p_k - p*||
    dist = np.linalg.norm(poincare_pts - p_star[None, :], axis=1)
    k = np.arange(1, len(dist) + 1)

    plt.figure(figsize=(13, 6))
    plt.title("Step-index Error Convergence")
    plt.semilogy(k, dist, "o-")
    plt.xlabel(r"step number ($k$)")
    plt.ylabel(r"$\|p_k - p^\star\|$ (log scale)")
    plt.xticks(k)  # ensures it ends at 15 (not 14) when NSTEPS=15
    plt.grid(True, which="both")
    plt.tight_layout()
    # plt.savefig("figures/CompassGaitFigures/ConvergencePlot.png", dpi=200, bbox_inches="tight")
'''




plt.show()



