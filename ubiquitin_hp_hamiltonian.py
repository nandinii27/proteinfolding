"""
Ubiquitin HP-Model Hamiltonian
==============================
Reference: Perdomo et al., arXiv:0801.3625v2 (2008)
"On the construction of model Hamiltonians for adiabatic quantum computation
 and its application to finding low energy conformations of lattice protein models"

OVERVIEW
--------
The Hydrophobic-Polar (HP) lattice model (Lau & Dill 1989) is the simplest
meaningful protein model: each residue is either Hydrophobic (H) or Polar (P),
placed on a 2D square grid.  The energy is -1 per non-bonded H-H contact.
The goal is to find the grid conformation (self-avoiding walk) that minimises
this energy — equivalent to maximising H-H contacts.

This script implements EXACTLY the binary variable encoding and Hamiltonian
construction from Perdomo et al. (2008), then solves it classically:
  - N=4 : EXACT enumeration of all valid configurations
  - N=8 : Simulated annealing
  - N=16: Simulated annealing (ubiquitin core)

Key equations from the paper (Section numbers in comments throughout):
  Eq. 1  : bit-string layout for N amino acids on a 2D grid
  Eq. 3  : mapping from binary variable q_i to spin σ_i^z
  Eq. 18 : H_protein = H_onsite + H_psc + H_pairwise
  Eq. 19-21: Onsite repulsion (no two residues on same site)
  Eq. 22-24: Primary sequence constraint (chain connectivity)
  Eq. 25-29: Pairwise hydrophobic interaction (H-H contacts)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import itertools
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Ubiquitin sequence and HP mapping
# ─────────────────────────────────────────────────────────────────────────────

# Full 76-residue human ubiquitin (UniProt P0CG48)
UBIQUITIN_FULL = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)
assert len(UBIQUITIN_FULL) == 76

# HP classification: Hydrophobic = non-polar / aromatic residues
# (Lau & Dill 1989 standard convention)
HP_TABLE = {
    'A': 'H', 'C': 'H', 'F': 'H', 'G': 'P', 'I': 'H',
    'L': 'H', 'M': 'H', 'V': 'H', 'W': 'H', 'Y': 'H',
    'D': 'P', 'E': 'P', 'H': 'P', 'K': 'P', 'N': 'P',
    'Q': 'P', 'R': 'P', 'S': 'P', 'T': 'P', 'P': 'P',
}

def sequence_to_hp(seq):
    """Convert a one-letter amino acid sequence to an HP binary string (H/P)."""
    return "".join(HP_TABLE[aa] for aa in seq)

UBIQUITIN_HP = sequence_to_hp(UBIQUITIN_FULL)

print("Full ubiquitin (76 aa):", UBIQUITIN_FULL)
print("HP encoding           :", UBIQUITIN_HP)
print(f"H count: {UBIQUITIN_HP.count('H')},  P count: {UBIQUITIN_HP.count('P')}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Bit-string layout and index helpers (Paper Sec. II.A, Eq. 1 & 21)
#
#     For N amino acids on a 2D grid, each residue needs 2 coordinates (x, y),
#     each encoded in log2(N) bits.  The full bit string has N*D*log2(N) bits.
#
#     Convention (Eq. 1):  the bit string is ordered as
#       q = ... | y_N x_N | ... | y_2 x_2 | y_1 x_1 |
#     where x_i = [bit_r1, bit_r2, ...] with bit_r1 = LSB.
#
#     f(i, k) (Eq. 21): index of the first (LSB) bit of the k-th coordinate
#     of the i-th amino acid (all indices 1-based in the paper; we use 0-based
#     array indices internally by subtracting 1 from r when accessing q).
# ─────────────────────────────────────────────────────────────────────────────

def f_index(i, k, N, D=2):
    """
    Eq. 21 — base bit-index for coordinate k of residue i.
    f(i,k) = D*(i-1)*log2(N) + (k-1)*log2(N)
    i, k are 1-based; returned index is 0-based (for use as q[f + r-1]).
    """
    log2N = int(math.log2(N))
    return D * (i - 1) * log2N + (k - 1) * log2N


def get_coordinate(q, i, k, N, D=2):
    """
    Decode the integer value of coordinate k for residue i from bit-string q.
    Bits are arranged LSB-first: coord = sum_r q[f+r-1] * 2^(r-1).
    """
    log2N = int(math.log2(N))
    base  = f_index(i, k, N, D)
    return sum(q[base + r - 1] * (2 ** (r - 1)) for r in range(1, log2N + 1))


def decode_conformation(q, N, D=2):
    """
    Convert a full bit-string q to a list of (x, y) lattice coordinates,
    one entry per amino acid (0-indexed list, i.e. coords[0] = residue 1).
    """
    return [(get_coordinate(q, i, 1, N, D),
             get_coordinate(q, i, 2, N, D)) for i in range(1, N + 1)]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Coordinate-based energy functions
#
#     The paper derives Boolean/polynomial energy functions over the binary
#     variables q_i.  Here we implement the equivalent COORDINATE-LEVEL
#     energy functions that are mathematically identical but far faster to
#     evaluate once the coordinates are decoded.
#
#     This is valid because the Hamiltonian mapping (Eq. 3) is a bijection:
#     every configuration of bits q corresponds to a unique set of coordinates,
#     so evaluating E(coords) is the same as evaluating E(q).
#
#     The binary/polynomial form is needed for the quantum Hamiltonian matrix;
#     the coordinate form is used for the classical optimiser.
# ─────────────────────────────────────────────────────────────────────────────

def energy_from_coords(coords, hp_string, N, lam0=None, lam1=None):
    """
    Compute H_protein = H_onsite + H_psc + H_pairwise directly from coordinates.
    This is the coordinate-level equivalent of Eq. 18.

    Parameters
    ----------
    coords    : list of (x,y) tuples, length N
    hp_string : HP-encoded sequence, length N  ('H' or 'P' per residue)
    N         : number of residues
    lam0      : onsite penalty  (default N+1, paper Sec. III.B.2)
    lam1      : chain penalty   (default N,   paper Sec. III.B.2)

    Returns
    -------
    float : total energy (lower = better fold)
    """
    if lam0 is None: lam0 = N + 1
    if lam1 is None: lam1 = N

    # ── H_onsite (Eq. 19-20): penalise any two residues at the same site ──
    # H_onsite = lam0 * #{pairs (i,j) with i<j and coords[i]==coords[j]}
    coord_set = {}
    onsite = 0
    for idx, c in enumerate(coords):
        if c in coord_set:
            onsite += 1   # collision with every previous residue at this site
        coord_set.setdefault(c, []).append(idx)
    # Use the exact pair count, not just occupancy count
    onsite = 0
    from itertools import combinations
    for c, indices in coord_set.items():
        onsite += len(list(combinations(indices, 2)))
    H_onsite_val = lam0 * onsite

    # ── H_psc (Eq. 22-24): chain constraint ──────────────────────────────
    # d²(m, m+1) = (x_{m+1}-x_m)^2 + (y_{m+1}-y_m)^2  (Euclidean²  = L1²
    #              for unit steps, but the paper uses L1 distance squared)
    # Actually Eq. 22 computes the L1 distance as sum of abs differences:
    # d²_PQ = sum_k (base-10 value of coord differences)^2
    # For unit steps this equals sum_k (delta_k)^2, which equals 1 for valid bonds.
    chain_sum = 0
    for m in range(N - 1):
        dx = coords[m + 1][0] - coords[m][0]
        dy = coords[m + 1][1] - coords[m][1]
        chain_sum += dx * dx + dy * dy
    # H_psc = lam1 * [-(N-1) + sum d²]  →  0 for valid chain (all d²=1)
    H_psc_val = lam1 * (-(N - 1) + chain_sum)

    # ── H_pairwise (Eq. 25-29): hydrophobic contacts ──────────────────────
    # H_pairwise = -#{non-bonded H-H nearest-neighbour pairs}
    hh = 0
    for i in range(N):
        for j in range(i + 2, N):          # skip sequence neighbours
            if hp_string[i] == 'H' and hp_string[j] == 'H':
                dx = abs(coords[i][0] - coords[j][0])
                dy = abs(coords[i][1] - coords[j][1])
                if dx + dy == 1:           # L1 nearest neighbour
                    hh += 1
    H_pairwise_val = -hh

    return H_onsite_val + H_psc_val + H_pairwise_val


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Translational symmetry fix (Paper Sec. II.A)
#
#     Fix the two middle residues (N/2 and N/2+1) at the grid centre to
#     remove translational degeneracy.  The remaining residues are free.
#
#     Paper convention:
#       residue N/2   → grid position (N/2, N/2)
#       residue N/2+1 → grid position (N/2+1, N/2)   (one step right)
# ─────────────────────────────────────────────────────────────────────────────

def build_fixed_q(N, D=2):
    """
    Returns a dict {bit_index: fixed_value} for the two anchor residues.
    """
    log2N = int(math.log2(N))
    mid   = N // 2
    mid1  = mid + 1
    center = N // 2         # integer grid coordinate for both anchor residues

    # Helper: encode integer v in log2N bits (LSB first)
    def to_bits(v):
        return [(v >> r) & 1 for r in range(log2N)]

    center_bits  = to_bits(center)
    center1_bits = to_bits(center + 1)   # N/2 + 1

    fixed = {}
    # Residue mid: x = N/2, y = N/2
    for k, cbits in enumerate([center_bits, center_bits], start=1):
        base = f_index(mid, k, N, D)
        for r in range(log2N):
            fixed[base + r] = cbits[r]

    # Residue mid+1: x = N/2+1, y = N/2
    for k, cbits in enumerate([center1_bits, center_bits], start=1):
        base = f_index(mid1, k, N, D)
        for r in range(log2N):
            fixed[base + r] = cbits[r]

    return fixed


def make_q_from_free(free_bits, fixed_map, total_bits):
    """
    Reconstruct full bit-string q from free bits (in index order) + fixed values.
    """
    q           = [0] * total_bits
    free_indices = sorted(idx for idx in range(total_bits) if idx not in fixed_map)
    for val, idx in zip(free_bits, free_indices):
        q[idx] = val
    for idx, val in fixed_map.items():
        q[idx] = val
    return q


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Exact enumeration (feasible only for small N)
#
#     For N=4: total bits = 4*2*2 = 16; fixed bits = 8 (both anchors use
#     2*D*log2(4) = 2*2*2 = 8 bits); free bits = 8 → 2^8 = 256 configs.
#     This is fully tractable.
#
#     For N=8: free bits = 8*2*3 - 2*2*3 = 48-12 = 36 → 2^36 ≈ 68B.
#     Not feasible for exact enumeration → use SA.
# ─────────────────────────────────────────────────────────────────────────────

def enumerate_energy_landscape(hp_string, N, D=2):
    """
    Exhaustively evaluate H_protein for every 2^(n_free) configuration.
    Practical only for N ≤ 4 (or small fragments where n_free ≤ ~20).

    Returns
    -------
    energies    : sorted list of (energy, q)
    best_energy : float
    best_q      : list[int]  (full bit-string)
    """
    log2N      = int(math.log2(N))
    total_bits = N * D * log2N
    fixed_map  = build_fixed_q(N, D)
    n_free     = total_bits - len(fixed_map)
    print(f"[Exact enum] N={N}, total bits={total_bits}, free bits={n_free}, "
          f"configs={2**n_free:,}")

    best_energy = float('inf')
    best_q      = None
    energies    = []

    for free_combo in itertools.product([0, 1], repeat=n_free):
        q      = make_q_from_free(list(free_combo), fixed_map, total_bits)
        coords = decode_conformation(q, N, D)
        E      = energy_from_coords(coords, hp_string, N)
        energies.append((E, q[:]))
        if E < best_energy:
            best_energy = E
            best_q      = q[:]

    energies.sort(key=lambda x: x[0])
    return energies, best_energy, best_q


# ─────────────────────────────────────────────────────────────────────────────
# 5b.  Valid SAW generator (for SA warm-start)
#
#      We generate a random self-avoiding walk on the 2D lattice and encode
#      it into the binary bit-string q.  This gives the SA a valid starting
#      point so it explores the valid-conformation subspace more efficiently.
# ─────────────────────────────────────────────────────────────────────────────

def random_saw(N, grid_size, seed=None):
    """
    Generate a random self-avoiding walk of length N on a grid_size × grid_size
    lattice using a backtracking approach.  Returns list of (x,y) or None.
    """
    rng   = random.Random(seed)
    DIRS  = [(1,0),(-1,0),(0,1),(0,-1)]
    start = (grid_size // 2, grid_size // 2)

    def backtrack(path, visited):
        if len(path) == N:
            return path
        dirs = DIRS[:]
        rng.shuffle(dirs)
        x, y = path[-1]
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    path.append((nx, ny))
                    result = backtrack(path, visited)
                    if result is not None:
                        return result
                    path.pop()
                    visited.discard((nx, ny))
        return None

    return backtrack([start], {start})


def saw_to_q(coords, N, D=2):
    """
    Encode a list of (x,y) coordinates into the binary bit-string q.
    The two anchor residues (N/2, N/2+1) are placed at the centre.
    We shift coords so that anchor residues align with the paper's convention.
    """
    log2N = int(math.log2(N))
    total_bits = N * D * log2N
    mid  = N // 2          # 1-indexed
    mid0 = mid - 1         # 0-indexed

    # Shift the walk so residue N/2 lands at (N/2, N/2)
    target_x = N // 2
    target_y = N // 2
    dx = target_x - coords[mid0][0]
    dy = target_y - coords[mid0][1]
    shifted = [(x + dx, y + dy) for x, y in coords]

    # Clamp to valid range [0, N-1]
    # (if shift pushes out of bounds, we just encode whatever; SA will fix it)
    q = [0] * total_bits
    for i, (x, y) in enumerate(shifted):
        res = i + 1          # 1-indexed residue
        # encode x (k=1)
        base_x = f_index(res, 1, N, D)
        for r in range(log2N):
            q[base_x + r] = (x >> r) & 1
        # encode y (k=2)
        base_y = f_index(res, 2, N, D)
        for r in range(log2N):
            q[base_y + r] = (y >> r) & 1
    return q


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Simulated Annealing (Kirkpatrick et al. 1983)
#
#     Single-bit-flip Metropolis MC with a geometric cooling schedule.
#     Each move flips one random free bit, recomputes the energy from
#     coordinates, and accepts/rejects via exp(-ΔE/T).
#
#     The key insight: each bit flip changes AT MOST one residue's position
#     by a power-of-two step, so the energy change is localised and cheap.
# ─────────────────────────────────────────────────────────────────────────────

def simulated_annealing(hp_string, N, D=2,
                         T_start=10.0, T_end=0.01,
                         n_steps=50_000, seed=42):
    """
    Minimise H_protein via simulated annealing with SAW warm-start.

    Initialisation: generate a random self-avoiding walk (valid chain) and
    encode it into q.  This puts the SA inside the valid-conformation subspace
    immediately, so the penalty terms H_onsite and H_psc start at 0 and the
    optimiser focuses on maximising H-H contacts.

    Returns
    -------
    best_E    : float   best energy found
    best_q    : list    corresponding bit-string
    trace     : list[(step, energy)]   for convergence plotting
    """
    random.seed(seed)
    log2N      = int(math.log2(N))
    total_bits = N * D * log2N
    fixed_map  = build_fixed_q(N, D)
    free_indices = sorted(idx for idx in range(total_bits) if idx not in fixed_map)
    n_free     = len(free_indices)

    # ── Warm-start: encode a random valid SAW ────────────────────────────
    # Try to generate a valid SAW on a (2N) x (2N) grid (enough room)
    saw_coords = random_saw(N, grid_size=2*N, seed=seed)
    if saw_coords is not None:
        q = saw_to_q(saw_coords, N, D)
        # Overwrite fixed bits with correct anchor values
        for idx, val in fixed_map.items():
            q[idx] = val
        print(f"  Warm-start from valid SAW (E_initial check)")
    else:
        # Fallback to random initialisation if SAW generation fails
        q = [0] * total_bits
        for idx in free_indices:
            q[idx] = random.randint(0, 1)
        for idx, val in fixed_map.items():
            q[idx] = val
        print(f"  Warm-start SAW failed, using random init")
    # Ensure list of ints
    q = [int(b) for b in q]

    current_coords = decode_conformation(q, N, D)
    current_E      = energy_from_coords(current_coords, hp_string, N)
    best_E         = current_E
    best_q         = q[:]
    trace          = [(0, current_E)]

    cooling = (T_end / T_start) ** (1.0 / n_steps)
    T       = T_start

    for step in range(1, n_steps + 1):
        # Propose: flip one random free bit
        flip_pos = free_indices[random.randrange(n_free)]
        q[flip_pos] ^= 1

        new_coords = decode_conformation(q, N, D)
        new_E      = energy_from_coords(new_coords, hp_string, N)
        delta_E    = new_E - current_E

        # Metropolis acceptance
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            current_E      = new_E
            current_coords = new_coords
            if new_E < best_E:
                best_E = new_E
                best_q = q[:]
        else:
            q[flip_pos] ^= 1   # revert

        T *= cooling

        if step % max(1, n_steps // 200) == 0:
            trace.append((step, current_E))

    print(f"  SA done (N={N}): best E = {best_E}")
    return best_E, best_q, trace


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_valid_conformation(coords, N):
    """
    Return (valid: bool, reason: str).
    Checks: (1) self-avoidance, (2) chain connectivity (L1 = 1 between neighbours).
    """
    if len(set(coords)) < N:
        return False, "Collision: two residues on same site."
    for m in range(N - 1):
        d = abs(coords[m+1][0]-coords[m][0]) + abs(coords[m+1][1]-coords[m][1])
        if d != 1:
            return False, f"Chain break between {m+1} and {m+2} (L1={d})."
    return True, "Valid self-avoiding walk."


def count_hh_contacts(coords, hp_string):
    """Count non-bonded H-H nearest-neighbour contacts (the HP model energy term)."""
    N = len(coords)
    hh = 0
    for i in range(N):
        for j in range(i + 2, N):
            if hp_string[i] == 'H' and hp_string[j] == 'H':
                if abs(coords[i][0]-coords[j][0]) + abs(coords[i][1]-coords[j][1]) == 1:
                    hh += 1
    return hh


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_conformation(coords, hp_string, title="", ax=None, filename=None):
    """
    Draw the 2D lattice conformation.
    Blue beads = H (hydrophobic), orange beads = P (polar).
    Dashed lines = H-H contacts, solid line = backbone.
    """
    N = len(coords)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    ax.plot(xs, ys, 'k-', linewidth=1.5, zorder=1)   # backbone

    # H-H contacts
    for i in range(N):
        for j in range(i + 2, N):
            if hp_string[i] == 'H' and hp_string[j] == 'H':
                if abs(coords[i][0]-coords[j][0]) + abs(coords[i][1]-coords[j][1]) == 1:
                    ax.plot([coords[i][0], coords[j][0]],
                            [coords[i][1], coords[j][1]],
                            'b--', lw=1.2, alpha=0.6, zorder=2)

    # Beads
    for idx, (x, y) in enumerate(coords):
        c = '#1a6bb5' if hp_string[idx] == 'H' else '#f07f1e'
        ax.add_patch(plt.Circle((x, y), 0.32, color=c, zorder=3))
        ax.text(x, y, str(idx+1), ha='center', va='center',
                fontsize=6, color='white', fontweight='bold', zorder=4)

    margin = 1.5
    ax.set_xlim(min(xs)-margin, max(xs)+margin)
    ax.set_ylim(min(ys)-margin, max(ys)+margin)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(handles=[
        mpatches.Patch(color='#1a6bb5', label='H (hydrophobic)'),
        mpatches.Patch(color='#f07f1e', label='P (polar)'),
    ], loc='upper right', fontsize=9)

    if filename:
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"  Saved: {filename}")


def plot_energy_landscape(energies, title="", filename=None):
    """Histogram of all enumerated configuration energies."""
    E_vals = [e for e, _ in energies]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(E_vals, bins=30, color='steelblue', edgecolor='k', alpha=0.8)
    ax.axvline(min(E_vals), color='red', ls='--', lw=2,
               label=f"Ground state E = {min(E_vals)}")
    ax.set_xlabel("Energy"); ax.set_ylabel("# configurations")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"  Saved: {filename}")


def plot_sa_trace(trace, title="", filename=None):
    """Plot SA energy vs MC step."""
    steps = [s for s, _ in trace]
    vals  = [e for _, e in trace]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, vals, color='darkorange', lw=1)
    ax.set_xlabel("MC step"); ax.set_ylabel("Energy")
    ax.set_title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150)
        print(f"  Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Main driver
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("UBIQUITIN HP-MODEL HAMILTONIAN  (Perdomo et al. 2008)")
    print("=" * 65)

    # ── 9a. N=4 exact enumeration ─────────────────────────────────────────
    print("\n─── PART A: N=4 exact enumeration ───")
    SEQ4 = UBIQUITIN_FULL[:4]          # "MQIF"
    HP4  = sequence_to_hp(SEQ4)
    print(f"Sequence: {SEQ4}  →  HP: {HP4}")

    energies4, best_E4, best_q4 = enumerate_energy_landscape(HP4, N=4)
    coords4   = decode_conformation(best_q4, 4)
    valid4, r4 = is_valid_conformation(coords4, 4)
    hh4        = count_hh_contacts(coords4, HP4)

    print(f"Best energy  : {best_E4}")
    print(f"Coordinates  : {coords4}")
    print(f"Valid walk   : {valid4}  ({r4})")
    print(f"H-H contacts : {hh4}")
    print(f"Ground-state degeneracy : "
          f"{sum(1 for e,_ in energies4 if e==best_E4)}")
    print(f"Unique energy levels    : "
          f"{sorted(set(e for e,_ in energies4))[:8]}")

    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 5))
    # Left: energy landscape
    E_vals = [e for e, _ in energies4]
    axes4[0].hist(E_vals, bins=25, color='steelblue', edgecolor='k', alpha=0.8)
    axes4[0].axvline(best_E4, color='red', ls='--', lw=2,
                     label=f"Ground state E={best_E4}")
    axes4[0].set_xlabel("Energy"); axes4[0].set_ylabel("# configs")
    axes4[0].set_title(f"Energy landscape — N=4 ({HP4})")
    axes4[0].legend()
    # Right: best conformation
    plot_conformation(coords4, HP4,
                      title=f"Best conformation N=4, E={best_E4}, HH={hh4}",
                      ax=axes4[1])
    plt.tight_layout()
    plt.savefig("/home/user/workspace/results_N4.png", dpi=150)
    print("  Saved: results_N4.png")

    # ── 9b. N=8 simulated annealing ───────────────────────────────────────
    print("\n─── PART B: N=8 simulated annealing ───")
    SEQ8 = UBIQUITIN_FULL[:8]          # "MQIFVKTL"
    HP8  = sequence_to_hp(SEQ8)
    print(f"Sequence: {SEQ8}  →  HP: {HP8}")

    # Run 5 independent SA restarts and keep the best valid solution
    best_E8, best_q8, trace8 = float('inf'), None, []
    for sa_seed in [7, 11, 23, 37, 53]:
        e, q_, t_ = simulated_annealing(
            HP8, N=8, T_start=4.0, T_end=0.005, n_steps=120_000, seed=sa_seed)
        c_ = decode_conformation(q_, 8)
        v_, _ = is_valid_conformation(c_, 8)
        if v_ and e < best_E8:
            best_E8, best_q8, trace8 = e, q_, t_
    # Fallback: if no valid found, take absolute best
    if best_q8 is None:
        best_E8, best_q8, trace8 = simulated_annealing(
            HP8, N=8, T_start=4.0, T_end=0.005, n_steps=120_000, seed=7)

    coords8   = decode_conformation(best_q8, 8)
    valid8, r8 = is_valid_conformation(coords8, 8)
    hh8        = count_hh_contacts(coords8, HP8)
    print(f"Best energy  : {best_E8}")
    print(f"Valid walk   : {valid8}  ({r8})")
    print(f"H-H contacts : {hh8}")

    fig8, axes8 = plt.subplots(1, 2, figsize=(12, 5))
    steps8 = [s for s,_ in trace8]; vals8 = [e for _,e in trace8]
    axes8[0].plot(steps8, vals8, color='darkorange', lw=1)
    axes8[0].set_xlabel("MC step"); axes8[0].set_ylabel("Energy")
    axes8[0].set_title(f"SA trace — N=8 ({HP8})")
    plot_conformation(coords8, HP8,
                      title=f"Best conformation N=8, E={best_E8}, HH={hh8}",
                      ax=axes8[1])
    plt.tight_layout()
    plt.savefig("/home/user/workspace/results_N8.png", dpi=150)
    print("  Saved: results_N8.png")

    # ── 9c. N=16 (ubiquitin core) simulated annealing ────────────────────
    print("\n─── PART C: N=16 (ubiquitin core residues 1-16) ───")
    SEQ16 = UBIQUITIN_FULL[:16]
    HP16  = sequence_to_hp(SEQ16)
    print(f"Sequence: {SEQ16}")
    print(f"HP      : {HP16}")

    best_E16, best_q16, trace16 = simulated_annealing(
        HP16, N=16, T_start=8.0, T_end=0.01, n_steps=150_000, seed=13)

    coords16   = decode_conformation(best_q16, 16)
    valid16, r16 = is_valid_conformation(coords16, 16)
    hh16         = count_hh_contacts(coords16, HP16)
    print(f"Best energy  : {best_E16}")
    print(f"Valid walk   : {valid16}  ({r16})")
    print(f"H-H contacts : {hh16}")

    fig16, axes16 = plt.subplots(1, 2, figsize=(14, 6))
    steps16 = [s for s,_ in trace16]; vals16 = [e for _,e in trace16]
    axes16[0].plot(steps16, vals16, color='darkorange', lw=1)
    axes16[0].set_xlabel("MC step"); axes16[0].set_ylabel("Energy")
    axes16[0].set_title(f"SA trace — N=16 ubiquitin core")
    plot_conformation(coords16, HP16,
                      title=f"Best conformation N=16, E={best_E16}, HH={hh16}",
                      ax=axes16[1])
    plt.tight_layout()
    plt.savefig("/home/user/workspace/results_N16.png", dpi=150)
    print("  Saved: results_N16.png")

    # ── 9d. Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    rows = [
        (SEQ4,  4,  "Exact enum",  best_E4,  hh4,  valid4),
        (SEQ8,  8,  "Sim. annealing", best_E8, hh8, valid8),
        (SEQ16, 16, "Sim. annealing", best_E16, hh16, valid16),
    ]
    hdr = f"{'Sequence':<18} {'N':>3} {'Method':<17} {'Best E':>7} {'HH':>4} {'Valid':>6}"
    print(hdr)
    print("-" * 60)
    for seq, n, method, be, hh, v in rows:
        print(f"{seq:<18} {n:>3} {method:<17} {be:>7} {hh:>4} {str(v):>6}")
    print("=" * 65)
    print("""
Scalability note (Eq. bit-count: N*D*log2(N))
  N=4  : 16 bits, 8 free  → 2^8   = 256      [exact]
  N=8  : 48 bits, 36 free → 2^36  = 68B      [SA only]
  N=16 : 128 bits, 116 free                  [SA only]
  N=76 (full ubiquitin, padded to 128):
        128*2*7 = 1792 qubits                 [quantum hardware needed]
""")
