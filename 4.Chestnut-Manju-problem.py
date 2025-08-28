# %%
"""
Chestnut Manju Exponential Growth Simulation
Tasks:
- Compute time to exceed target volume given doubling every 5 minutes.
- Provide general, reusable functions for any target.
- Include preset targets: Solar System (30 AU, 120 AU), example venue.
"""
import math
# Constants & basic utilities
AU_METERS = 1.496e11  # 1 astronomical unit in meters
def sphere_volume(radius_m: float) -> float:
    """Volume of a sphere, V = (4/3) * pi * r^3 (m^3)."""
    return (4.0 / 3.0) * math.pi * (radius_m ** 3)

def ceil_log2(x: float) -> int:
    """Return ceil(log2(x)) for x > 0. If x <= 1, returns 0."""
    if x <= 1.0:
        return 0
    return int(math.ceil(math.log(x, 2)))
# Manju growth model
def initial_manju_volume(radius_m: float = 0.03) -> float:
    """
    Assume one spherical chestnut manju with radius ~3 cm (diameter ~6 cm).
    Returns volume in m^3.
    """
    return sphere_volume(radius_m)

def doublings_to_exceed(target_volume_m3: float,
                        start_volume_m3: float) -> int:
    """
    Minimum doublings n so that start_volume * 2^n >= target_volume.
    """
    ratio = target_volume_m3 / start_volume_m3
    return ceil_log2(ratio)

def time_from_doublings(n_doublings: int, interval_minutes: int = 5):
    """
    Convert number of doublings to time in minutes/hours/days/years.
    """
    minutes = n_doublings * interval_minutes
    hours = minutes / 60.0
    days = hours / 24.0
    years = days / 365.25
    return minutes, hours, days, years

def total_volume_after_doublings(start_volume_m3: float, n_doublings: int) -> float:
    """V_total = V0 * 2^n."""
    return start_volume_m3 * (2 ** n_doublings)

def count_items_after_doublings(n_doublings: int) -> int:
    """
    Number of manju items after n doublings starting from one piece.
    """
    return 2 ** n_doublings

# Target builders
def solar_system_volume_neptune() -> float:
    """Approximate Solar System volume as a sphere out to Neptune (~30 AU)."""
    return sphere_volume(30.0 * AU_METERS)

def solar_system_volume_heliopause() -> float:
    """Very rough sphere out to heliopause (~120 AU)."""
    return sphere_volume(120.0 * AU_METERS)

# Example: venue/container (e.g., a stadium)
def container_volume_example() -> float:
    """
    Example container volume (adjust as needed).
    For instance, a large stadium might be on the order of ~1.2e6 m^3.
    """
    return 1.24e6

# Optional plotting (requires matplotlib)
def plot_growth_until_target(start_volume_m3: float,
                             target_volume_m3: float,
                             interval_minutes: int = 5):
    """
    Plot volume vs time until we meet/exceed the target.
    (Requires matplotlib; safe to comment out if not desired.)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    n = doublings_to_exceed(target_volume_m3, start_volume_m3)
    volumes = [start_volume_m3 * (2 ** k) for k in range(n + 1)]
    times_min = [k * interval_minutes for k in range(n + 1)]

    plt.figure(figsize=(9, 5))
    plt.plot(times_min, volumes, label="Total Manju Volume (m³)")
    plt.axhline(target_volume_m3, linestyle="--", label="Target Volume")
    plt.title("Chestnut Manju Exponential Growth (doubles every 5 min)")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Volume (m³)")
    plt.yscale("log")  # exponential → easier to see on log scale
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Demo / Report
if __name__ == "__main__":
    # Start from 1 manju (sphere radius ≈ 0.03 m)
    V0 = initial_manju_volume(radius_m=0.03)

    # Targets
    V_solar_neptune = solar_system_volume_neptune()
    V_solar_heliopause = solar_system_volume_heliopause()
    V_container = container_volume_example()  # e.g., "Tokyo Dome" scale example

    # Compute doublings and times
    n_nep = doublings_to_exceed(V_solar_neptune, V0)
    n_hel = doublings_to_exceed(V_solar_heliopause, V0)
    n_cont = doublings_to_exceed(V_container, V0)

    t_nep = time_from_doublings(n_nep)
    t_hel = time_from_doublings(n_hel)
    t_cont = time_from_doublings(n_cont)
    # Pretty print helper
    def fmt_time(tup):
        m, h, d, y = tup
        return f"{m:.0f} min | {h:.2f} h | {d:.2f} d | {y:.4f} y"

    print("=== Chestnut Manju (Baibayin) Exponential Growth Report ===")
    print(f"Initial one-manju volume V0: {V0:.6e} m^3")
    print()

    print("[Target] Solar System ~ Neptune orbit (~30 AU):")
    print(f"- Doublings needed: {n_nep}")
    print(f"- Time (5 min each): {fmt_time(t_nep)}")
    print(f"- Manju count then: 2^{n_nep} ≈ {count_items_after_doublings(n_nep):.3e}")
    print()

    print("[Target] Solar System ~ Heliopause (~120 AU, very rough):")
    print(f"- Doublings needed: {n_hel}")
    print(f"- Time (5 min each): {fmt_time(t_hel)}")
    print(f"- Manju count then: 2^{n_hel} ≈ {count_items_after_doublings(n_hel):.3e}")
    print()

    print("[Target] Example large venue (~1.24e6 m^3):")
    print(f"- Doublings needed: {n_cont}")
    print(f"- Time (5 min each): {fmt_time(t_cont)}")
    print(f"- Manju count then: 2^{n_cont} ≈ {count_items_after_doublings(n_cont):.3e}")
    print()
    # Optional: quick plot to the first target
    # plot_growth_until_target(V0, V_solar_neptune)
# %%
