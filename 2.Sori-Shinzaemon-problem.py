# %%
import matplotlib.pyplot as plt
def compute_sorori_shinzaemon(n_days=100):
    """
    Compute the rice grains received daily and in total,
    based on Sorori Shinzaemon's exponential growth request.

    Parameters
    ----------
    n_days : int
        Number of days rice is given (default = 100)

    Returns
    -------
    list_n_grains : list
        Rice grains received each day
    list_total_grains : list
        Cumulative rice grains received up to that day
    """
    list_n_grains = []        # rice per day
    list_total_grains = []    # cumulative rice

    total = 0
    for day in range(n_days):
        grains_today = 2 ** day   # exponential growth
        total += grains_today
        list_n_grains.append(grains_today)
        list_total_grains.append(total)

    return list_n_grains, list_total_grains
# Example: calculate for 100 days
list_n_grains, list_total_grains = compute_sorori_shinzaemon(n_days=100)

print(f"Rice on day 100: {list_n_grains[-1]:,} grains")
print(f"Total rice after 100 days: {list_total_grains[-1]:,} grains")
# (b) Plot graph
plt.figure(figsize=(10,5))
plt.plot(range(1, 101), list_n_grains, label="Rice per day")
plt.plot(range(1, 101), list_total_grains, label="Cumulative rice")

plt.title("Sorori Shinzaemon Rice Growth (100 days)")
plt.xlabel("Day")
plt.ylabel("Number of grains")
plt.yscale("log")  # exponential growth → log scale for better visibility
plt.legend()
plt.grid(True)
plt.show()
# (c) Estimate human survival
def estimate_survival(total_grains, people=1):
    """
    Estimate how many days people can live on the rice.

    Assumption: 50,000 grains = 1 person per day.

    Parameters
    ----------
    total_grains : int
        Total rice grains available
    people : int
        Number of people sharing the rice

    Returns
    -------
    days : float
        Number of days the rice can sustain the people
    """
    grains_per_day = 50_000 * people
    return total_grains / grains_per_day
# Example: how long can 5 people survive on 100-day total rice?
survival_days = estimate_survival(list_total_grains[-1], people=5)
print(f"With 5 people, the rice lasts about {survival_days:,.2f} days.")
# %%
