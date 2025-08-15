from pathlib import Path
import re, json

import numpy as np

import matplotlib.pyplot as plt

folder_path     = Path("/home/mila/n/nearyc/scratch/projects/vlaps_data/experiments/libero/results")
task_suites      = ["spatial", "object", "goal", "10", "90"]
checkpoints = {10000, 50000, 100000, 150000, 200000}

vla_exp_data = {
    'spatial' : {
        10000 : '2025-07-24_17-19-01_octo_vla_spatial_ckpt10000',
        50000 : '2025-07-24_17-51-01_octo_vla_spatial_ckpt50000',
        100000 : '2025-07-24_18-16-38_octo_vla_spatial_ckpt100000',
        150000 : '2025-07-24_18-34-02_octo_vla_spatial_ckpt150000',
        200000 : '2025-07-24_18-50-08_octo_vla_spatial_ckpt200000',
    },
    'object' : {
        10000 : '2025-07-24_18-31-46_octo_vla_object_ckpt10000',
        50000 : '2025-07-24_18-59-55_octo_vla_object_ckpt50000',
        100000 : '2025-07-24_19-26-24_octo_vla_object_ckpt100000',
        150000 : '2025-07-24_19-49-39_octo_vla_object_ckpt150000',
        200000 : '2025-07-24_20-14-10_octo_vla_object_ckpt200000',
    },
    'goal' : {
        10000 : '2025-07-24_18-30-08_octo_vla_goal_ckpt10000',
        50000 : '2025-07-24_19-08-00_octo_vla_goal_ckpt50000',
        100000 : '2025-07-24_19-32-00_octo_vla_goal_ckpt100000',
        150000 : '2025-07-24_19-46-38_octo_vla_goal_ckpt150000',
        200000 : '2025-07-24_20-01-16_octo_vla_goal_ckpt200000',
    },
    '90' : {
        10000 : '2025-07-24_19-05-50_octo_vla_90_ckpt10000',
        50000 : '2025-07-24_20-20-55_octo_vla_90_ckpt50000',
        100000 : '2025-07-24_21-28-24_octo_vla_90_ckpt100000',
        150000 : '2025-07-24_22-11-03_octo_vla_90_ckpt150000',
        200000 : '2025-07-24_22-51-39_octo_vla_90_ckpt200000',
    },
    '10' : {
        10000 : '2025-07-24_19-05-50_octo_vla_10_ckpt10000',
        50000 : '2025-07-24_20-25-04_octo_vla_10_ckpt50000',
        100000 : '2025-07-24_21-45-43_octo_vla_10_ckpt100000',
        150000 : '2025-07-24_22-51-22_octo_vla_10_ckpt150000',
        200000 : '2025-07-24_23-48-56_octo_vla_10_ckpt200000',
    },
}

vlaps_exp_data = {
    'spatial' : {
        10000 : '2025-07-25_03-23-04_octo_vlaps_spatial_ckpt10000',
        50000 : '2025-07-27_13-19-46_octo_vlaps_spatial_ckpt50000',
        100000 : '2025-07-28_12-35-33_octo_vlaps_spatial_ckpt100000',
        150000 : '2025-07-28_13-17-45_octo_vlaps_spatial_ckpt150000',
        200000 : '2025-07-28_14-15-19_octo_vlaps_spatial_ckpt200000',
    },
    'object' : {
        10000 : '2025-07-25_04-42-05_octo_vlaps_object_ckpt10000',
        50000 : '2025-07-27_13-19-54_octo_vlaps_object_ckpt50000',
        100000 : '2025-07-28_12-44-38_octo_vlaps_object_ckpt100000',
        150000 : '2025-07-28_16-55-57_octo_vlaps_object_ckpt150000',
        200000 : '2025-07-29_02-11-44_octo_vlaps_object_ckpt200000',
    },
    'goal' : {
        10000 : '2025-07-25_05-09-40_octo_vlaps_goal_ckpt10000',
        50000 : '2025-07-27_13-19-58_octo_vlaps_goal_ckpt50000',
        100000 : '2025-07-28_12-38-38_octo_vlaps_goal_ckpt100000',
        150000 : '2025-07-28_14-18-19_octo_vlaps_goal_ckpt150000',
        200000 : '2025-07-28_15-42-50_octo_vlaps_goal_ckpt200000',
    },
    '90' : {
        10000 : '2025-07-25_07-06-10_octo_vlaps_90_ckpt10000',
        50000 : '2025-07-27_19-55-24_octo_vlaps_90_ckpt50000',
        100000 : '2025-07-28_12-49-14_octo_vlaps_90_ckpt100000',
        150000 : '2025-07-28_15-06-38_octo_vlaps_90_ckpt150000',
        200000 : '2025-07-28_17-11-13_octo_vlaps_90_ckpt200000',
    },
    '10' : {
        10000 : '2025-07-25_05-45-48_octo_vlaps_10_ckpt10000',
        50000 : '2025-07-27_13-19-36_octo_vlaps_10_ckpt50000',
        100000 : '2025-07-29_15-18-52_octo_vlaps_10_ckpt100000',
        150000 : '2025-07-29_02-13-09_octo_vlaps_10_ckpt150000',
        200000 : '2025-07-29_02-14-23_octo_vlaps_10_ckpt200000',
    },
}

vla_success_rates = {ts: {} for ts in task_suites}
vla_times = {ts: {} for ts in task_suites}
vlaps_success_rates = {ts: {} for ts in task_suites}
vlaps_times = {ts: {} for ts in task_suites}

vla_success_times = {ts : {ckpt : [] for ckpt in checkpoints} for ts in task_suites}
vla_fail_times = {ts : {ckpt : [] for ckpt in checkpoints} for ts in task_suites}
vlaps_success_times = {ts : {ckpt : [] for ckpt in checkpoints} for ts in task_suites}
vlaps_fail_times = {ts : {ckpt : [] for ckpt in checkpoints} for ts in task_suites}

for suite in vla_exp_data.keys():
    for ckpt in vla_exp_data[suite].keys():
        exp_name = vla_exp_data[suite][ckpt]

        # read the stats
        stats_file = folder_path / exp_name / "multi_run_statistics.json"
        with stats_file.open(encoding="utf-8") as f:
            data = json.load(f)

        vla_success_rates[suite][ckpt] = data["num_successes"] / data["num_runs"]
        vla_times[suite][ckpt] = [min(v, 600) for k,v in data['run_times'].items()]

        for run_name in data['run_successes']:
            vla_success_times[suite][ckpt].append(min(data['run_times'][run_name], 600))
        for run_name in data['run_failures']:
            vla_fail_times[suite][ckpt].append(min(data['run_times'][run_name], 600))

for suite in vlaps_exp_data.keys():
    for ckpt in vlaps_exp_data[suite].keys():
        exp_name = vlaps_exp_data[suite][ckpt]

        # read the stats
        stats_file = folder_path / exp_name / "multi_run_statistics.json"
        with stats_file.open(encoding="utf-8") as f:
            data = json.load(f)

        vlaps_success_rates[suite][ckpt] = data["num_successes"] / data["num_runs"]
        vlaps_times[suite][ckpt] = [min(v, 600) for k,v in data['run_times'].items()]

        for run_name in data['run_successes']:
            vlaps_success_times[suite][ckpt].append(min(data['run_times'][run_name], 600))
        for run_name in data['run_failures']:
            vlaps_fail_times[suite][ckpt].append(min(data['run_times'][run_name], 600))
        
# Get the average success rate across all task suites for each ckpt.
overall_vlaps_success_rates_per_ckpt = {
    ckpt : np.mean([vlaps_success_rates[ts][ckpt] for ts in task_suites]) for ckpt in checkpoints
}
overall_vla_success_rates_per_ckpt = {
    ckpt : np.mean([vla_success_rates[ts][ckpt] for ts in task_suites]) for ckpt in checkpoints
}

# Get the average runtime across all successful task evaluations in all task suites for each ckpt.
overall_vlaps_success_times_per_ckpt = {
    ckpt : np.mean([np.mean(vlaps_success_times[ts][ckpt]) for ts in task_suites if vlaps_success_times[ts][ckpt]]) for ckpt in checkpoints
}
overall_vla_success_times_per_ckpt = {
    ckpt : np.mean([np.mean(vla_success_times[ts][ckpt]) for ts in task_suites if vla_success_times[ts][ckpt]]) for ckpt in checkpoints
}

# Get the average runtime across all task evaluations in all task suites for each ckpt.
overall_vlaps_times_per_ckpt = {
    ckpt : np.mean([np.mean(vlaps_times[ts][ckpt]) for ts in task_suites if vlaps_times[ts][ckpt]]) for ckpt in checkpoints
}
overall_vla_times_per_ckpt = {
    ckpt : np.mean([np.mean(vla_times[ts][ckpt]) for ts in task_suites if vla_times[ts][ckpt]]) for ckpt in checkpoints
}

# Get the average runtime across all successful task evaluations for each task suite and each ckpt.
overall_vlaps_success_times_per_suite_and_ckpt = {
    ts : {ckpt : np.mean(vlaps_success_times[ts][ckpt]) for ckpt in checkpoints} for ts in task_suites
}
overall_vla_success_times_per_suite_and_ckpt = {
    ts : {ckpt : np.mean(vla_success_times[ts][ckpt]) for ckpt in checkpoints} for ts in task_suites
}

# Get the average runtime across all task evaluations for each task suite and each ckpt.
overall_vlaps_times_per_suite_and_ckpt = {
    ts : {ckpt : np.mean(vlaps_times[ts][ckpt]) for ckpt in checkpoints} for ts in task_suites
}
overall_vla_times_per_suite_and_ckpt = {
    ts : {ckpt : np.mean(vla_times[ts][ckpt]) for ckpt in checkpoints} for ts in task_suites
}

# exp_descriptions = ["ckpt", "octo"]
# exp_dates = ["2025-07-18", "2025-07-14", "2025-07-17"]
# # checkpoints      = {10000, 20000, 30000, 40000, 50000, 60000,
# #                     70000, 80000, 90000, 100000, 110000, 120000,
# #                     130000, 140000, 150000, 160000, 170000, 180000,
# #                     190000, 200000}

# ckpt_re = re.compile(r"t(\d+)(?!\d)")          # “t” followed by digits, not followed by another digit

# for exp_path in folder_path.iterdir():
#     name = exp_path.name

#     # basic filters
#     if not (all(s in name for s in exp_descriptions) and any(date in name for date in exp_dates)):
#         continue

#     # which task suite is this?
#     tokens = set(name.split('_'))     # {'octo', 'ckpt', '2025-07-14', '90', 't100000', ...}
#     suite  = next((ts for ts in task_suites if ts in tokens), None)
#     if suite is None:
#         continue

#     # pull the *actual* checkpoint id from the name
#     m = ckpt_re.search(name)
#     if not m:
#         continue
#     ckpt = int(m.group(1))
#     if ckpt not in checkpoints:                # skip any runs you’re not interested in
#         continue

#     # read the stats
#     stats_file = exp_path / "multi_run_statistics.json"
#     with stats_file.open(encoding="utf-8") as f:
#         data = json.load(f)

#     if "vlaps" in name:
#         vlaps_success_rates[suite][ckpt] = data['num_successes'] / data['num_runs']
#         vlaps_times[suite][ckpt] = [min(v, 600) for k,v in data['run_times'].items()]
#     else:
#         vla_success_rates[suite][ckpt] = data["num_successes"] / data["num_runs"]
#         vla_times[suite][ckpt] = [min(v, 600) for k,v in data['run_times'].items()]

# print(vla_success_rates)

suites_with_data = [s for s in task_suites if vla_success_rates[s] or vlaps_success_rates[s]]
if not suites_with_data:
    raise ValueError("success_rates is empty – nothing to plot.")

n = len(suites_with_data)
fig, axes = plt.subplots(nrows=n,
                         ncols=1,
                         figsize=(10, 3.2 * n),   # height grows with # of suites
                         sharex=True)

# If there's only one suite, axes is a single Axes object, so wrap in list
if n == 1:
    axes = [axes]

bar_width = 5000  # width of each bar
offset = bar_width // 2

for ax, suite in zip(axes, suites_with_data):
    vla_data = vla_success_rates[suite]
    vlaps_data = vlaps_success_rates[suite]

    # Get union of all ckpts to ensure consistent x-axis
    all_ckpts = sorted(set(vla_data.keys()) | set(vlaps_data.keys()))
    
    vla_rates   = [vla_data.get(ckpt, 0.0) for ckpt in all_ckpts]
    vlaps_rates = [vlaps_data.get(ckpt, 0.0) for ckpt in all_ckpts]

    # Shift bars left and right for side-by-side plotting
    vla_x = [ckpt - offset for ckpt in all_ckpts]
    vlaps_x = [ckpt + offset for ckpt in all_ckpts]

    ax.bar(vla_x, vla_rates, width=bar_width, color='blue', label='VLA')
    ax.bar(vlaps_x, vlaps_rates, width=bar_width, color='orange', label='VLAPS')

    ax.set_title(f"{suite} task suite")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

# Shared axis labels
axes[-1].set_xlabel("Checkpoint")
fig.text(0.04, 0.5, "Success rate", va="center", rotation="vertical")

fig.tight_layout(rect=[0.06, 0.05, 1, 0.97])
plt.savefig("success_rates_by_suite.png")

# ---- Runtime Plot ----
fig_runtime, axes_runtime = plt.subplots(nrows=n,
                                         ncols=1,
                                         figsize=(10, 3.2 * n),
                                         sharex=True)

if n == 1:
    axes_runtime = [axes_runtime]

for ax, suite in zip(axes_runtime, suites_with_data):
    vla_data = vla_times[suite]
    vlaps_data = vlaps_times[suite]

    all_ckpts = sorted(set(vla_data.keys()) | set(vlaps_data.keys()))

    vla_avgs = [sum(vla_data.get(ckpt, [])) / len(vla_data.get(ckpt, [1]))
                for ckpt in all_ckpts]
    vlaps_avgs = [sum(vlaps_data.get(ckpt, [])) / len(vlaps_data.get(ckpt, [1]))
                  for ckpt in all_ckpts]

    vla_medians = [
        np.median([min(v, 600) for v in vla_data.get(ckpt, [])]) if vla_data.get(ckpt) else 0
        for ckpt in all_ckpts
        ]
    vlaps_medians = [
        np.median([min(v, 600) for v in vlaps_data.get(ckpt, [])]) if vlaps_data.get(ckpt) else 0
        for ckpt in all_ckpts
    ]

    vla_x = [ckpt - offset for ckpt in all_ckpts]
    vlaps_x = [ckpt + offset for ckpt in all_ckpts]

    ax.bar(vla_x, vla_medians, width=bar_width, color='blue', label='VLA')
    ax.bar(vlaps_x, vlaps_medians, width=bar_width, color='orange', label='VLAPS')

    ax.set_title(f"{suite} task suite")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

axes_runtime[-1].set_xlabel("Checkpoint")
fig_runtime.text(0.04, 0.5, "Average runtime (seconds)", va="center", rotation="vertical")

fig_runtime.tight_layout(rect=[0.06, 0.05, 1, 0.97])
plt.savefig("average_runtimes_by_suite.png")

# --- Average success rate across task suites ---
overall_vla_success = {}
overall_vlaps_success = {}

for ckpt in checkpoints:
    vla_vals = [vla_success_rates[suite].get(ckpt, 0.0) for suite in suites_with_data]
    vlaps_vals = [vlaps_success_rates[suite].get(ckpt, 0.0) for suite in suites_with_data]
    overall_vla_success[ckpt] = sum(vla_vals) / len(vla_vals)
    overall_vlaps_success[ckpt] = sum(vlaps_vals) / len(vlaps_vals)

# --- Median runtime across task suites (capped) ---
overall_vla_times = {}
overall_vlaps_times = {}

for ckpt in checkpoints:
    vla_runtimes = []
    vlaps_runtimes = []
    for suite in suites_with_data:
        vla_runtimes.extend([min(v, 600) for v in vla_times[suite].get(ckpt, [])])
        vlaps_runtimes.extend([min(v, 600) for v in vlaps_times[suite].get(ckpt, [])])
    overall_vla_times[ckpt] = np.median(vla_runtimes) if vla_runtimes else 0
    overall_vlaps_times[ckpt] = np.median(vlaps_runtimes) if vlaps_runtimes else 0

# --- Plot overall success rates ---
fig, ax = plt.subplots(figsize=(10, 4))
all_ckpts = sorted(checkpoints)
vla_x = [ckpt - offset for ckpt in all_ckpts]
vlaps_x = [ckpt + offset for ckpt in all_ckpts]
vla_rates = [overall_vla_success[ckpt] for ckpt in all_ckpts]
vlaps_rates = [overall_vlaps_success[ckpt] for ckpt in all_ckpts]

ax.bar(vla_x, vla_rates, width=bar_width, color='blue', label='VLA')
ax.bar(vlaps_x, vlaps_rates, width=bar_width, color='orange', label='VLAPS')
ax.set_title("Average Success Rate Across Task Suites")
ax.set_xlabel("Checkpoint")
ax.set_ylabel("Success Rate")
ax.set_ylim(0, 1)
ax.grid(True, axis='y', alpha=0.3)
ax.legend()
fig.tight_layout()
plt.savefig("average_success_rate_overall.png")

# --- Plot overall median runtime ---
fig2, ax2 = plt.subplots(figsize=(10, 4))
vla_medians = [overall_vla_times[ckpt] for ckpt in all_ckpts]
vlaps_medians = [overall_vlaps_times[ckpt] for ckpt in all_ckpts]

ax2.bar(vla_x, vla_medians, width=bar_width, color='blue', label='VLA')
ax2.bar(vlaps_x, vlaps_medians, width=bar_width, color='orange', label='VLAPS')
ax2.set_title("Median Runtime Across Task Suites (Capped at 600s)")
ax2.set_xlabel("Checkpoint")
ax2.set_ylabel("Runtime (s)")
ax2.grid(True, axis='y', alpha=0.3)
ax2.legend()
fig2.tight_layout()
plt.savefig("average_runtime_overall.png")


# suites_with_data = [s for s, d in vla_success_rates.items() if d]
# if not suites_with_data:
#     raise ValueError("success_rates is empty – nothing to plot.")

# n = len(suites_with_data)
# fig, axes = plt.subplots(nrows=n,
#                          ncols=1,
#                          figsize=(10, 3.2 * n),   # height grows with # of suites
#                          sharex=True)

# # If there's only one suite, axes is a single Axes object, so wrap in list
# if n == 1:
#     axes = [axes]

# for ax, suite in zip(axes, suites_with_data):
#     data = vla_success_rates[suite]
#     ckpts, rates = zip(*sorted(data.items()))   # sort by checkpoint

#     ax.bar(ckpts, rates, width=5000)                        # or ax.plot(…, marker='o')
#     ax.set_title(f"{suite} task suite")
#     ax.set_ylim(0, 1)
#     ax.grid(True, axis="y", alpha=0.3)

# # Shared axis labels
# axes[-1].set_xlabel("Checkpoint")
# fig.text(0.04, 0.5, "Success rate", va="center", rotation="vertical")

# fig.tight_layout(rect=[0.06, 0.05, 1, 0.97])
# plt.savefig("success_rates_by_suite.png")

# import numpy as np

# fig, ax = plt.subplots(figsize=(20, 6))

# labels = []
# vla_vals = []
# vlaps_vals = []

# for suite in suites_with_data:
#     all_ckpts = sorted(set(vla_success_rates[suite].keys()) | set(vlaps_success_rates[suite].keys()))
#     for ckpt in all_ckpts:
#         labels.append(f"{suite}-{ckpt//1000}k")
#         vla_vals.append(vla_success_rates[suite].get(ckpt, 0.0))
#         vlaps_vals.append(vlaps_success_rates[suite].get(ckpt, 0.0))

# x = np.arange(len(labels))
# width = 0.35

# ax.bar(x - width/2, vla_vals, width, label='VLA', color='blue')
# ax.bar(x + width/2, vlaps_vals, width, label='VLAPS', color='orange')

# ax.set_ylabel('Success rate')
# ax.set_title('Success Rates Across Task Suites and Checkpoints')
# ax.set_xticks(x)
# ax.set_xticklabels(labels, rotation=45, ha='right')
# ax.legend()
# ax.grid(axis='y', alpha=0.3)

# fig.tight_layout()
# plt.savefig("all_success_rates_combined.png")