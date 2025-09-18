import hdf5storage
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from datetime import datetime, timedelta
import os
import pandas as pd
import hdf5storage
from matplotlib.colors import LinearSegmentedColormap

# this code is transferred from matlab
# it is to draw spectrograms form the .mat datafiles
# with aligned time with 4 diff operations
# we should run the time_align.py first to gain the time_align info!

# define color map
def truncate_colormap(cmap_name='jet', minval=0.0, maxval=0.75, n=256):
    cmap = plt.get_cmap(cmap_name)
    new_cmap = LinearSegmentedColormap.from_list(
        f'{cmap_name}_trunc',
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap

def drawspectrum_py(starttime, endtime, tt, ifline=False, extralines=None,
                    time_align=None, arrow_timestamps=None, freq_lines=None,
                    folder="/Users/xluan3/Desktop/Projects/layer1to5/spectrograms",
                    save_folder="/Users/xluan3/Desktop/Projects/layer1to5/result"):

    tt1_candidates = [t for t in tt if t > (starttime - timedelta(seconds=2))]
    tt2_candidates = [t for t in tt if t < endtime]
    if not tt1_candidates or not tt2_candidates:
        print("No valid timestamps in window.")
        return

    tt1 = min(tt1_candidates)
    tt2 = max(tt2_candidates)
    if tt2 < tt1:
        return

    first = tt.index(tt1)
    last = tt.index(tt2)

    startfill = (tt1 - starttime).total_seconds()
    endfill = (endtime - timedelta(seconds=2) - tt2).total_seconds()

    loc = []
    y = np.empty((200, 0))
    tprev = tt1

    for ii in range(first, last + 1):
        mat_path = os.path.join(folder, f"{ii + 1}.mat")
        if not os.path.exists(mat_path):
            print(f"Missing file: {mat_path}")
            continue

        x = loadmat(mat_path)
        if 's_' not in x:
            print(f"'s_' not found in {mat_path}")
            continue

        a = np.abs(x['s_'][:, :, 0])  # shape: 200 × 98

        loc.append(1 + 49 * (tt[ii] - starttime).total_seconds())

        if ii == first:
            if startfill < 0:
                start_idx = int(1 - 49 * startfill)
                y = np.hstack([y, a[:, start_idx:98]])
            else:
                y = np.hstack([y, np.full((200, int(49 * startfill)), np.nan), a[:, 0:98]])
        elif ii != last:
            fill = max((tt[ii] - tprev).total_seconds() - 2, 0)
            if (tt[ii] - tprev).total_seconds() < 2:
                print(f"Warning: timestamps {tt[ii]} and {tprev} are < 2 seconds apart")
            y = np.hstack([y, np.full((200, int(49 * fill)), np.nan), a[:, 0:98]])
        else:
            fill = max((tt[ii] - tprev).total_seconds() - 2, 0)
            if (tt[ii] - tprev).total_seconds() < 2:
                print(f"Warning: timestamps {tt[ii]} and {tprev} are < 2 seconds apart")
            if endfill < 0:
                end_idx = int(49 * (2 + endfill))
                y = np.hstack([y, np.full((200, int(49 * fill)), np.nan), a[:, 0:end_idx]])
            else:
                y = np.hstack([
                    y,
                    np.full((200, int(49 * fill)), np.nan),
                    a[:, 0:98],
                    np.full((200, int(49 * endfill)), np.nan)
                ])

        tprev = tt[ii]

    # Log transform
    z = np.log(y)

    # custom colormap
    cmap = truncate_colormap('jet', 0.0, 0.75)
    cmap.set_bad(color='darkblue')

    # Plot only the spectrogram
    fig, ax = plt.subplots(figsize=(25, 6))  # wider and shorter

    im = ax.imshow(z, aspect='auto',
                   vmin=np.nanpercentile(z, 1),
                   vmax=np.nanpercentile(z, 99),
                   origin='lower',
                   cmap=cmap)
    ax.set_xticks(loc)
    ax.set_xticklabels([t.strftime('%H:%M:%S') for t in tt[first:last+1]], rotation=90, fontsize=6)

    # y label
    yticks = np.arange(0, 200, 20)
    ytick_labels = [f"{0.625 + i * 1.25:.1f}" for i in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    ax.set_ylabel("Frequency (kHz)")

    # fig.suptitle(f"{starttime.strftime('%H:%M:%S')} - {endtime.strftime('%H:%M:%S')}",
    #              fontsize=14, y=1.2)  # Move title up
    ax.set_title(f"{starttime.strftime('%H:%M:%S')} - {endtime.strftime('%H:%M:%S')}", pad=20)
    fig.colorbar(im, ax=ax)

    # if ifline:
    #     for xloc in loc:
    #         ax.axvline(x=xloc, color='black', linewidth=0.8)

    if time_align is not None:
        # Get only the marks visible in this window
        visible_marks = [t for t in time_align if starttime <= t <= endtime]

        for local_i, tmark in enumerate(visible_marks):
            percentage = (tmark - starttime).total_seconds() / (endtime - starttime).total_seconds()
            lineloc = round(y.shape[1] * percentage)

            if 0 < lineloc < y.shape[1]:
                # Every 4th line is solid, others dashed
                if (local_i + 2) % 4 == 0:
                    ax.axvline(x=lineloc, color='white', linestyle='-', linewidth=1.5)
                else:
                    ax.axvline(x=lineloc, color='white', linestyle='--', linewidth=1)

    # Draw vertical arrows from arrow_timestamps (smaller arrows above spectrum)
    if arrow_timestamps is not None:
        visible_arrows = [t for t in arrow_timestamps if starttime <= t <= endtime]

        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        ax2.axis('off')

        for tmark in visible_arrows:
            percentage = (tmark - starttime).total_seconds() / (endtime - starttime).total_seconds()
            lineloc = round(y.shape[1] * percentage)
            if 0 < lineloc < y.shape[1]:
                ax2.annotate('', xy=(lineloc, 1.0), xytext=(lineloc, 1.05),
                             xycoords=('data', 'axes fraction'),
                             textcoords=('data', 'axes fraction'),
                             arrowprops=dict(facecolor='darkblue', edgecolor='darkblue',
                                             arrowstyle='-|>', lw=1.2))

    # Horizontal dashed lines for specific frequencies
    if freq_lines:
        for freq in freq_lines:
            y_index = (freq - 0.625) / 1.25  # interpolate exact float index
            # y_index = freq
            if 0 <= y_index <= 200:
                ax.axhline(y=y_index, color='red', linestyle='--', linewidth=0.8)
                ax.text(2, y_index + 1, f'{freq:.1f} kHz', color='red', fontsize=8)

    plt.tight_layout()

    # Save the figure
    os.makedirs(save_folder, exist_ok=True)
    fname = f"spectrum_{starttime.strftime('%H-%M-%S')}_{endtime.strftime('%H-%M-%S')}.png"
    figpath = os.path.join(save_folder, fname)
    fig.savefig(figpath, dpi=300)
    print(f"Figure saved to: {figpath}")

    return fig, y

## usage
# layer 1 to 5:
# Load timestamp CSV
# Load tt and time alignment data
tt_df = pd.read_csv("/Users/xluan3/Desktop/Projects/AM data_acoustic/Codes/tt_export.csv")
tt_list = pd.to_datetime(tt_df['Time']).tolist()
# arrow timestamps
arrow_timestamps = pd.read_csv("/Users/xluan3/Desktop/Projects/layer1to5/result/selected_timestamps.csv")
arrow_timestamps = pd.to_datetime(arrow_timestamps['time']).tolist()

# Plot loop
global_start = pd.Timestamp("2021-06-11 19:24:02")
global_end = pd.Timestamp("2021-06-11 19:44:01")
window_size = timedelta(minutes=4)

current_start = global_start
while current_start < global_end:
    current_end = min(current_start + window_size, global_end)
    print(f"Drawing spectrogram from {current_start} to {current_end}")
    try:
        drawspectrum_py(current_start, current_end, tt_list,
                        ifline=True,
                        time_align=time_align,
                        arrow_timestamps=arrow_timestamps,
                        freq_lines=[76.875, 118.125, 143.125, 174.375, 195.625])
    except Exception as e:
        print(f"Failed for window {current_start} to {current_end}: {e}")
    current_start += window_size

# # layer 1 to 20:
# # Load timestamp CSV
# # Load tt and time alignment data
# tt_df = pd.read_csv("/Users/xluan3/Desktop/Projects/AM data_acoustic/Codes/tt_export.csv")
# tt_list = pd.to_datetime(tt_df['Time']).tolist()
# # arrow timestamps
# arrow_timestamps = pd.read_csv("/Users/xluan3/Desktop/Projects/layer1to20/selected_timestamps.csv")
# arrow_timestamps = pd.to_datetime(arrow_timestamps['time']).tolist()
#
# # Plot loop
# global_start = pd.Timestamp("2021-06-11 19:30:00")
# global_end = pd.Timestamp("2021-06-11 20:11:59")
# window_size = timedelta(minutes=4)
#
# current_start = global_start
# while current_start < global_end:
#     current_end = min(current_start + window_size, global_end)
#     print(f"Drawing spectrogram from {current_start} to {current_end}")
#     try:
#         drawspectrum_py(current_start, current_end, tt_list,
#                         ifline=True,
#                         time_align=time_align,
#                         arrow_timestamps=arrow_timestamps,
#                         freq_lines=[86,96,190])
#     except Exception as e:
#         print(f"Failed for window {current_start} to {current_end}: {e}")
#     current_start += window_size

