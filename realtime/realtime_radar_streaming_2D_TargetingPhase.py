#!/usr/bin/env python3
# realtime_radar_threaded_async.py

import threading
import queue
import time
import struct
import os
import numpy as np
from listener import UDPListener
import organizer_realtime as org

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from utils.range_spectrum import compute_range_spectrum_onChirp
from utils.range_spectrum import construct_polar_space, compute_rangeFFT_BF, get_phase
from collections import deque



import pickle
import sys
import config

# ——— Device CONFIG ———
fps             = config.fps          # Radar frame rate [Hz]
num_chirp_loops = config.num_chirp_loops           # chirps per Tx cycle
num_rx          = config.num_rx           # number of Rx antennas
num_tx          = config.num_tx           # number of Tx antennas (3 TX)
num_chirp_samples     = config.num_chirp_samples         # samples per chirp
fmin, fmax      = 5, 1000     # display band [Hz]
QUEUE_MAXSIZE   = 20          # pending frames
interval        = 300 / fps  # update interval [ms] smaller then 1000/fps will be good

# ——— User CONFIG ———
limit_meter     = 1           # range limit [m]
data_tofu       = 8          # decide to loss how many frame for streaming stablization [4 is the sweet spot - not recommand to change]
save_recording  = False      # save frames to file

# ——— Real fps Compute CONFIG ———
last_time = time.time()
frame_count = 0
real_fps = 0

# ——— Realtime CONFIG ———
position = [0, 0]  # Initial position
# position limit
position_x_limit = [-config.AZIM_FFT//2, config.AZIM_FFT//2]
position_y_limit = [0, config.RANGE_FFT]
# ————————————

# Precompute packet/frame sizes
dummy = ([], [], [], "", "")
org_dummy = org.Organizer(dummy, num_chirp_loops, num_rx, num_tx, num_chirp_samples)
PACKETS_PER_FRAME = org_dummy.NUM_PACKETS_PER_FRAME
BYTES_PER_FRAME   = org_dummy.BYTES_IN_FRAME

print("PACKETS_PER_FRAME: ", PACKETS_PER_FRAME)
print("BYTES_PER_FRAME: ", BYTES_PER_FRAME)

# Thread communication
frame_queue  = queue.Queue()
write_queue = queue.Queue()
stop_evt = threading.Event()

realtime_frames = []

def frameQ_update_thread():
    cache_data = []
    cache_bc = []
    cache_pkt = []
    while not stop_evt.is_set():
        try:
            item = write_queue.get(timeout=1)
            if item is None: 
                continue

            pkt_num, bc, data = item
            cache_data.append(data)
            cache_bc.append(bc)
            cache_pkt.append(pkt_num)

            if len(cache_data) >= data_tofu*PACKETS_PER_FRAME:
                o = org.Organizer((cache_data, cache_pkt, cache_bc), num_chirp_loops, num_rx, num_tx, num_chirp_samples)
                frame = o.organize()
                if frame is None:
                    print("Frame is None")
                    continue
                frame = np.squeeze(frame[1])
                frame_queue.put(frame)
                cache_data = []
                cache_bc = []
                cache_pkt = []
        except queue.Empty:
            continue
        except IndexError as e:
            print("IndexError: ", e)
            continue


class PacketListenerThread(threading.Thread):
    """
    Reads UDP continuously, assembles full frames, pushes frames to frame_queue.
    """
    def run(self):
        listener = UDPListener(fileroot="noop")
        listener.data_socket.settimeout(1)

        try:
            while not stop_evt.is_set():
                pkt_num, bc, data = listener._read_data_packet()
                write_queue.put((pkt_num, bc, data))
                

        except Exception as e:
            return       



def on_key_press(event):
    global position
    key = event.key

    if key == 'up':
        if position[1] < position_y_limit[1]:
            position[1] += 1
        else:
            position[1] = position_y_limit[1]
    elif key == 'down':
        if position[1] > position_y_limit[0]:
            position[1] -= 1
        else:
            position[1] = position_y_limit[0]
    elif key == 'left':
        if position[0] > position_x_limit[0]:
            position[0] -= 1
        else:
            position[0] = position_x_limit[0]
    elif key == 'right':
        if position[0] < position_x_limit[1]:
            position[0] += 1
        else:
            position[0] = position_x_limit[1]
    else:
        return  

if __name__ == "__main__":
    
    # Start the writer thread
    t = threading.Thread(target=frameQ_update_thread, daemon=True)
    t.start()
    
    listener = PacketListenerThread(daemon=True)

    # Set up the plot
    az_idx, rng_idx = position[0], position[1]
    fps_fix = config.fps / data_tofu
    window_sec = 3
    max_len = int(fps_fix * window_sec)
    displacement_history = deque(maxlen=max_len)
    time_history = deque(maxlen=max_len)
    phase_history = []

    range_limit = int(limit_meter/0.0422) # 5m
    x_axis, y_axis = construct_polar_space()
    Z0 = np.zeros((x_axis.shape[0], range_limit))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
    # subplot 1：2D radar plot + red dot targeting
    mesh = ax1.pcolormesh(y_axis[:, :range_limit], x_axis[:, :range_limit], Z0)
    dot, = ax1.plot([], [], 'ro', markersize=8)
    fps_text = ax1.text(0.01, 0.99, '', transform=ax1.transAxes, va='top')
    ax1.set_xlabel('X-Axis (m)')
    ax1.set_ylabel('Y-Axis (m)')
    ax1.set_ylim([0, limit_meter])
    mesh.set_clim(0, 350000)

    # subplot 2：phase at red dot
    line, = ax2.plot([], [], '-')
    ax2.set_xlim(0, window_sec)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Displacement (mm)')
    ax2.set_title(f'Displacement at Bin (az:{position[0]}, rng:{position[1]})')
    ax2.grid(True)

    def init():
        mesh.set_array(Z0.ravel())
        line.set_data([], [])
        dot.set_data([], [])
        return mesh, dot, line

    def update(_):
        global last_time, frame_count, real_fps, position, az_idx, rng_idx, phase_history
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            return mesh, dot, line
        
        if save_recording:
            realtime_frames.append(frame)

        if frame_queue.qsize() % 100 == 99:
            print("Frame queue size OVER: ", frame_queue.qsize())

        frame_count += 1
        current_time = time.time()
        # Update the frame rate every second
        if current_time - last_time >= 1.0:
            real_fps = frame_count / (current_time - last_time)
            fps_text.set_text(f'FPS: {real_fps:.1f}')
            last_time = current_time
            frame_count = 0

        # --- 2D fft ---
        fft_data = compute_rangeFFT_BF(frame)
        # --- update pcolormesh ---
        Spec_2D = np.abs(fft_data)
        Z = Spec_2D[:, :range_limit]
        mesh.set_array(Z.ravel())
        # Update the position of the dot
        x_dot = x_axis[position[0],position[1]]
        y_dot = y_axis[position[0],position[1]]
        if position[0] < 0:
                y_dot = -y_dot
                x_dot = -x_dot
        dot.set_data([x_dot],
             [y_dot])
        # vmin, vmax = Z.min(), Z.max()
        # mesh.set_clim(vmin, vmax)

        # --- update phase plot ---
        if az_idx != position[0] or rng_idx != position[1]:
            displacement_history.clear()
            time_history.clear()
            az_idx, rng_idx = position[0], position[1]
        disp = get_phase(fft_data[position[0], position[1]])
        t = current_time
        time_history.append(t)
        displacement_history.append(disp)
        t0 = time_history[0]
        time_shifted = [ti - t0 for ti in time_history]
        # unwrap and detrend the phase when have enough data
        if len(time_shifted) > max_len/2:
            phase_history = np.unwrap(displacement_history).tolist()
        else:
            phase_history = displacement_history
        # update the line plot
        line.set_data(time_shifted, phase_history)
        ax2.set_xlim(0, window_sec)
        ax2.relim()            # recalculate data limits
        ax2.autoscale_view()
        ax2.set_title(f'Displacement at Bin (az:{position[0]}, rng:{position[1]})')
        return mesh, dot, line

    try:
        listener.start()
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        ani = FuncAnimation(
            fig,
            update,
            init_func=init,
            interval=interval,
            blit=False
        )
        plt.show()

    except Exception as e:
        t.join()
        listener.join()
        print("Stopped listening for packets.")
    except KeyboardInterrupt:
        print("Keyboard interrupt detected.")
        if save_recording:
            print("Saving frames...")
            with open('realtime_frames.pkl', 'wb') as f:
                pickle.dump(realtime_frames, f, protocol=pickle.HIGHEST_PROTOCOL)

        stop_evt.set()             
        write_queue.put(None)    
        if listener.is_alive(): 
            listener.join(timeout=2)   
        t.join(timeout=2)
        print("Stopped all threads cleanly.")
