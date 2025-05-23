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
from utils.range_spectrum import construct_polar_space, compute_rangeFFT_BF


import pickle
import sys

# ——— Device CONFIG ———
fps             = 50          # plot update rate [Hz]
num_chirp_loops = 1           # chirps per Tx cycle
num_rx          = 4           # number of Rx antennas
num_tx          = 3           # number of Tx antennas (3 TX)
num_chirp_samples     = 512         # samples per chirp
fmin, fmax      = 5, 1000     # display band [Hz]
QUEUE_MAXSIZE   = 20          # pending frames
interval        = 300 / fps  # update interval [ms] smaller then 1000/fps will be good

# ——— User CONFIG ———
limit_meter     = 1           # range limit [m]
data_tofu       = 4          # decide to loss how many frame for streaming stablization
save_recording  = False      # save frames to file
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

if __name__ == "__main__":
    
    # Start the writer thread
    t = threading.Thread(target=frameQ_update_thread, daemon=True)
    t.start()
    
    listener = PacketListenerThread(daemon=True)
    range_limit = int(limit_meter/0.0422) # 5m

    x_axis, y_axis = construct_polar_space()
    # init
    Z0 = np.zeros((x_axis.shape[0], range_limit))

    fig, ax = plt.subplots(figsize=(14, 8))
    mesh = ax.pcolormesh(
        y_axis[:, :range_limit],
        x_axis[:, :range_limit],
        Z0,
    )
    cbar = fig.colorbar(mesh)
    ax.set_xlabel('X-Axis (meters)')
    ax.set_ylabel('Y-Axis (meters)')
    ax.set_ylim([0, limit_meter])
    mesh.set_clim(0, 350000)

    def init():
        mesh.set_array(Z0.ravel())
        return mesh,

    def update(_):
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            return (mesh,)
        
        if save_recording:
            realtime_frames.append(frame)

        fft_data = compute_rangeFFT_BF(frame)
        Z = fft_data[:, :range_limit]
        mesh.set_array(Z.ravel())
        # vmin, vmax = Z.min(), Z.max()
        # mesh.set_clim(vmin, vmax)
        return mesh,

    try:
        listener.start()
        ani = FuncAnimation(
            fig,
            update,
            init_func=init,
            interval=interval,
            blit=False
        )
        plt.show()
        # while True:
        #     print("Waiting for frames...")
        #     # print frame queue size
        #     print("Frame queue size: ", frame_queue.qsize())
        #     print("Write queue size: ", write_queue.qsize())
        #     time.sleep(1)
        #     for i in range(fps):
        #         try:
        #             if i == fps-1:
        #                 print("Getting frame...")
        #                 frame = frame_queue.get(timeout=1)
        #                 print("Frame shape: ", frame.shape)
        #         except Exception as e:
        #             continue
    

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
