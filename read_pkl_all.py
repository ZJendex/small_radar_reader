#!/usr/bin/env python3

import os
import sys
import pickle
from scipy.io import savemat
import organizer_copy as org

def process_file(file_path):
    """
    Load a pickled session, organize frames, and save to .mat and new .pkl.
    """
    base, ext = os.path.splitext(file_path)
    if ext.lower() != '.pkl':
        print(f"  • Skipping '{file_path}' (not a .pkl file).")
        return

    # Load the original pickle
    with open(file_path, 'rb') as f:
        session = pickle.load(f)
    print(f"Processing '{os.path.basename(file_path)}'... ", end='')

    # Organize the frames
    organizer = org.Organizer(session, 1, 4, 3, 512)
    frames = organizer.organize()
    print(f"done (frames.shape = {frames.shape}).")

    # Prepare output directories
    parent_dir = os.path.dirname(base)
    mat_dir = os.path.join(parent_dir, 'mat_data')
    pkl_dir = os.path.join(parent_dir, 'pkl_data')
    os.makedirs(mat_dir, exist_ok=True)
    os.makedirs(pkl_dir, exist_ok=True)

    # Base filename without extension
    name = os.path.basename(base)

    # Save MATLAB .mat
    mat_out = os.path.join(mat_dir, f"{name}.mat")
    savemat(mat_out, {
        'frames':      frames,
        'start_time':  session[3],
        'end_time':    session[4]
    })

    # Save new pickle
    to_save = {
        'frames':      frames,
        'start_time':  session[3],
        'end_time':    session[4],
        'num_frames':  len(frames)
    }
    pkl_out = os.path.join(pkl_dir, f"{name}_read.pkl")
    with open(pkl_out, 'wb') as f:
        pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

def main():
    if len(sys.argv) != 2:
        print("Usage: python read_pkl_all.py <folder_path>")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a valid directory.")
        sys.exit(1)

    # Iterate through all files in the folder
    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if os.path.isfile(path):
            process_file(path)

if __name__ == "__main__":
    main()
    print("All files processed.")