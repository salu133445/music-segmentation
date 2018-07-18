import json
import msaf
import numpy as np
import os
import os.path as op
import pretty_midi
import sys
import string
from pypianoroll import Multitrack, Track

lpd_root = "/home/herman/lpd/lpd/lpd_cleansed/"
id_path = "/home/herman/lpd/cleansed_ids.txt"
id_lists_root = "/home/herman/lpd/id_lists"
synth_root = "/home/bgenchel/PlanNet/lpd_synthesized"

MIN_SEGMENT_LEN = 3  # seconds

with open(id_path) as f:
    id_dict = {line.split()[1]: line.split()[0] for line in f}
    synth_id_dict = {k:v for k,v in id_dict.items() if "TRA" in k[:3]}

def get_npz_path(msd_id):
    return os.path.join(lpd_root, msd_id[2], msd_id[3], msd_id[4], msd_id,
            id_dict[msd_id] + '.npz')

def get_synthesized_path(msd_id):
    return op.join(synth_root, "songs",  msd_id[2], msd_id[3], msd_id[4], msd_id, 
            id_dict[msd_id] + '.mp3')

def get_segmentation_path(msd_id):
    return op.join(synth_root, "segmentation",  msd_id[2], msd_id[3], msd_id[4], 
            msd_id, synth_id_dict[msd_id] + '.json')

def segment_song(msd_id):
    boundaries, labels = msaf.process(get_synthesized_path(msd_id),
        boundaries_id="olda", labels_id="scluster", feature="mfcc")

    # merge short segments
    new_boundaries = [boundaries[0]]
    new_labels = [labels[0]]
    for i in range(1, len(boundaries)):
        if (boundaries[i] - boundaries[i-1]) > MIN_SEGMENT_LEN:
            new_boundaries.append(boundaries[i])
            new_labels.append(labels[i])

    boundaries = new_boundaries
    labels = new_labels

    # calculate tick values for segments
    mt_midi = Multitrack(get_npz_path(msd_id))
    tempo = mt_midi.tempo[0]
    beat_resolution = mt_midi.beat_resolution
    # beats/min * ticks/beat * min/sec = ticks/sec
    ticks_per_second = (tempo*mt_midi.beat_resolution) // 60
    
    def get_nearest(num, nearest="downbeat"):
        if nearest == "beat":
            multiple = beat_resolution
        elif nearest == "downbeat":
            multiple = beat_resolution * 4
        else:
            raise ValueError("Argument to get_nearest should be either 'beat' or 'downbeat'")
    
        factor = num // multiple
        remainder = num % multiple
        if remainder < (multiple // 2):
            ndb = multiple*factor
        else:
            ndb = multiple*(factor + 1)
        return ndb

    prev_l = -1
    tick_boundaries = []
    for b, l in zip(boundaries, labels):
        if l != prev_l:
            tick_boundaries.append(get_nearest(b*ticks_per_second, "downbeat"))
        else:
            tick_boundaries.append(get_nearest(b*ticks_per_second, "beat"))

    second_boundaries = [tb/ticks_per_second for tb in tick_boundaries]
            
    # change labels to letters and order
    letter_labels = []
    label_map = dict()
    for label in labels:
        if label not in label_map:
            label_map[label] = string.ascii_uppercase[len(label_map)]
        letter_labels.append(label_map[label])
    
    return second_boundaries, tick_boundaries, letter_labels

for i, msd_id in enumerate(synth_id_dict.keys()):
    print("processing file %d/%d, %.2f%% complete." % (i, len(synth_id_dict.keys()), 
        i / len(synth_id_dict.keys())), end="\r")

    segpath = get_segmentation_path(msd_id)
    if not op.exists(op.dirname(segpath)):
        os.makedirs(op.dirname(segpath))
    else:
        continue

    second_boundaries, tick_boundaries, letter_labels = segment_song(msd_id)
    obj = [{'seconds': sb, 'ticks': tb, 'label': l} for sb, tb, l in
            zip(second_boundaries, tick_boundaries, letter_labels)]

    with open(segpath, 'w') as fp:
        json.dump(obj, fp, indent=4)

    sys.stdout.flush()
