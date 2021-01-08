import subprocess
import re
import time

global TIME_DATA
try:
    x = TIME_DATA
except:
    TIME_DATA = {'active': {}, 'measurements': {}}

def get_vram_usage():
    '''
    Gets current GPU Memory usage, in mb
    '''
    smi_output = subprocess.run("nvidia-smi", stdout=subprocess.PIPE).stdout.decode('utf-8')
    important_ix = smi_output.index('MiB')
    good_output = smi_output[important_ix-20:important_ix+30]
    spl = good_output.split('|')
    important = None
    for each_seg in spl:
        if 'MiB' in each_seg:
            important = each_seg
            break
    curr_str, max_str = important.replace("MiB","").split('/')
    curr_i = int(curr_str)
    max_i = int(max_str)
    return curr_i, max_i

def timer(name):
    if name is None:
        raise ValueError("Cannot use a timer name of None")
    t = time.time()
    if name in TIME_DATA['active']:
        elapsed = t - TIME_DATA['active'][name]
        if name not in TIME_DATA['measurements']:
            TIME_DATA['measurements'][name] = []
        TIME_DATA['measurements'][name].append(elapsed)
        # Clear so we can time again
        del TIME_DATA['active'][name]
    else:
        TIME_DATA['active'][name] = t

def avgtime(name = None):
    if name is None:
        keys = sorted(TIME_DATA['measurements'].keys())
        # Though it shouldn't happen, avoid infinite loops
        return {k: avgtime(k) for k in keys if k is not None}
    else:
        msr = TIME_DATA['measurements'].get(name, [0])
        return (len(msr), sum(msr) / len(msr))

