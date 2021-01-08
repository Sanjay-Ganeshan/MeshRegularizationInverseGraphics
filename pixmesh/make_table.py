import json
import os
from os.path import isdir

mydir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(mydir, "..", "data", "output"))

def read_metrics(research_question, group_name, exp_id):
    path_to_metrics = os.path.join(OUTPUT_DIR, f"{research_question}_{group_name}_{exp_id}", "metrics.json")
    try:
        with open(path_to_metrics) as f:
            return json.load(f)
    except:
        return None

def get_experiment_groups():
    poss_dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    all_groups = {} # rq: [(rq, group, exp_id) ... ]
    for exp_name in poss_dirs:
        try:
            rq, group, exp_id = exp_name.split('_')
            exp_id = int(exp_id)
            if rq not in all_groups:
                all_groups[rq] = []
            all_groups[rq].append((rq, group, exp_id))

        except:
            print("Skipping", exp_name)
    
    return all_groups

def get_line(rq, group, exp_id, line_template):
    metrics = read_metrics(rq, group, exp_id)
    if metrics is None:
        return None
    else:
        n_iters = metrics['niters']
        vox_iou = metrics['predicted']['iou']
        im_loss = metrics['predicted']['imageloss']

        line_formatted = line_template % (f"{rq}-{group}-{exp_id}", vox_iou, im_loss*1000, n_iters)
        return line_formatted

def main():
    with open('table_skel.txt', 'r') as inf:
        line_template = inf.readline()
        skel = inf.read()
        
    # We need to give skel 3 things,
    # a string with all the lines in the table
    # a string with the name of the experiment group
    # a string with the reference name

    exp_groups = get_experiment_groups()

    tables = []

    for each_rq in sorted(exp_groups.keys()):
        if each_rq == 'TST' or each_rq == 'GN':
            continue
        lines = []
        for (rq, group, exp_id) in sorted(exp_groups[each_rq]):
            lines.append(get_line(rq, group, exp_id, line_template))
        line_content = '\n'.join([l for l in lines if l is not None])
        exp_group_name = f"{each_rq}"
        ref_name = exp_group_name
        table_code = skel % (line_content, exp_group_name, ref_name)
        tables.append(table_code)
    
            
    with open('tables.txt', 'w') as outf:
        outf.write("\n\n\n".join(tables))
        outf.write("\n")

if __name__ == "__main__":
    main()

