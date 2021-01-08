import torch
import numpy as np
import os
import uuid
import glob
from pathlib import Path



def load_obj(file):
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert len(face_vertex_ids) == 3
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


def export(file, vs, faces, vn=None, color=None):
    with open(file, 'w+') as f:
        for vi, v in enumerate(vs):
            if color is None:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            else:
                f.write("v %f %f %f %f %f %f\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            if vn is not None:
                f.write("vn %f %f %f\n" % (vn[vi, 0], vn[vi, 1], vn[vi, 2]))
        for face in faces:
            f.write("f %d %d %d\n" % (face[0] + 1, face[1] + 1, face[2] + 1))


def random_file_name(ext, prefix='temp'):
    return f'{prefix}{uuid.uuid4()}.{ext}'
