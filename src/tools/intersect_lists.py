"""Perform boolean operations on a pair of lists from datasets/celeba/lists."""

import argparse
import os.path as op

def read_input_list(path):
  assert op.exists(path), path
  with open(path) as f:
    content = f.read().splitlines()
  assert len(content) > 0, 'list should not be empty'
  return content

def logical_and(content1, content2):
  return list(set(content1) & set(content2))

parser = argparse.ArgumentParser()
parser.add_argument('--list1', required=True)
parser.add_argument('--list2', required=True)
parser.add_argument('--list_out', required=False, default='/dev/null')
args = parser.parse_args()

content1 = read_input_list(args.list1)
content2 = read_input_list(args.list2)
content_out = logical_and(content1, content2)
print ('%d intersect %d lines = %d lines' %
        (len(content1), len(content2), len(content_out)))
with open(args.list_out, 'w') as f:
  f.write("\n".join(content_out))
