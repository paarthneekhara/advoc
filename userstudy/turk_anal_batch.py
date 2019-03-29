from collections import defaultdict
import csv
import sys
import numpy as np

ANSWER_TO_INT = {
    'Bad - Completely unnatural speech': 1,
    'Poor - Mostly unnatural speech': 2,
    'Fair - Equally natural and unnatural speech': 3,
    'Good - Mostly natural speech': 4,
    'Excellent - Completely natural speech': 5
}

key_fp, results_fp = sys.argv[1:]

# Create key datastructures
with open(key_fp, 'r') as f:
  key = [l.split(',') for l in f.read().strip().splitlines()]
uuid_to_method = {l[0]:l[1] for l in key}
method_to_uuids = defaultdict(list)
for uuid, method in uuid_to_method.items():
  method_to_uuids[method].append(uuid)

# Load results
uuid_to_ratings = defaultdict(list)
worker_to_avg_seconds = {}
with open(results_fp, 'r') as f:
  reader = csv.DictReader(f)
  for row in reader:
    uuid = row['Input.audio_url'].split('/')[-1].split('.')[0]
    answer = row['Answer.audio-naturalness.label']
    uuid_to_ratings[uuid].append(ANSWER_TO_INT[answer])

    worker_id = row['WorkerId']
    worker_approval_rate = row['LifetimeApprovalRate']
    worker_seconds = row['WorkTimeInSeconds']

    print(worker_id, worker_seconds, worker_approval_rate)

# Print n, scores, 95% confidence intervals
for method, uuids in method_to_uuids.items():
  scores = []
  for uuid in uuids:
    if uuid in uuid_to_ratings:
      scores.extend(uuid_to_ratings[uuid])

  mean = np.mean(scores)
  std = np.std(scores)
  print(method)
  print(len(scores), mean, 1.96 * std)
