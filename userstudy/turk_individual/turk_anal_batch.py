from collections import defaultdict, Counter
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
uuid_to_wids_ratings = defaultdict(list)
wid_to_count = Counter()
wid_to_avg_seconds = {}
with open(results_fp, 'r') as f:
  reader = csv.DictReader(f)
  for row in reader:
    wid = row['WorkerId']
    wid_to_count[wid] += 1
    wid_approval_rate = row['LifetimeApprovalRate']
    wid_seconds = row['WorkTimeInSeconds']

    uuid = row['Input.audio_url'].split('/')[-1].split('.')[0]
    answer = row['Answer.audio-naturalness.label']
    uuid_to_wids_ratings[uuid].append((wid, ANSWER_TO_INT[answer]))

print(wid_to_count)

# Print n, scores, 95% confidence intervals
for method, uuids in method_to_uuids.items():
  scores = []
  for uuid in uuids:
    if uuid in uuid_to_wids_ratings:
      wids_ratings = uuid_to_wids_ratings[uuid]
      for wid, rating in wids_ratings:
        scores.append(rating)

  mean = np.mean(scores)
  std = np.std(scores)
  count = len(scores)
  print(method)
  print(len(scores), mean, 1.96 * std / np.sqrt(float(count)))
