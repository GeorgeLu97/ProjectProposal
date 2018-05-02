# Turns [[1, 2], [3, 4], [5, 6], [7, 8]] into [[1, 3, 5, 7], [2, 4, 6, 8]]

def rotate_stats(y):
  results_arr = [[] for _ in range(len(y[0]))]

  for entry in y:
    for i in range(len(results_arr)):
      results_arr[i].append(entry[i])

  return results_arr