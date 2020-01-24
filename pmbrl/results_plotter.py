import json

results_path = "log_sparse_cartpole_exploration"
results_dict = json.load("results_path")
print(type(results_dict))
print(len(results_dict))
print(results_dict)
