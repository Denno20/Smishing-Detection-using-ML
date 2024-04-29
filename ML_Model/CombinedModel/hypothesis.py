#https://www.geeksforgeeks.org/understanding-hypothesis-testing/
import numpy as np
from scipy import stats


# Data

with open("before_output.txt", "r") as f:
	data1 = f.readlines()

with open("after_output.txt", "r") as f:
	data2 = f.readlines()

arr1 = [int(entry.strip("\n")) for entry in data1]
arr2 = [int(entry.strip("\n")) for entry in data2]

public_dataset = np.array(arr1)
combined_dataset = np.array(arr2)




# Step 1: Null and Alternate Hypotheses
null_hypothesis = "The new dataset has no effect on the accuracy of the machine learning model"
alternate_hypothesis = "The new dataset has an effect on the accuracy of the machine learning model"

# Step 2: Significance Level
alpha = 0.05
# Step 3: Paired T-test
t_statistic, p_value = stats.ttest_rel(combined_dataset, public_dataset)
# Step 4: Calculate T-statistic manually
m = np.mean(combined_dataset - public_dataset)
s = np.std(combined_dataset - public_dataset, ddof=1) # using ddof=1 for sample standard deviation
n = len(public_dataset)
t_statistic_manual = m / (s / np.sqrt(n))


print(p_value)
print(p_value <= alpha)

# Step 5: Decision
if p_value <= alpha:
	decision = "Reject"
else:
	decision = "Fail to reject"

# Conclusion
if decision == "Reject":
	conclusion = "The training dataset that contains the AI generated smishing texts has a significant impact on the model performance."
else:
	conclusion = "The training dataset that contains the AI generated smishing texts has no impact on the model performance."

# Display results
print("T-statistic (from scipy):", t_statistic)
print("P-value (from scipy):", p_value)
print("T-statistic (calculated manually):", t_statistic_manual)
print(f"Decision: {decision} the null hypothesis at alpha={alpha}.")
print("Conclusion:", conclusion)
