import matplotlib.pyplot as plt

data_file = 'test.data'
# read data
# data of the form [eval, average, best]
data = []
with open(data_file) as read_file:
    for line in read_file:
        line = line.split()
        if line != []:
            if line[0].isdigit():
                line[0] = int(line[0])
                line[1] = float(line[1])
                line[2] = float(line[2])
                data.append(line)

unique_evals = set()
for i in range(len(data)):
    unique_evals.add(data[i][0])

unique_evals = list(unique_evals)
dont_check_again = set()
unique_evals.sort()
print(unique_evals)
orginized = []
for i in range(len(unique_evals)):
    temp = []
    for item in data:
        if item[0] == unique_evals[i]:
            temp.append(item[2])
    orginized.append(temp)
print(orginized)
# fitness = []
# for item in orginized:
#     fitness.append(item[0])
# print(fitness)
# # get average overall fitness vs eval
# avg_overall_fitness_data = []
# best_overall_fitness_data = []
# dont_check_again = set()

# # visit every data point
# for i in range(len(data)):
#     if data[i][0] in dont_check_again:
#         continue
#     count = 1
#     running_avg_total = data[i][1]
#     running_best_total = data[i][2]
#     for j in range(i+1, len(data)):
#         # if the number of evals match track the fitness
#         if data[j][0] == data[i][0]:
#             count += 1
#             running_avg_total += data[j][1]
#             running_best_total += data[j][2]

#     avg_overall_fitness_data.append((data[i][0], running_avg_total/count))
#     best_overall_fitness_data.append((data[i][0], running_best_total/count))
#     dont_check_again.add(data[i][0])

# avg_overall_fitness_data.sort(key=lambda x: x[0])
# best_overall_fitness_data.sort(key=lambda x: x[0])

# avg_evals = []
# avg_fitness = []
# for i in range(len(avg_overall_fitness_data)):
#     avg_evals.append(avg_overall_fitness_data[i][0])
#     avg_fitness.append(avg_overall_fitness_data[i][1])

# best_evals = []
# best_fitness = []
# for i in range(len(best_overall_fitness_data)):
#     best_evals.append(best_overall_fitness_data[i][0])
#     best_fitness.append(best_overall_fitness_data[i][1])


# with open('../report_questions/q2_random_board_repair_ur_avg_data.txt', 'a+') as write_file:
#     for i in range(len(avg_fitness)):
#         write_file.write(str(avg_fitness[i]))
#     write_file.write('\n\n')
# write_file.close()

# with open('../report_questions/q2_random_board_repair_ur_best_data.txt', 'a+') as write_file:
#     for i in range(len(best_fitness)):
#         write_file.write(str(best_fitness[i]))
#     write_file.write('\n\n')
# write_file.close()


plt.boxplot(orginized, positions=unique_evals)
plt.title('Global Best Vs Evaluations')
plt.xlabel('Evaluations')
plt.ylabel('Normalized Fitness Score')
plt.show()
# plt.plot(5)

# plt.boxplot(orginized)

# plt.plot(best_evals, best_fitness, 'b', label='Best')
# plt.ylabel('Average Fitness')
# plt.xlabel('Number of Evaluations')
# plt.title('Best Vs. Average Fitness for Constraint E.A.\nwith Repair Function Acting on Assigned 14X14 Board')
# plt.legend()

# red_patch = mpatches.Patch(color='red')
# blue_patch = mpatches.Patch(color='blue')
# plt.show()
