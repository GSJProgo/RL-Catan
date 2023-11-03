# Define an array
my_array = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Define the indices for the parts you want to color
start_index = 2
end_index = 6

# Create an array of color-coded strings
colored_array = []
for i, num in enumerate(my_array):
    if start_index <= i <= end_index:
        colored_array.append("\u001b[48;5;<16>" + str(num))  # Red color (91) for the specified range
    else:
        colored_array.append(str(num))

# Join the colored parts to create a single string
colored_text = ' '.join(colored_array)

# Print the colored array
print(colored_text)