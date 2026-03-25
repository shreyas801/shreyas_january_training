'''nums = [1,2,3,4,5,6,7,8,9,10]
print(remove_even(nums))

Incorrect Output (observed):
[1, 3, 5, 7, 9, 10]

# Correct code:
nums = [1,2,3,4,5,6,7,8,9,10]
print(remove_even(nums))

Incorrect Output (observed):
[1, 3, 5, 7, 9, 10] '''

# Correct code:
def remove_even(numbers):
    return [n for n in numbers if n % 2 != 0]


nums = [1,2,3,4,5,6,7,8,9,10]
print(remove_even(nums))

