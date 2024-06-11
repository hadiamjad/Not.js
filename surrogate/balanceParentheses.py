def find_ending_index(string, start_index):
    stack = []
    opening_brackets = ["("]
    closing_brackets = [")"]
    index = start_index

    while index < len(string):
        char = string[index]

        if char in opening_brackets:
            stack.append(char)
        elif char in closing_brackets:
            # If stack is empty or the opening bracket doesn't match the closing bracket
            if len(stack) == 0 or opening_brackets.index(
                stack.pop()
            ) != closing_brackets.index(char):
                break

            # If stack becomes empty, return the current index
            if len(stack) == 0:
                return index

        index += 1

    # If the loop completes without finding the ending index
    return -1
