from balanceParentheses import find_ending_index
import os


def replace_function_call(js_file, modified_js_file, line_number, column_number):
    # Determine the target file (either modified_js_file or js_file)
    target_file = modified_js_file if os.path.exists(modified_js_file) else js_file

    # Read the JavaScript code file
    with open(target_file, "r") as file:
        code_lines = file.readlines()

    # Replace the function call at the specified line and column
    line_index = line_number - 1
    column_index = column_number - 1
    line = code_lines[line_index]
    end_index = find_ending_index(line, column_index)

    if end_index != -1:
        function_call = line[column_index : end_index + 1]
        blockme_length = len("blockme()")
        function_call_length = len(function_call)

        if blockme_length > function_call_length:
            # Reduce the characters in blockme() to match the function call length
            replacement_content = "blockme"[: function_call_length - 2] + "()"
        elif blockme_length < function_call_length:
            # Add spaces inside the parenthesis of blockme() to match the function call length
            num_spaces = function_call_length - blockme_length
            replacement_content = "blockme(" + " " * num_spaces + ")"
        else:
            # Length of blockme() and function call are equal, use blockme() as it is
            replacement_content = "blockme()"

        # Apply the change to the line
        code_lines[line_index] = (
            line[:column_index] + replacement_content + line[end_index + 1 :]
        )

        # Write the modified code to the same file
        with open(modified_js_file, "w") as file:
            file.writelines(code_lines)
        return 0
    else:
        return -1
