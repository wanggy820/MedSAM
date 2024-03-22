#!/usr/bin/python

filename = '/Users/wang/code/VV/VV-N-IM-SDK/mars/log/crypt/wfchat_20240308.xlog.log'
search_string = 'encrypt timestamp'

with open(filename, 'r', encoding='latin-1') as file:
    for line_number, line in enumerate(file, start=1):
        index = line.find(search_string)
        # print(line)
        if index != -1:
            start_index = index + 18
            end_index = start_index + 9
            substring = line[start_index:end_index]
            substring_as_int = int(substring)
            result = substring_as_int / 3600 - 54193
            result_as_int = int(result)
            if result_as_int > 12 or result_as_int < -12:
                print(f"{line, result_as_int}")


