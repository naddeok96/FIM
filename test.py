def get_name_from_filename(filename):
    name = ""
    for c in reversed(list(filename[:-3])):
        if c == "/":
            break

        name = c + name
        
    return name
