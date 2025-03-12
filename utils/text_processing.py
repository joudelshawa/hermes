def remove_think_text(text):
    s = text.split("</think>")
    return s[1]