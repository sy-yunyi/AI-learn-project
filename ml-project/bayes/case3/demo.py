import numpy as np

def load_text():
    fd = open('./data/stop_words_en.txt')
    lines_list = fd.readlines()
    file_data = []
    for line in lines_list:
        line = line.strip()
        file_data.append(line)
    print(len(file_data))

def demo_set():
    s = set([1,2,3])
    s1 = set([3,4])
    # print(s - s1)
    m = ['11','22','33','44','11','22']
    print(type(m))
if __name__ == '__main__':
    demo_set()