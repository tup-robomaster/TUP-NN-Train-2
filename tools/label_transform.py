import os

def transform_number(num):
    old_mapping = [
        "BGs", "B1b", "B2s", "B3s", "B4s", "B5s", "BOs", "BBs", "BBb",
        "RGs", "R1b", "R2s", "R3s", "R4s", "R5s", "ROs", "RBs", "RBb",
        "NGs", "N1b", "N2s", "N3s", "N4s", "N5s", "NOs", "NBs", "NBb",
        "PGs", "P1b", "P2s", "P3s", "P4s", "P5s", "POs", "PBs", "PBb"
    ]

    new_mapping = [
        "BGs", "B1s", "B2s", "B3s", "B4s", "B5s", "BOs", "BBs", "BGb",
        "B1b", "B2b", "B3b", "B4b", "B5b", "BOb", "BBb", "RGs", "R1s",
        "R2s", "R3s", "R4s", "R5s", "ROs", "RBs", "RGb", "R1b", "R2b",
        "R3b", "R4b", "R5b", "ROb", "RBb", "NGs", "N1s", "N2s", "N3s",
        "N4s", "N5s", "NOs", "NBs", "NGb", "N1b", "N2b", "N3b", "N4b",
        "N5b", "NOb", "NBb", "PGs", "P1s", "P2s", "P3s", "P4s", "P5s",
        "POs", "PBs", "PGb", "P1b", "P2b", "P3b", "P4b", "P5b", "POb",
        "PBb"
    ]

    return new_mapping.index(old_mapping[num])

def convert_old_to_new(file_path):

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i in range(len(lines)):
        line = lines[i].strip()
        if line:  # 跳过空行
            words = line.split()
            if words and words[0].isdigit():
                old_number = int(words[0])
                new_number = transform_number(old_number)
                print(old_number,"->",new_number)
                words[0] = str(new_number)
                lines[i] = ' '.join(words) + '\n'

    with open(file_path, 'w') as file:
        file.writelines(lines)

def process_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            convert_old_to_new(file_path)

if __name__ == "__main__":
    # 指定包含txt文件的文件夹路径
    folder_path = "./labels"  # 将此处替换为实际的文件夹路径

    # 调用函数处理文件夹中的所有txt文件
    process_files_in_folder(folder_path)
