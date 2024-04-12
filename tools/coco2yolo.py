import json
import os
import shutil

def coco_to_yolo(coco_file_path, input_image_folder, output_folder_path, target_category_ids):
    with open(coco_file_path, 'r') as coco_file:
        coco_data = json.load(coco_file)

    image_id_mapping = {image['id']: image for image in coco_data['images']}
    counter = 44423  # 从044423开始累加

    for annotation in coco_data['annotations']:
        if annotation['category_id'] in target_category_ids:
            image_info = image_id_mapping[annotation['image_id']]
            file_name = f"{counter:06d}.jpg"  # 生成新的图像文件名
            counter += 1

            width = image_info['width']
            height = image_info['height']

            x1, y1, x2, y2, x3, y3, x4, y4 = annotation['segmentation'][0]

            x1 /= width
            y1 /= height
            x2 /= width
            y2 /= height
            x3 /= width
            y3 /= height
            x4 /= width
            y4 /= height

            yolo_annotation = f"{annotation['category_id']} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"

            output_file_path = os.path.join(output_folder_path, file_name.replace('.jpg', '.txt'))
            output_image_path = os.path.join(output_folder_path, file_name)

            # 将YOLO格式标注写入文件
            with open(output_file_path, 'a') as output_file:
                output_file.write(yolo_annotation + '\n')
                print("saved ", output_file_path)

            # 将图像移动到新文件夹
            input_image_path = os.path.join(input_image_folder, image_info['file_name'])
            shutil.copy(input_image_path, output_image_path)
            print("moved ", input_image_path, " -> ", output_image_path)

if __name__ == "__main__":
    coco_file_path = "/home/nine-fish/datasets/armor_finnal/Armor_final_Dataset/annotations/instances_train2017.json"  # 替换为COCO标注文件的路径
    input_image_folder = "/home/nine-fish/datasets/armor_finnal/Armor_final_Dataset/images"  # 替换为原始图像文件夹路径
    output_folder_path = "/home/nine-fish/datasets/armor_finnal/output"  # 替换为输出YOLO格式标注和图像的文件夹路径
    target_category_ids = [0, 8, 10, 11, 12, 13, 14, 16, 24, 26, 27, 28, 29, 30, 32, 40, 42, 43, 44, 45, 46, 48, 56, 58, 59, 60, 61, 62]  # 替换为你想要转换的类别ID列表

    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    coco_to_yolo(coco_file_path, input_image_folder, output_folder_path, target_category_ids)
