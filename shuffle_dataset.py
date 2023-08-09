import os
import random
import shutil

# Set the seed for reproducibility
random.seed(42)

data_dir = "data"
output_dir = "random_selected_data"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for dataset_folder in ["training_data", "test_data"]:
    
    dataset_path = os.path.join(data_dir, dataset_folder)
    output_dataset_path = os.path.join(output_dir, dataset_folder)
    print(output_dataset_path)
    os.makedirs(output_dir + "/" + dataset_folder)

    images_path = os.path.join(dataset_path, "images")
    masks_path = os.path.join(dataset_path, "masks")

    output_images_path = os.path.join(output_dataset_path, "images")
    output_masks_path = os.path.join(output_dataset_path, "masks")

    image_filenames = os.listdir(images_path)
    random.shuffle(image_filenames)

    num_samples = len(image_filenames) // 4  # Select 25% of the data

    selected_image_filenames = image_filenames[:num_samples]
    selected_mask_filenames = [filename.replace("_sat.jpg", "_mask.png") for filename in selected_image_filenames]

    for image_filename, mask_filename in zip(selected_image_filenames, selected_mask_filenames):
        image_src = os.path.join(images_path, image_filename)
        mask_src = os.path.join(masks_path, mask_filename)

        image_dst = os.path.join(output_images_path, image_filename)
        mask_dst = os.path.join(output_masks_path, mask_filename)

        os.makedirs(output_images_path, exist_ok=True)
        os.makedirs(output_masks_path, exist_ok=True)

        shutil.copy(image_src, image_dst)
        shutil.copy(mask_src, mask_dst)

    print("done")

print("Randomly selected data copied to", output_dir)
