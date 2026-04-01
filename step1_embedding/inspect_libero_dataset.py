import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from PIL import Image


def inspect_libero_object_no_noops(data_dir: str = "/data/users/kongyilun/code/RLinf/dataset") -> None:
    """Load libero_object_no_noops dataset and print its structure.

    Args:
        data_dir: Root directory where TFDS datasets (including libero_object_no_noops) are stored.
    """
    dataset_name = "libero_object_no_noops"

    print(f"Loading TFDS dataset '{dataset_name}' from data_dir='{data_dir}' ...")
    ds, ds_info = tfds.load(
        dataset_name,
        split="train",
        data_dir=data_dir,
        shuffle_files=False,
        with_info=True,
    )

    print("\n===== Dataset Info =====")
    print(ds_info)
    print("\nSplits:")
    for split_name, split_info in ds_info.splits.items():
        print(f"  - {split_name}: num_examples={split_info.num_examples}")

    print("\n===== One Example Structure (train split) =====")
    # instruction_set=[]
    # # task="pick up the orange juice and place it in the basket"
    # task="pick up the ketchup and place it in the basket"
    # for i,example in enumerate(ds):#.take(1)
    #     # print(len(example['steps']))
    #     for j,e in enumerate(example['steps']):
    #         print(e['episode_id'].numpy())
    #         break
            # if e['is_last'].numpy():
            #     if e['reward'].numpy() != 1:
            #         print(f'Failed Traj {i}')
                # print(f"Found is_last at step {j}")
                # print(f"Reward at is_last step: {e['reward'].numpy()}")
            # print(e['observation']['joint_state'][:5])
            # break
            # if j==5:
            #     # print(j)
            #     if e['language_instruction'].numpy().decode("utf-8") == task:
            #         tmp=e['observation']['image'].numpy()
            #         Image.fromarray(tmp.astype("uint8")).save(f"task_1/first_image_{i}.png")
            # # img.show()
            # print(tmp)
            # print(type(tmp))
            # instruction_set.append(e['language_instruction'].numpy().decode("utf-8"))
            # break
        # # only inspect the first example
        # break
        

    # print(set(instruction_set))
    # print(len(set(instruction_set)))


if __name__ == "__main__":
    inspect_libero_object_no_noops()
