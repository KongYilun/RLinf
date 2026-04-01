import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_datasets.core import dataset_builder
from tensorflow_datasets.core import read_only_builder as ro_builder


# 原数据所在的 TFDS data_dir
BASE_DATA_DIR = "/data/dataset"
BASE_NAME = "libero_object_no_noops"


class LiberoObjectNoNoopsWithId(dataset_builder.DatasetBuilder):
    """Virtual TFDS Builder: wraps libero_object_no_noops and adds episode_id."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Wraps libero_object_no_noops and adds `episode_id` to each step."
    }

    def __init__(self, base_data_dir: str = BASE_DATA_DIR, **kwargs):
        """
        Args:
            base_data_dir: 原始 libero_object_no_noops TFDS 已经生成好的 data_dir
                           (包含 /libero_object_no_noops/*/features.json 等)
        """
        # 不调用父类的 download_and_prepare 逻辑，只用它的基础属性管理
        super().__init__(**kwargs)
        self._base_data_dir = base_data_dir

        # 用 ReadOnlyBuilder 从磁盘恢复原始 builder
        self._base_builder = ro_builder.builder_from_files(
            BASE_NAME, data_dir=self._base_data_dir
        )

        # 用原 builder 的 DatasetInfo 作为基础
        # 注意：这里没有把 episode_id 正式写进 features，属于 runtime 字段
        self._info = self._base_builder.info
        # 给 full_name 改个名字，方便区分
        self._info._name = "libero_object_no_noops_with_id"  # 私有属性，调试用

    # ===== DatasetBuilder 抽象方法实现 =====

    def _info(self) -> tfds.core.DatasetInfo:
        # 返回我们包装后的 info（基本等于原 info）
        return self._info

    def _split_generators(self, dl_manager):
        """
        这里不真正使用 TFDS 的 download_and_prepare 机制，
        所以 SplitGenerators 只做一个“声明”，不会被实际调用。
        """
        # 直接使用底层 builder 的 splits 定义
        return [
            tfds.core.SplitGenerator(
                name=split_name, gen_kwargs={"split": split_name}
            )
            for split_name in self._base_builder.info.splits.keys()
        ]

    def _generate_examples(self, *args, **kwargs):
        """
        不会被调用：我们不使用 download_and_prepare，
        只通过 as_dataset 在线包装原始数据。
        """
        raise AssertionError(
            "LiberoObjectNoNoopsWithId does not generate its own data. "
            "Use .as_dataset() which wraps the existing libero_object_no_noops data."
        )
    # === 新增：只读 builder，不支持 download_and_prepare ===
    def _download_and_prepare(self, *args, **kwargs):
        raise AssertionError(
            "LiberoObjectNoNoopsWithId is read-only. "
            "Use existing libero_object_no_noops files via _as_dataset()."
        )

    # === 关键：实现 _as_dataset，而不是只重写 as_dataset ===
    def _as_dataset(self, split="train", **kwargs):
        """
        真正构造 dataset 的地方，DatasetBuilder.as_dataset 会调用这里。
        返回 episode-level RLDS：
          每个元素是一个 episode，其中 episode['steps'] 的每个 step
          都多了 int64 的 episode_id 字段。
        """
        # 直接使用 ReadOnlyBuilder 从磁盘加载原 dataset
        base_ds = self._base_builder.as_dataset(split=split, **kwargs)

        def _add_episode_id(ep_idx, episode):
            def _add_to_step(step):
                step["episode_id"] = tf.cast(ep_idx, tf.int64)
                return step

            steps_with_id = episode["steps"].map(
                _add_to_step, num_parallel_calls=tf.data.AUTOTUNE
            )
            episode["steps"] = steps_with_id
            return episode

        ds_with_id = base_ds.enumerate().map(
            _add_episode_id, num_parallel_calls=tf.data.AUTOTUNE
        )
        return ds_with_id

    # 可选：保留 as_dataset，直接调用父类实现（它会调用 _as_dataset）
    def as_dataset(self, split="train", **kwargs):
        return super().as_dataset(split=split, **kwargs)

    # # ===== 核心：as_dataset 包装原始 RLDS，给 steps 加 episode_id =====

    # def as_dataset(self, split="train", **kwargs):
    #     """
    #     返回一个 episode-level RLDS dataset：
    #       每个元素是一个 episode，其中 episode['steps'] 的每个 step
    #       都多了一个 int64 的 episode_id 字段 (表示该 episode 在该 split 中的顺序)。
    #     """
    #     # 直接使用 ReadOnlyBuilder 从磁盘加载原 dataset
    #     base_ds = self._base_builder.as_dataset(split=split, **kwargs)

    #     # base_ds：episode-level dataset (each element: {'steps': Dataset, 'episode_metadata': ...})

    #     def _add_episode_id(ep_idx, episode):
    #         """
    #         ep_idx: scalar int64, the index of the episode in this split
    #         episode: dict with key 'steps' (Dataset) and maybe 'episode_metadata'
    #         """
    #         def _add_to_step(step):
    #             # 在每一个 step 里加 episode_id 字段
    #             step["episode_id"] = tf.cast(ep_idx, tf.int64)
    #             return step

    #         # episode["steps"] 是一个 tf.data.Dataset (per-step)
    #         steps_with_id = episode["steps"].map(
    #             _add_to_step, num_parallel_calls=tf.data.AUTOTUNE
    #         )
    #         episode["steps"] = steps_with_id
    #         return episode

    #     # enumerate 生成连续的 episode index
    #     ds_with_id = base_ds.enumerate().map(
    #         _add_episode_id, num_parallel_calls=tf.data.AUTOTUNE
    #     )

    #     return ds_with_id